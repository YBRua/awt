import os
import torch
import logging
import numpy as np
import torch.nn as nn

import lang_model
from model_mt_autoenc_cce import (
    TranslatorGeneratorModel,
    TranslatorDiscriminatorModel,
)
from sentence_transformers import SentenceTransformer

from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from utils import repackage_hidden, prettify_res_dict
from data import Corpus
from code_dataset import (
    CodeVocab,
    WikiTextVocabWrapper,
    CodeSearchNetDataset,
    CodeSearchNetProcessor,
    CSNWatermarkingCollator,
)

from typing import List


def parse_args_csn_sampling():
    parser = ArgumentParser()

    # dataset
    parser.add_argument('--vocab_source', type=str, default='wikitext')
    parser.add_argument('--data',
                        type=str,
                        default='data/CodeSearchNet',
                        help='location of the data corpus')
    parser.add_argument('--lang', type=str, default='java')
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--file_ids', nargs='+', type=int, default=0)
    parser.add_argument('--dataset_subsample_num', type=int, default=-1)

    # model and training
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--bptt', type=int, default=80, help='sequence length')

    parser.add_argument('--gen_path',
                        type=str,
                        default='gen.pt',
                        help='path to the generator')
    parser.add_argument('--disc_path',
                        type=str,
                        default='disc.pt',
                        help='path to the discriminator')

    # watermark msg and decoding
    parser.add_argument('--given_msg',
                        type=list,
                        default=[],
                        help='test against this msg only')
    parser.add_argument('--msg_len',
                        type=int,
                        default=8,
                        help='The length of the binary message')
    parser.add_argument('--msgs_num', type=int, default=3, help='generate msgs for test')
    parser.add_argument('--repeat_cycle',
                        type=int,
                        default=2,
                        help='Number of sentences to average')
    parser.add_argument('--msgs_segment', type=int, default=5, help='Long message')

    parser.add_argument('--bert_threshold',
                        type=float,
                        default=20,
                        help='Threshold on the bert distance')
    parser.add_argument('--samples_num', type=int, default=10, help='Decoder beam size')

    # language loss
    parser.add_argument('--use_lm_loss',
                        action='store_true',
                        help='whether to use language model loss')
    parser.add_argument('--lm_ckpt',
                        type=str,
                        default='./ckpts/WT2_lm.pt',
                        help='path to the fine tuned language model')

    # gumbel softmax arguments
    parser.add_argument('--gumbel_temp',
                        type=int,
                        default=0.5,
                        help='Gumbel softmax temprature')
    parser.add_argument('--gumbel_hard',
                        type=bool,
                        default=True,
                        help='whether to use one hot encoding in the forward pass')

    # language model params.
    parser.add_argument('--model',
                        type=str,
                        default='LSTM',
                        help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize_lm',
                        type=int,
                        default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid',
                        type=int,
                        default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3, help='number of layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth',
                        type=float,
                        default=0.3,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti_lm',
                        type=float,
                        default=0.15,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument(
        '--dropoute_lm',
        type=float,
        default=0.05,
        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument(
        '--wdrop',
        type=float,
        default=0,
        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')

    return parser.parse_args()


def setup_logger_eval_csn_bert():
    logger = logging.getLogger('CodeSearchNet Sampling Evaluation (BERT-Guided)')
    logger.setLevel(logging.INFO)

    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.INFO)
    # stream_formatter = logging.Formatter("%(message)s")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    filename = f'csn-eval-bert-{timestamp}.log'

    file_handler = logging.FileHandler(filename=f'./logs/{filename}',
                                       mode='a',
                                       encoding='utf-8')
    # file_formatter = logging.Formatter("[%(levelname)s %(asctime)s]: %(message)s")
    file_formatter = logging.Formatter("%(message)s")

    # stream_handler.setFormatter(stream_formatter)
    file_handler.setFormatter(file_formatter)

    # logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def get_jsonl_filenames(args) -> List[str]:
    dataset_dir = os.path.join(args.data, args.lang, 'final', 'jsonl', args.split)
    return [
        os.path.join(dataset_dir, f'{args.lang}_{args.split}_{i}.jsonl')
        for i in args.file_ids
    ]


def get_idx_from_logits(sequence, seq_len, bsz):
    sequence = sequence.view(seq_len, bsz, sequence.size(1))
    sequence = torch.softmax(sequence, dim=-1)
    sequence_idx = torch.argmax(sequence, dim=-1)
    return sequence_idx


def noisy_sampling(model_gen, sent_encoder_out, both_embeddings, data, samples_num,
                   gumbel_temp):
    candidates_emb = []
    candidates_one_hot = []
    candidates_soft_prob = []
    for i in range(0, samples_num):
        with torch.no_grad():
            sent_out_emb, sent_out_hot, sent_out_soft = model_gen.forward_sent_decoder(
                both_embeddings, data, sent_encoder_out, gumbel_temp)
            candidates_emb.append(sent_out_emb)
            candidates_one_hot.append(sent_out_hot)
            candidates_soft_prob.append(sent_out_soft)
    return candidates_emb, candidates_one_hot, candidates_soft_prob


def compare_msg_whole(msgs, msg_out, msg_len, msgs_segment):
    per_sample_corrects = (msgs == msg_out).long().sum(dim=1)
    return (per_sample_corrects == msg_len * msgs_segment).long().sum().item()


def compare_msg_bits(msgs, msg_out):
    return (msgs == msg_out).long().sum().item()


def evaluate(model_gen: TranslatorGeneratorModel,
             model_disc: TranslatorDiscriminatorModel, lm: lang_model.RNNModel,
             sbert_model: SentenceTransformer, dataloader: DataLoader, vocab: CodeVocab,
             device: torch.device, logger: logging.Logger, args):
    REAL_LABEL_VAL = 1
    FAKE_LABEL_VAL = 0

    model_gen.eval()
    model_disc.eval()
    sbert_model.eval()
    lm.eval()

    total_loss_lm = 0
    tot_count = 0
    correct_msg_count = 0
    tot_count_bits = 0
    correct_msg_count_bits = 0
    batch_count = 0
    l2_distances = 0
    bert_diff = [0 for i in range(0, args.samples_num)]
    long_msg_count = 0
    long_msg = np.zeros([1, args.msg_len * args.msgs_segment])
    long_msg_out = np.zeros([1, args.msg_len * args.msgs_segment])
    long_msg_out = torch.from_numpy(long_msg_out).float().cuda()
    long_msg = torch.from_numpy(long_msg).float().cuda()
    p_value = []
    real_correct = 0
    fake_correct = 0
    y_out = []
    y_label = []

    criterion_lm = nn.CrossEntropyLoss()
    for bid, batch in enumerate(tqdm(dataloader)):
        data, lengths, msgs = batch
        data = data.to(device)
        msgs = msgs.to(device)

        long_msg[:, long_msg_count * args.msg_len:long_msg_count * args.msg_len +
                 args.msg_len] = msgs

        if args.use_lm_loss:
            hidden = lm.init_hidden(bsz=1)

        with torch.no_grad():
            # watermark encoding
            # both_embeddings: (B, H) embedding of sentence + watermark
            # sent_encoder_out: (L, B, H) final hidden output of transformer encoder
            both_embeddings, sent_encoder_out = model_gen.forward_sent_encoder(
                data, msgs, args.gumbel_temp)

            # candidates_emb: (n_samples, L, B, H)
            # candidates_one_hot: (n_samples, L, B, V)
            # candidates_soft_prob: (n_samples, LB, V)
            candidates_emb, candidates_one_hot, candidates_soft_prob = noisy_sampling(
                model_gen, sent_encoder_out, both_embeddings, data, args.samples_num,
                args.gumbel_temp)
            data_emb = model_gen.forward_sent(data,
                                              msgs,
                                              args.gumbel_temp,
                                              only_embedding=True)
            real_out = model_disc(data_emb)

            label = torch.full((data.size(1), 1), REAL_LABEL_VAL)
            label = label.to(device)

            real_out_label = torch.round(torch.sigmoid(real_out))
            real_correct += (label == real_out_label).float().sum().item()
            y_label.append(label.detach().cpu().numpy().astype(int)[0, 0])
            y_out.append(real_out_label.detach().cpu().numpy().astype(int)[0, 0])

            # text decoding
            output_text_beams = []
            word_idx_beams = []
            for beam in range(0, args.samples_num):
                # word_idx: L, B
                word_idx = get_idx_from_logits(candidates_soft_prob[beam],
                                               data.size(0),
                                               bsz=1)
                word_idx_beams.append(word_idx)
                output_text = ''
                orig_text = ''
                for k in range(0, data.size(0)):
                    output_text += f'{vocab.get_token_by_id(word_idx[k, 0])} '
                    orig_text += f'{vocab.get_token_by_id(data[k, 0])} '
                output_text_beams.append(output_text)
                sentences = [output_text, orig_text]
                sbert_embs = sbert_model.encode(sentences)
                bert_diff[beam] = np.linalg.norm(sbert_embs[0] - sbert_embs[1])

            # get the best beam with non-zero diff
            best_beam_idx = -1
            beam_argsort = np.argsort(np.asarray(bert_diff))
            for beam in range(0, args.samples_num):
                if bert_diff[beam_argsort[beam]] > 0:
                    best_beam_idx = beam_argsort[beam]
                    break
            # if all distances are zero
            if best_beam_idx == -1:
                best_beam_idx = beam_argsort[0]

            # watermark decoding
            best_beam_data = word_idx_beams[best_beam_idx]
            best_beam_emb = model_gen.forward_sent(best_beam_data,
                                                   msgs,
                                                   args.gumbel_temp,
                                                   only_embedding=True)
            msg_out = model_gen.forward_msg_decode(best_beam_emb)

            fake_out = model_disc(candidates_emb[best_beam_idx].detach())
            label.fill_(FAKE_LABEL_VAL)
            fake_out_label = torch.round(torch.sigmoid(fake_out))
            fake_correct += (label == fake_out_label).float().sum().item()
            y_label.append(label.detach().cpu().numpy().astype(int)[0, 0])
            y_out.append(fake_out_label.detach().cpu().numpy().astype(int)[0, 0])
            # language loss
            if args.use_lm_loss:
                lm_targets = word_idx_beams[best_beam_idx][
                    1:candidates_one_hot[best_beam_idx].size(0)]
                lm_targets = lm_targets.view(lm_targets.size(0) * lm_targets.size(1), )
                lm_inputs = word_idx_beams[best_beam_idx][
                    0:candidates_one_hot[best_beam_idx].size(0) - 1]
                lm_out, hidden = lm(lm_inputs, hidden, decode=True)
                lm_loss = criterion_lm(lm_out, lm_targets)
                total_loss_lm += lm_loss.data
                hidden = repackage_hidden(hidden)

            if bert_diff[best_beam_idx] < args.bert_threshold:
                long_msg_out[:,
                             long_msg_count * args.msg_len:long_msg_count * args.msg_len +
                             args.msg_len] = msg_out
                l2_distances = l2_distances + bert_diff[best_beam_idx]
                # meteor_tot = meteor_tot + meteor_beams[best_beam_idx]
                logger.info('=' * 90)
                logger.info(f'sample #{batch_count}')
                logger.info('-' * 90)
                logger.info(f'gt  msg: {msgs.tolist()}')
                logger.info(f'dec msg: {torch.round(torch.sigmoid(msg_out)).tolist()}')
                correct_bits = compare_msg_bits(msgs, torch.round(torch.sigmoid(msg_out)))
                logger.info(f'correct bits: {int(correct_bits)}')
                logger.info(f'bert diff: {bert_diff[best_beam_idx]:.4f}')
                logger.info('-' * 90)
                logger.info(f'[ORIGINAL] {orig_text}')
                logger.info(f'[MODIFIED] {output_text_beams[best_beam_idx]}')
                logger.info('=' * 90)
            else:
                # meteor_pair = 1
                # meteor_tot = meteor_tot + meteor_pair
                msg_out_random = model_gen.forward_msg_decode(data_emb)
                long_msg_out[:,
                             long_msg_count * args.msg_len:long_msg_count * args.msg_len +
                             args.msg_len] = msg_out_random
        if ((batch_count != 0 and (batch_count + 1) % args.msgs_segment == 0)
                or args.msgs_segment == 1):
            long_msg_count = 0
            tot_count = tot_count + 1
            tot_count_bits += long_msg.shape[0] * long_msg.shape[1]
            long_msg_out = torch.round(torch.sigmoid(long_msg_out))
            similar_bits = compare_msg_bits(long_msg, long_msg_out)
            all_bits = long_msg.shape[0] * long_msg.shape[1]
            correct_msg_count += compare_msg_whole(long_msg, long_msg_out, args.msg_len,
                                                   args.msgs_segment)
            correct_msg_count_bits += similar_bits
            p_value.append(binomtest(similar_bits, all_bits, 0.5).pvalue)
        else:
            long_msg_count = long_msg_count + 1

        batch_count = batch_count + 1

    f1s = f1_score(y_label, y_out)
    p_value_smaller = sum(i < 0.05 for i in p_value)
    if args.use_lm_loss:
        total_loss_lm = total_loss_lm.item()

    return {
        'loss': total_loss_lm / batch_count,
        'msg_acc': correct_msg_count / tot_count,
        'bit_acc': correct_msg_count_bits / tot_count_bits,
        'l2dist': l2_distances / batch_count,
        'disc_fake_acc': fake_correct / batch_count,
        'disc_real_acc': real_correct / batch_count,
        'disc_f1': f1s,
        'p_value': np.mean(p_value),
        'p_value_inst': p_value_smaller / len(p_value),
    }


def main(args):
    # setting things up
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger = setup_logger_eval_csn_bert()
    logger.info(args)

    device = torch.device('cuda')

    # prepare data
    processor = CodeSearchNetProcessor()
    instances = processor.process_jsonls(get_jsonl_filenames(args))

    if args.vocab_source == 'wikitext':
        corpus = Corpus('./data/wikitext-2')
        vocab = WikiTextVocabWrapper(corpus.dictionary)
    else:
        _train_dataset = CodeSearchNetDataset.from_json(args.vocab_source)
        vocab = _train_dataset.vocab

    vocab_size = len(vocab)
    logger.info(f'vocab size: {vocab_size}')

    instances = [inst for inst in instances if len(inst.tokens) <= 120]
    logger.info(f'num of total instances: {len(instances)}')

    # random subsample for testing
    if args.dataset_subsample_num <= 0:
        subsampled = instances
    else:
        subsampled_idx = np.random.choice(len(instances),
                                          args.dataset_subsample_num,
                                          replace=False)
        subsampled = [instances[i] for i in subsampled_idx]
    logger.info(f'num of subsampled instances: {len(subsampled)}')

    dataset = CodeSearchNetDataset(subsampled, vocab)
    collator = CSNWatermarkingCollator(0, args.msg_len, 512)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator)

    dataset.dump_json('processed.json')

    # Load the best saved model.
    with open(args.gen_path, 'rb') as f:
        model_gen: TranslatorGeneratorModel
        model_gen, _, _, _ = torch.load(f)
    with open(args.disc_path, 'rb') as f:
        model_disc: TranslatorDiscriminatorModel
        model_disc, _, _, _ = torch.load(f)
    # load sbert and lm
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    if args.use_lm_loss:
        with open(args.lm_ckpt, 'rb') as f:
            pretrained_lm, _, _ = torch.load(f, map_location='cpu')
            lm = lang_model.RNNModel(args.model,
                                     vocab_size,
                                     args.emsize_lm,
                                     args.nhid,
                                     args.nlayers,
                                     args.dropout,
                                     args.dropouth,
                                     args.dropouti_lm,
                                     args.dropoute_lm,
                                     args.wdrop,
                                     tie_weights=True,
                                     loaded_model=pretrained_lm)

    model_gen.to(device)
    model_disc.to(device)
    sbert_model.to(device)
    lm.to(device)

    res = evaluate(model_gen, model_disc, lm, sbert_model, dataloader, vocab, device,
                   logger, args)

    res_str = prettify_res_dict(res, prefix=f'| {args.lang}-{args.split} ')
    logger.info(res_str)
    print(res_str)


if __name__ == '__main__':
    args = parse_args_csn_sampling()
    main(args)
