import os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer

import lang_model
from fb_semantic_encoder import BLSTMEncoder
from utils import repackage_hidden, prettify_res_dict
from model_mt_autoenc_cce import (
    TranslatorGeneratorModel,
    TranslatorDiscriminatorModel,
)
from code_dataset import (
    CodeVocab,
    DataInstance,
    CodeSearchNetDataset,
    CodeSearchNetProcessor,
    CSNWatermarkingCollator,
    build_dataset,
)

from typing import List


def parse_args_csn_training():
    parser = ArgumentParser('CodeSearchNet Training')
    # dataset arguments
    parser.add_argument('--data',
                        type=str,
                        default='data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--lang', type=str, default='java')
    parser.add_argument('--train_subsample_num', type=int, default=0)
    parser.add_argument('--eval_subsample_num', type=int, default=0)
    parser.add_argument('--allow_cached_dataset', action='store_true')

    parser.add_argument('--codebert', action='store_true')

    # training arguments
    parser.add_argument('--lr', type=float, default=0.00003, help='initial learning rate')
    parser.add_argument('--disc_lr',
                        type=float,
                        default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--optimizer',
                        type=str,
                        default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--wdecay',
                        type=float,
                        default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--when',
                        nargs="+",
                        type=int,
                        default=[-1],
                        help='When (which epochs) to divide the learning rate by 10')
    parser.add_argument('--epochs', type=int, default=8000, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_false', help='use CUDA')
    parser.add_argument('--nonmono', type=int, default=5, help='random seed')

    # model arguments
    parser.add_argument('--emsize', type=int, default=512, help='size of word embeddings')
    parser.add_argument('--dropout_transformer',
                        type=float,
                        default=0.1,
                        help='dropout applied to transformer layers (0 = no dropout)')
    parser.add_argument('--dropouti',
                        type=float,
                        default=0.1,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument(
        '--dropoute',
        type=float,
        default=0.1,
        help='dropout to remove words from embedding layer (0 = no dropout)')

    # checkpointing
    parser.add_argument('--save', type=str, default='./ckpts/csn_model')
    parser.add_argument('--resume', type=str, default='', help='path of model to resume')
    parser.add_argument('--save_interval',
                        type=int,
                        default=20,
                        help='saving models regualrly')

    # message arguments
    parser.add_argument('--msg_len',
                        type=int,
                        default=64,
                        help='The length of the binary message')
    parser.add_argument('--msgs_num',
                        type=int,
                        default=3,
                        help='The total number of messages')
    parser.add_argument('--msg_in_mlp_layers',
                        type=int,
                        default=1,
                        help='message encoding FC layers number')
    parser.add_argument('--msg_in_mlp_nodes',
                        type=list,
                        default=[],
                        help='nodes in the MLP of the message')

    # transformer arguments
    parser.add_argument('--attn_heads',
                        type=int,
                        default=4,
                        help='The number of attention heads in the transformer')
    parser.add_argument('--encoding_layers',
                        type=int,
                        default=3,
                        help='The number of encoding layers')
    parser.add_argument(
        '--shared_encoder',
        type=bool,
        default=True,
        help='If the message encoder and language encoder will share weights')

    # adv. transformer arguments
    parser.add_argument('--adv_attn_heads',
                        type=int,
                        default=4,
                        help='The number of attention heads in the adversary transformer')
    parser.add_argument('--adv_encoding_layers',
                        type=int,
                        default=3,
                        help='The number of encoding layers in the adversary transformer')

    # gumbel softmax arguments
    parser.add_argument('--gumbel_temp',
                        type=int,
                        default=0.5,
                        help='Gumbel softmax temprature')

    # Adam optimizer arguments
    parser.add_argument('--scheduler',
                        action='store_true',
                        help='whether to schedule the lr')
    parser.add_argument('--warm_up',
                        type=int,
                        default=6000,
                        help='number of linear warm up steps')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1 parameter')
    parser.add_argument('--beta2', type=float, default=0.98, help='Adam beta2 parameter')
    parser.add_argument('--eps', type=float, default=1e-9, help='Adam eps parameter')

    # GAN arguments
    parser.add_argument('--msg_weight',
                        type=float,
                        default=25,
                        help='The factor multiplied with the message loss')

    # GAN arguments
    parser.add_argument('--discr_interval',
                        type=int,
                        default=1,
                        help='when to update the discriminator')
    parser.add_argument(
        '--autoenc_path',
        type=str,
        default='',
        help='path of the autoencoder path to use as init to the generator')
    parser.add_argument('--gen_weight',
                        type=float,
                        default=2,
                        help='The factor multiplied with the gen loss')

    # fb InferSent semantic loss
    parser.add_argument('--use_semantic_loss',
                        action='store_true',
                        help='whether to use semantic loss')
    parser.add_argument('--glove_path',
                        type=str,
                        default='sent_encoder/GloVe/glove.840B.300d.txt',
                        help='path to glove embeddings')
    parser.add_argument('--infersent_path',
                        type=str,
                        default='sent_encoder/infersent2.pkl',
                        help='path to the trained sentence semantic model')
    parser.add_argument('--sem_weight',
                        type=float,
                        default=40,
                        help='The factor multiplied with the semantic loss')

    # language loss
    parser.add_argument('--use_lm_loss',
                        action='store_true',
                        help='whether to use language model loss')
    parser.add_argument('--lm_weight',
                        type=float,
                        default=1,
                        help='The factor multiplied with the lm loss')
    parser.add_argument('--lm_ckpt',
                        type=str,
                        default='ckpts/WT2_lm.pt',
                        help='path to the fine tuned language model')

    # lang model params.
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
                        default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument(
        '--dropoute_lm',
        type=float,
        default=0.1,
        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument(
        '--wdrop',
        type=float,
        default=0,
        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')

    # reconstruction loss
    parser.add_argument('--use_reconst_loss',
                        action='store_true',
                        help='whether to use language reconstruction loss')
    parser.add_argument('--reconst_weight',
                        type=float,
                        default=1,
                        help='The factor multiplied with the reconstruct loss')

    return parser.parse_args()


def setup_logger_eval_csn_bert():
    logger = logging.getLogger('CodeSearchNet AWT Training')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    filename = f'csn-train-{timestamp}.log'

    file_handler = logging.FileHandler(filename=f'./logs/{filename}',
                                       mode='a',
                                       encoding='utf-8')
    file_formatter = logging.Formatter("[%(levelname)s %(asctime)s]: %(message)s")
    # file_formatter = logging.Formatter("%(message)s")

    stream_handler.setFormatter(stream_formatter)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def get_jsonl_filenames(args, split: str, chunk_ids: List[int]) -> List[str]:
    dataset_dir = os.path.join(args.data, args.lang, 'final', 'jsonl', split)
    return [
        os.path.join(dataset_dir, f'{args.lang}_{split}_{i}.jsonl') for i in chunk_ids
    ]


def resume_models(file_prefix: str):
    with open(f'{file_prefix}_gen.pt', 'rb') as f:
        gen_ckpt = torch.load(f)
        model_gen, criterion, criterion_recon, optimizer_gen = gen_ckpt
    with open(f'{file_prefix}_disc.pt', 'rb') as f:
        discr_ckpt = torch.load(f)
        model_discr, criterion, criterion_recon, optimizer_discr = discr_ckpt

    return {
        'model_gen': model_gen,
        'optimizer_gen': optimizer_gen,
        'model_disc': model_discr,
        'optimizer_disc': optimizer_discr,
        'criterion': criterion,
        'criterion_reconst': criterion_recon,
    }


def save_models(file_prefix: str, model_gen: TranslatorGeneratorModel,
                model_disc: TranslatorDiscriminatorModel,
                optimizer_gen: torch.optim.Optimizer,
                optimizer_disc: torch.optim.Optimizer, criterion: torch.nn.Module,
                criterion_recon: torch.nn.Module):
    with open(f'{file_prefix}_gen.pt', 'wb') as f:
        torch.save((model_gen, criterion, criterion_recon, optimizer_gen), f)
    with open(f'{file_prefix}_disc.pt', 'wb') as f:
        torch.save((model_disc, criterion, criterion_recon, optimizer_disc), f)


class LRScheduler:
    def __init__(self, d_model: int, warm_up: int):
        self._step_num = 1
        self.d_model = d_model
        self.warm_up = warm_up

    def get_lr(self):
        lr = np.power(self.d_model, -0.8) * min(
            np.power(self._step_num, -0.5), self._step_num * np.power(self.warm_up, -1.5))
        return lr

    def step(self):
        self._step_num += 1


def instance_filter(instance: DataInstance):
    return len(instance.tokens) <= 85


REAL_LABEL_VAL = 1
FAKE_LABEL_VAL = 0


def evaluate(eid: int, model_gen: TranslatorGeneratorModel,
             model_disc: TranslatorDiscriminatorModel, lm: lang_model.RNNModel,
             sent_encoder: BLSTMEncoder, dataloader: DataLoader, criterion: nn.Module,
             criterion_sem: nn.Module, criterion_lm: nn.Module,
             criterion_recon: nn.Module, device: torch.device, args):
    model_gen.eval()
    model_disc.eval()
    if args.use_semantic_loss:
        sent_encoder.eval()
    if args.use_lm_loss:
        lm.eval()

    total_loss_gen = 0
    total_loss_disc = 0
    total_loss_msg = 0
    total_loss_sem = 0
    total_loss_reconst = 0
    total_loss_lm = 0
    total_msg_acc = 0

    progress = tqdm(dataloader)
    progress.set_description('| eval |')
    batch_count = 0
    with torch.no_grad():
        for bid, batch in enumerate(progress):
            token_ids, lengths, msgs, src_padding_mask = batch

            token_ids = token_ids.to(device)
            msgs = msgs.to(device)
            src_padding_mask = src_padding_mask.to(device)

            B = token_ids.size(1)

            if args.use_lm_loss:
                hidden = lm.init_hidden(B)

            # generate a batch of fake (edited) sequence
            fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent(
                token_ids, msgs, args.gumbel_temp, src_key_padding_mask=src_padding_mask)
            msg_out = model_gen.forward_msg_decode(fake_data_emb, src_padding_mask)
            msg_preds = torch.round(torch.sigmoid(msg_out))
            total_msg_acc += (msg_preds == msgs).float().mean().item()

            # get prediction (and the loss) of the discriminator on the real sequence
            # first get the embeddings of the original sentences
            data_emb = model_gen.forward_sent(token_ids,
                                              msgs,
                                              args.gumbel_temp,
                                              only_embedding=True,
                                              src_key_padding_mask=src_padding_mask)
            real_out = model_disc(data_emb, src_padding_mask)
            label = torch.full((token_ids.size(1), 1), REAL_LABEL_VAL)
            label = label.to(device)
            disc_loss_real = criterion(real_out, label.float())

            # get prediction (and the loss) of the discriminator on the fake sequence
            fake_out = model_disc(fake_data_emb.detach(), src_padding_mask)
            label.fill_(FAKE_LABEL_VAL)
            disc_loss_fake = criterion(fake_out, label.float())
            disc_loss = disc_loss_real + disc_loss_fake

            # generator loss
            label.fill_(REAL_LABEL_VAL)
            loss_gen_adv = args.gen_weight * criterion(fake_out, label.float())

            # semantic loss
            if args.use_semantic_loss:
                if args.codebert:
                    cb_attn_mask = 1 - src_padding_mask.long()
                    orig_sem_emb = sent_encoder(
                        token_ids.permute(1,
                                          0), attention_mask=cb_attn_mask).pooler_output
                    L, B, V = fake_one_hot.shape
                    fake_input_emb = fake_one_hot.view(L * B, V)\
                        .mm(sent_encoder.embeddings.word_embeddings.weight)\
                        .view(L, B, -1).permute(1, 0, 2)
                    fake_sem_emb = sent_encoder(
                        inputs_embeds=fake_input_emb,
                        attention_mask=cb_attn_mask).pooler_output
                else:
                    orig_sem_emb = sent_encoder.forward_encode(token_ids, lengths)
                    fake_sem_emb = sent_encoder.forward_encode(fake_one_hot,
                                                               lengths,
                                                               one_hot=True)
                sem_loss = args.sem_weight * criterion_sem(orig_sem_emb, fake_sem_emb)
                total_loss_sem += sem_loss.item()

            # msg loss of the generator
            msg_loss = args.msg_weight * criterion(msg_out, msgs.float())

            # lm loss of the generator
            if args.use_lm_loss:
                lm_targets = fake_one_hot[1:fake_one_hot.size(0)]
                lm_targets = torch.argmax(lm_targets, dim=-1)
                lm_targets = lm_targets.view(lm_targets.size(0) * lm_targets.size(1), )
                lm_inputs = fake_one_hot[0:fake_one_hot.size(0) - 1]
                lm_out, hidden = lm.forward_padded(lm_inputs,
                                                   hidden,
                                                   lengths - 1,
                                                   decode=True,
                                                   one_hot=True)
                lm_loss = args.lm_weight * criterion_lm(lm_out, lm_targets)
                total_loss_lm += lm_loss.item()
                hidden = repackage_hidden(hidden)

            # reconstruction loss
            reconst_loss = args.reconst_weight * criterion_recon(
                fake_data_prob, token_ids.view(-1))
            total_loss_reconst += reconst_loss.item()

            total_loss_gen += loss_gen_adv.item()
            total_loss_disc += disc_loss.item()
            total_loss_msg += msg_loss.item()
            batch_count = batch_count + 1

    return {
        'epoch': eid,
        'msg_acc': total_msg_acc / batch_count,
        'gen_loss': total_loss_gen / batch_count,
        'msg_loss': total_loss_msg / batch_count,
        'recon_loss': total_loss_reconst / batch_count,
        'disc_loss': total_loss_disc / batch_count,
        'sem_loss': total_loss_sem / batch_count,
        'lm_loss': total_loss_lm / batch_count
    }


def train(eid: int, model_gen: TranslatorGeneratorModel,
          model_disc: TranslatorDiscriminatorModel, lm: lang_model.RNNModel,
          sent_encoder: BLSTMEncoder, optimizer_gen: optim.Optimizer,
          optimizer_disc: optim.Optimizer, scheduler_gen: LRScheduler,
          scheduler_disc: LRScheduler, dataloader: DataLoader, criterion: nn.Module,
          criterion_sem: nn.Module, criterion_lm: nn.Module, criterion_reconst: nn.Module,
          device: torch.device, args):
    model_gen.train()
    model_disc.train()
    if args.use_semantic_loss:
        sent_encoder.train()
        # set parameters trainable to false.
        for p in sent_encoder.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in the generator update
    if args.use_lm_loss:
        lm.train()
        # set parameters trainable to false.
        for p in lm.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in the generator update
        hidden = None

    total_loss_gen = 0
    total_loss_msg = 0
    total_loss_disc = 0
    total_loss_recon = 0
    total_loss_sem = 0
    total_loss_lm = 0
    total_msg_acc = 0

    progress = tqdm(dataloader)
    for bid, batch in enumerate(progress):
        token_ids, lengths, msgs, src_padding_mask = batch
        token_ids = token_ids.to(device)
        msgs = msgs.to(device)
        src_padding_mask = src_padding_mask.to(device)
        B = token_ids.size(1)

        if args.use_lm_loss:
            if hidden is None:
                hidden = lm.init_hidden(B)
            hidden = repackage_hidden(hidden)

        # optimizer
        optimizer_gen.zero_grad()
        optimizer_disc.zero_grad()
        # update lr
        if args.scheduler:
            optimizer_gen.param_groups[0]['lr'] = scheduler_gen.get_lr()
            optimizer_disc.param_groups[0]['lr'] = scheduler_disc.get_lr()

        # discriminator network update
        # maximize log(D(x)) + log(1 - D(G(z)))
        label = torch.full((B, 1), REAL_LABEL_VAL)
        label = label.to(device)

        # log(D(x))
        data_emb = model_gen.forward_sent(token_ids,
                                          msgs,
                                          args.gumbel_temp,
                                          only_embedding=True,
                                          src_key_padding_mask=src_padding_mask)
        real_out = model_disc(data_emb, src_key_padding_mask=src_padding_mask)
        loss_disc_real = criterion(real_out, label.float())
        loss_disc_real.backward()

        # log(1 - D(G(z)))
        fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent(
            token_ids, msgs, args.gumbel_temp, src_key_padding_mask=src_padding_mask)
        fake_out = model_disc(fake_data_emb.detach(),
                              src_key_padding_mask=src_padding_mask)
        label.fill_(FAKE_LABEL_VAL)
        loss_disc_fake = criterion(fake_out, label.float())
        loss_disc_fake.backward()
        # add the gradients
        loss_disc = loss_disc_real + loss_disc_fake
        # update the discriminator
        if bid % args.discr_interval == 0 and bid > 0:
            optimizer_disc.step()

        # generator network update
        # maximize log(D(G(z)))
        # fool discriminator to misclassify fake data as real
        label.fill_(REAL_LABEL_VAL)
        fake_out2 = model_disc(fake_data_emb, src_key_padding_mask=src_padding_mask)
        loss_gen_adv = args.gen_weight * criterion(fake_out2, label.float())

        # decode watermark
        msg_out = model_gen.forward_msg_decode(fake_data_emb,
                                               src_key_padding_mask=src_padding_mask)
        loss_msg = args.msg_weight * criterion(msg_out, msgs.float())
        msg_preds = torch.round(torch.sigmoid(msg_out))
        total_msg_acc += (msgs == msg_preds).float().mean().item()
        # reconstruct input sequence
        loss_recon = args.reconst_weight * criterion_reconst(fake_data_prob,
                                                             token_ids.view(-1))

        if args.use_semantic_loss:
            # Compute sentence embedding #
            if args.codebert:
                cb_attn_mask = 1 - src_padding_mask.long()
                orig_sent_emb = sent_encoder(token_ids.permute(1, 0),
                                             attention_mask=cb_attn_mask).pooler_output
                L, B, V = fake_one_hot.shape
                fake_input_emb = fake_one_hot.view(L * B, V)\
                    .mm(sent_encoder.embeddings.word_embeddings.weight)\
                    .view(L, B, -1).permute(1, 0, 2)
                fake_sent_emb = sent_encoder(inputs_embeds=fake_input_emb,
                                             attention_mask=cb_attn_mask).pooler_output
            else:
                orig_sent_emb = sent_encoder.forward_encode(token_ids, lengths)
                fake_sent_emb = sent_encoder.forward_encode(fake_one_hot,
                                                            lengths,
                                                            one_hot=True)
            loss_sem = args.sem_weight * criterion_sem(orig_sent_emb, fake_sent_emb)
            loss_gen = (loss_gen_adv + loss_msg + loss_sem + loss_recon)
            total_loss_sem += loss_sem.item()
        else:
            loss_gen = (loss_gen_adv + loss_msg + loss_recon)

        if args.use_lm_loss:
            lm_targets = fake_one_hot[1:fake_one_hot.size(0)]
            lm_targets = torch.argmax(lm_targets, dim=-1)
            lm_targets = lm_targets.view(lm_targets.size(0) * lm_targets.size(1), )
            lm_inputs = fake_one_hot[0:fake_one_hot.size(0) - 1]
            lm_out, hidden = lm.forward_padded(lm_inputs,
                                               hidden,
                                               lengths - 1,
                                               decode=True,
                                               one_hot=True)
            lm_loss = args.lm_weight * criterion_lm(lm_out, lm_targets)
            loss_gen = loss_gen + lm_loss
            total_loss_lm += lm_loss.item()

        # update the generator
        loss_gen.backward()
        optimizer_gen.step()

        total_loss_recon += loss_recon.item()
        total_loss_gen += loss_gen_adv.item()
        total_loss_msg += loss_msg.item()
        total_loss_disc += loss_disc.item()

        # report progress
        avg_loss_gen = total_loss_gen / (bid + 1)
        avg_loss_msg = total_loss_msg / (bid + 1)
        avg_loss_disc = total_loss_disc / (bid + 1)
        avg_loss_recon = total_loss_recon / (bid + 1)
        avg_loss_sem = total_loss_sem / (bid + 1)
        avg_loss_lm = total_loss_lm / (bid + 1)
        avg_msg_acc = total_msg_acc / (bid + 1)

        progress.set_description(f'| train {eid:3d} '
                                 f'| acc {avg_msg_acc:.4f}'
                                 f'| l_gen {avg_loss_gen:.4f} '
                                 f'| l_msg {avg_loss_msg:.4f} '
                                 f'| l_recon {avg_loss_recon:.4f} '
                                 f'| l_disc {avg_loss_disc:.4f} '
                                 f'| l_sem {avg_loss_sem:.4f} '
                                 f'| l_lm {avg_loss_lm:.4f} ')

        # update schedulers
        if args.scheduler:
            scheduler_gen.step()
            scheduler_disc.step()

    return {
        'epoch': eid,
        'msg_acc': avg_msg_acc,
        'gen_loss': total_loss_gen / (bid + 1),
        'msg_loss': total_loss_msg / (bid + 1),
        'recon_loss': total_loss_recon / (bid + 1),
        'disc_loss': total_loss_disc / (bid + 1),
        'sem_loss': total_loss_sem / (bid + 1),
        'lm_loss': total_loss_lm / (bid + 1)
    }


def main(args):
    # setting things up
    logger = setup_logger_eval_csn_bert()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.codebert:
        roberta_tokenizer: RobertaTokenizer
        roberta_tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    # loading data
    if args.codebert:
        train_cache_path = os.path.join(
            args.data, args.lang, f'train_{args.train_subsample_num}_codebert.json')
        valid_cache_path = os.path.join(args.data, args.lang,
                                        f'valid_{args.eval_subsample_num}_codebert.json')
        test_cache_path = os.path.join(args.data, args.lang,
                                       f'test_{args.eval_subsample_num}_codebert.json')
    else:
        vocab_cache_path = os.path.join(args.data, args.lang,
                                        f'vocab_{args.train_subsample_num}.json')
        train_cache_path = os.path.join(args.data, args.lang,
                                        f'train_{args.train_subsample_num}.json')
        valid_cache_path = os.path.join(args.data, args.lang,
                                        f'valid_{args.eval_subsample_num}.json')
        test_cache_path = os.path.join(args.data, args.lang,
                                       f'test_{args.eval_subsample_num}.json')

    train_dataset = None
    valid_dataset = None
    test_dataset = None
    # assumes atomicity of train, valid and test cache files
    if args.allow_cached_dataset and os.path.exists(train_cache_path):
        train_dataset = CodeSearchNetDataset.from_json(train_cache_path)
        valid_dataset = CodeSearchNetDataset.from_json(valid_cache_path)
        test_dataset = CodeSearchNetDataset.from_json(test_cache_path)
        if args.codebert:
            vocab = roberta_tokenizer
        else:
            vocab = CodeVocab().load(vocab_cache_path)

    if any([train_dataset is None, valid_dataset is None, test_dataset is None]):
        processor = CodeSearchNetProcessor()
        train_files = get_jsonl_filenames(args, 'train', list(range(16)))
        valid_files = get_jsonl_filenames(args, 'valid', [0])
        test_files = get_jsonl_filenames(args, 'test', [0])

        if args.codebert:
            train_instances = processor.process_jsonls_codebert(
                train_files, roberta_tokenizer, instance_filter=instance_filter)
            valid_instances = processor.process_jsonls_codebert(
                valid_files, roberta_tokenizer, instance_filter=instance_filter)
            test_instances = processor.process_jsonls_codebert(
                test_files, roberta_tokenizer, instance_filter=instance_filter)
        else:
            train_instances = processor.process_jsonls(train_files,
                                                       instance_filter=instance_filter)
            valid_instances = processor.process_jsonls(valid_files,
                                                       instance_filter=instance_filter)
            test_instances = processor.process_jsonls(test_files,
                                                      instance_filter=instance_filter)

        if args.train_subsample_num > 0:
            subsampled_idx = np.random.choice(len(train_instances),
                                              min(args.train_subsample_num,
                                                  len(train_instances)),
                                              replace=False)
            train_instances = [train_instances[i] for i in subsampled_idx]
        if args.eval_subsample_num > 0:
            subsampled_idx = np.random.choice(len(valid_instances),
                                              min(args.eval_subsample_num,
                                                  len(valid_instances)),
                                              replace=False)
            valid_instances = [valid_instances[i] for i in subsampled_idx]
            subsampled_idx = np.random.choice(len(test_instances),
                                              min(args.eval_subsample_num,
                                                  len(test_instances)),
                                              replace=False)
            test_instances = [test_instances[i] for i in subsampled_idx]

        if args.codebert:
            vocab = roberta_tokenizer
        else:
            vocab = processor.build_vocabulary_on_instances(train_instances)

        train_dataset = CodeSearchNetDataset(build_dataset(train_instances, vocab))
        valid_dataset = CodeSearchNetDataset(build_dataset(valid_instances, vocab))
        test_dataset = CodeSearchNetDataset(build_dataset(test_instances, vocab))

        train_dataset.dump_json(train_cache_path)
        valid_dataset.dump_json(valid_cache_path)
        test_dataset.dump_json(test_cache_path)

    logger.info(f'train dataset size: {len(train_dataset)}')
    logger.info(f'valid dataset size: {len(valid_dataset)}')
    logger.info(f'test dataset size: {len(test_dataset)}')
    vocab_size = 50265 if args.codebert else len(vocab)
    pad_idx = roberta_tokenizer.pad_token_id if args.codebert else vocab.pad_idx
    logger.info(f'vocab size: {vocab_size}')

    collator = CSNWatermarkingCollator(padding_idx=pad_idx,
                                       n_bits=args.msg_len,
                                       require_masks=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=80,
                              shuffle=True,
                              collate_fn=collator)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=80,
                              shuffle=False,
                              collate_fn=collator)
    test_loader = DataLoader(test_dataset,
                             batch_size=80,
                             shuffle=False,
                             collate_fn=collator)

    # constructing models
    scheduler_gen = LRScheduler(d_model=args.emsize, warm_up=args.warm_up)
    scheduler_disc = LRScheduler(d_model=args.emsize, warm_up=args.warm_up)

    if args.autoenc_path != '':
        with open(args.autoenc_path, 'rb') as f:
            autoenc_model, _, _ = torch.load(f)
    else:
        autoenc_model = None

    if args.resume:
        logger.info(f'resuming from {args.resume}')
        saved_dict = resume_models(args.resume)
        model_gen = saved_dict['model_gen']
        model_disc = saved_dict['model_disc']
    else:
        model_gen = TranslatorGeneratorModel(vocab_size, args.emsize, args.msg_len,
                                             args.msg_in_mlp_layers,
                                             args.msg_in_mlp_nodes, args.encoding_layers,
                                             args.dropout_transformer, args.dropouti,
                                             args.dropoute, True, args.shared_encoder,
                                             args.attn_heads, autoenc_model)
        if args.codebert:
            codebert_model = RobertaModel.from_pretrained('microsoft/codebert-base')
            model_gen.embeddings.load_state_dict(
                codebert_model.embeddings.word_embeddings.state_dict())

        model_disc = TranslatorDiscriminatorModel(args.emsize, args.adv_encoding_layers,
                                                  args.dropout_transformer,
                                                  args.adv_attn_heads, args.dropouti)
        for p in model_disc.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    criterion = nn.BCEWithLogitsLoss()
    criterion_sem = nn.L1Loss()
    criterion_lm = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion_recon = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # semantic model
    if args.use_semantic_loss:
        if args.codebert:
            sent_encoder = RobertaModel.from_pretrained('microsoft/codebert-base')
        else:
            word2idx = vocab.word2idx
            idx2word = vocab.idx2word

            sent_encoder = BLSTMEncoder(word2idx, idx2word, args.glove_path)
            encoder_state = torch.load(args.infersent_path)
            state = sent_encoder.state_dict()
            for k in encoder_state:
                if k in state:
                    state[k] = encoder_state[k]
            sent_encoder.load_state_dict(state)
    else:
        sent_encoder = None

    # language model
    if args.use_lm_loss:
        with open(args.lm_ckpt, 'rb') as f:
            pretrained_lm, _, _ = torch.load(f)
            lm = lang_model.RNNModel(args.model, vocab_size, args.emsize_lm, args.nhid,
                                     args.nlayers, args.dropout, args.dropouth,
                                     args.dropouti_lm, args.dropoute_lm, args.wdrop, True,
                                     pretrained_lm)
        del pretrained_lm
    else:
        lm = None

    # put everything on GPU
    device = torch.device('cuda')
    model_gen = model_gen.to(device)
    model_disc = model_disc.to(device)
    if args.use_semantic_loss:
        sent_encoder = sent_encoder.to(device)
    if args.use_lm_loss:
        lm = lm.to(device)

    params = list(model_gen.parameters()) + list(model_disc.parameters())
    params_gen = model_gen.parameters()
    params_disc = model_disc.parameters()

    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0]
                       for x in params if x.size())
    logger.info(f'total parameters: {total_params:,}')

    # construct optimizers
    # if args.resume:
    #     optimizer_gen = saved_dict['optimizer_gen']
    #     optimizer_disc = saved_dict['optimizer_disc']
    #     if args.scheduler:
    #         optimizer_gen.param_groups[0]['lr'] = scheduler_gen.get_lr()
    #         optimizer_disc.param_groups[0]['lr'] = scheduler_disc.get_lr()
    #     else:
    #         optimizer_gen.param_groups[0]['lr'] = args.lr
    #         optimizer_disc.param_groups[0]['lr'] = args.lr
    # else:
    if args.optimizer == 'sgd':
        # optimizer_gen = torch.optim.SGD(params_gen,
        #                                 lr=args.lr,
        #                                 weight_decay=args.wdecay)
        # optimizer_disc = torch.optim.SGD(params_disc,
        #                                  lr=args.lr,
        #                                  weight_decay=args.wdecay)
        raise NotImplementedError('sgd not implemented yet')
    elif args.optimizer == 'adam':
        optimizer_gen = torch.optim.Adam(
            params_gen,
            lr=scheduler_gen.get_lr() if args.scheduler else args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.wdecay)
        optimizer_disc = torch.optim.Adam(
            params_disc,
            lr=scheduler_disc.get_lr() if args.scheduler else args.disc_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.wdecay)
    else:
        raise RuntimeError(f'unknown optimizer: {args.optimizer}')

    # training loop
    stored_loss = float('inf')
    stored_loss_msg = float('inf')
    stored_loss_text = float('inf')
    for eid in range(args.epochs):
        train_res = train(eid, model_gen, model_disc, lm, sent_encoder, optimizer_gen,
                          optimizer_disc, scheduler_gen, scheduler_disc, train_loader,
                          criterion, criterion_sem, criterion_lm, criterion_recon, device,
                          args)
        logger.info(prettify_res_dict(train_res, prefix='| train '))

        if 't0' in optimizer_gen.param_groups[0]:
            tmp = {}
            for prm in model_gen.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer_gen.state[prm]['ax'].clone()

            tmp_disc = {}
            for prm in model_disc.parameters():
                tmp_disc[prm] = prm.data.clone()
                prm.data = optimizer_disc.state[prm]['ax'].clone()

            eval_res = evaluate(eid, model_gen, model_disc, lm, sent_encoder,
                                valid_loader, criterion, criterion_sem, criterion_lm,
                                criterion_recon, device, args)
            logger.info(prettify_res_dict(eval_res, prefix='| valid '))

            tot_eval_loss = (eval_res['gen_loss'] + eval_res['recon_loss'] +
                             eval_res['msg_loss'] + eval_res['lm_loss'] +
                             eval_res['sem_loss'])
            if tot_eval_loss < stored_loss:
                save_models(args.save, model_gen, model_disc, optimizer_gen,
                            optimizer_disc, criterion, criterion_recon)
                logger.info('saving averaged')
                stored_loss = tot_eval_loss

            for prm in model_gen.parameters():
                prm.data = tmp[prm].clone()

            for prm in model_disc.parameters():
                prm.data = tmp[prm].clone()

        else:
            eval_res = evaluate(eid, model_gen, model_disc, lm, sent_encoder,
                                valid_loader, criterion, criterion_sem, criterion_lm,
                                criterion_recon, device, args)
            tot_eval_loss = (eval_res['gen_loss'] + eval_res['recon_loss'] +
                             eval_res['msg_loss'] + eval_res['lm_loss'] +
                             eval_res['sem_loss'])
            text_eval_loss = (eval_res['recon_loss'] + eval_res['sem_loss'] +
                              eval_res['lm_loss'])
            logger.info(prettify_res_dict(eval_res, prefix='| valid '))

            if tot_eval_loss < stored_loss:
                save_models(args.save, model_gen, model_disc, optimizer_gen,
                            optimizer_disc, criterion, criterion_recon)
                logger.info('saving model (new best overall performance)')
                stored_loss = tot_eval_loss
            if eval_res['msg_loss'] < stored_loss_msg:
                save_models(f'{args.save}_msg', model_gen, model_disc, optimizer_gen,
                            optimizer_disc, criterion, criterion_recon)
                logger.info('Saving model (new best msg validation)')
                stored_loss_msg = eval_res['msg_loss']
            if text_eval_loss < stored_loss_text:
                save_models(f'{args.save}_recon', model_gen, model_disc, optimizer_gen,
                            optimizer_disc, criterion, criterion_recon)
                logger.info('Saving model (new best reconstruct validation)')
                stored_loss_text = text_eval_loss
            if eid % args.save_interval == 0:
                save_models(f'{args.save}_interval', model_gen, model_disc, optimizer_gen,
                            optimizer_disc, criterion, criterion_recon)
                logger.info('Saving model (intervals)')

    # run on test set
    best_saved_dict = resume_models(args.save)
    model_gen = best_saved_dict['model_gen']
    model_disc = best_saved_dict['model_disc']

    model_gen = model_gen.to(device)
    model_disc = model_disc.to(device)

    test_res = evaluate(eid, model_gen, model_disc, lm, sent_encoder, test_loader,
                        criterion, criterion_sem, criterion_lm, criterion_recon, device,
                        args)
    logger.info(prettify_res_dict(test_res, prefix='| test '))


if __name__ == '__main__':
    args = parse_args_csn_training()
    main(args)
