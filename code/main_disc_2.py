import argparse
import time
import math
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.metrics import f1_score
import data_disc_lm

import model_discriminator
import model_discriminator_lstm

from typing import List
from utils import batchify, get_batch_fixed, generate_msgs, repackage_hidden, get_batch_no_msg

parser = argparse.ArgumentParser(
    description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data',
                    type=str,
                    default='data/penn/',
                    help='location of the data corpus')

parser.add_argument('--ratio',
                    type=int,
                    default=1,
                    help='Ratio of the training data real text to watermarked text')

parser.add_argument('--emsize', type=int, default=512, help='size of word embeddings')
parser.add_argument('--lr', type=float, default=0.00003, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=8000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=80, help='sequence length')
parser.add_argument('--fixed_length',
                    type=int,
                    default=0,
                    help='whether to use a fixed input length (bptt value)')

parser.add_argument('--dropouti',
                    type=float,
                    default=0.1,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute',
                    type=float,
                    default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')

parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--nonmono', type=int, default=5, help='random seed')
parser.add_argument('--cuda', action='store_false', help='use CUDA')
parser.add_argument('--log-interval',
                    type=int,
                    default=200,
                    metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save',
                    type=str,
                    default=randomhash,
                    help='path to save the final model')
parser.add_argument('--resume', type=str, default='', help='path of model to resume')
parser.add_argument('--optimizer',
                    type=str,
                    default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--wdecay',
                    type=float,
                    default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument(
    '--when',
    nargs="+",
    type=int,
    default=[-1],
    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

#transformer arguments
parser.add_argument('--classifier',
                    type=str,
                    default='lstm',
                    help='if the classifier is lstm or transformer')

parser.add_argument('--dropout_transformer',
                    type=float,
                    default=0.1,
                    help='dropout applied to transformer layers (0 = no dropout)')
parser.add_argument('--attn_heads',
                    type=int,
                    default=4,
                    help='The number of attention heads in the transformer')
parser.add_argument('--encoding_layers',
                    type=int,
                    default=6,
                    help='The number of encoding layers')

#lstm arguments
parser.add_argument('--lstm_layers',
                    type=int,
                    default=2,
                    help='Number of layers in the discriminator lstm')
parser.add_argument('--lstm_dim',
                    type=int,
                    default=512,
                    help='The dim of the discriminator lstm')
parser.add_argument('--dropouth',
                    type=float,
                    default=0.2,
                    help='dropout on the lstm output')
parser.add_argument('--dropout',
                    type=float,
                    default=0.4,
                    help='dropout before the last ff layer')

#Adam optimizer arguments
parser.add_argument(
    '--scheduler',
    type=int,
    default=1,
    help=
    'whether to schedule the lr according to the formula in: Attention is all you need')
parser.add_argument('--warm_up',
                    type=int,
                    default=6000,
                    help='number of linear warm up steps')
parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1 parameter')
parser.add_argument('--beta2', type=float, default=0.98, help='Adam beta2 parameter')
parser.add_argument('--eps', type=float, default=1e-9, help='Adam eps parameter')

args = parser.parse_args()
args.tied = True
args.tied_lm = True

log_file_loss_val = open('log_file_loss_val.txt', 'w')

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################


def model_save(fn):
    with open(fn + '.pt', 'wb') as f:
        torch.save([model_disc, criterion, optimizer_disc], f)


def model_load(fn):
    global model_disc, criterion, optimizer_disc
    with open(fn + '.pt', 'rb') as f:
        model_disc, criterion, optimizer_disc = torch.load(f, map_location='cpu')


def construct_dataset(data_real, data_fake):
    instances = []

    for i in range(0, len(data_real) // 80 * 80, 80):
        instances.append(AWTDiscriminatorDataInstance(data_real[i:i + 80], 1))

    for i in range(0, len(data_fake) // 80 * 80, 80):
        instances.append(AWTDiscriminatorDataInstance(data_fake[i:i + 80], 0))

    return instances


class AWTDiscriminatorDataInstance:
    def __init__(self, token_ids, label):
        self.token_ids = token_ids
        self.label = label


class AWTDiscriminatorDataset(Dataset):
    def __init__(self, data_instances: List[AWTDiscriminatorDataInstance]) -> None:
        super().__init__()
        self.instances = data_instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index: int):
        return self.instances[index]


class AWTDiscriminatorCollator:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, batch):
        token_ids = []
        labels = []

        for item in batch:
            token_ids.append(item.token_ids)
            labels.append(item.label)

        # B, L -> L, B
        token_ids = torch.stack(token_ids).transpose(0, 1)
        labels = torch.tensor(labels, dtype=torch.long)

        return token_ids, labels


import os
import hashlib

fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())

corpus = data_disc_lm.Corpus(args.data)
print('Running on fine-tuned model samples output')

eval_batch_size = 1
test_batch_size = 1

data_len = len(corpus.train_real)
train_real_ids = corpus.train_real[:data_len - int(data_len / (args.ratio + 1))]
train_fake_ids = corpus.train_fake[data_len - int(data_len / (args.ratio + 1)):]
train_instances = construct_dataset(train_real_ids, train_fake_ids)
train_dataset = AWTDiscriminatorDataset(train_instances)
print(f'Train dataset size: {len(train_dataset)}')

# train_fake_data = batchify(corpus.train_fake, args.batch_size, args)
val_fake_data = batchify(corpus.valid_fake, eval_batch_size, args)
test_fake_data = batchify(corpus.test_fake, test_batch_size, args)

# train_real_data = batchify(corpus.train_real, args.batch_size, args)
val_real_data = batchify(corpus.valid_real, eval_batch_size, args)
test_real_data = batchify(corpus.test_real, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

criterion = None

ntokens = len(corpus.dictionary)
print(ntokens)
word2idx = corpus.dictionary.word2idx
idx2word = corpus.dictionary.idx2word

## global variable for the number of steps ( batches) ##
step_num = 1


def learing_rate_scheduler():
    d_model = args.emsize
    warm_up = args.warm_up
    lr = np.power(d_model, -0.95) * min(np.power(step_num, -0.5),
                                        step_num * np.power(warm_up, -1.5))
    return lr


if args.resume != '':
    print('Resuming model ...')
    model_load(args.resume)
    optimizer_disc.param_groups[0]['lr'] = learing_rate_disc_scheduler(
    ) if args.scheduler else args.lr
else:
    if args.classifier == 'lstm':
        model_disc = model_discriminator_lstm.DiscriminatorModel(
            ntokens, args.emsize, args.lstm_layers, args.lstm_dim, args.dropouti,
            args.dropoute, args.dropout, args.dropouth)
    elif args.classifier == 'transformer':
        model_disc = model_discriminator.DiscriminatorModel(ntokens, args.emsize,
                                                            args.encoding_layers,
                                                            args.dropout_transformer,
                                                            args.attn_heads)

###
if not criterion:
    criterion = nn.BCEWithLogitsLoss()
###

if args.cuda:
    model_disc = model_disc.cuda()
    criterion = criterion.cuda()

###
params = list(list(criterion.parameters()) + list(model_disc.parameters()))
params_disc = model_disc.parameters()

total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0]
                   for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


#convert word ids to text
def convert_idx_to_words(idx):
    batch_list = []
    for i in range(0, idx.size(1)):
        sent_list = []
        for j in range(0, idx.size(0)):
            sent_list.append(corpus.dictionary.idx2word[idx[j, i]])
        batch_list.append(sent_list)
    return batch_list


### conventions for real and fake labels during training ###
real_label = 1
fake_label = 0

###############################################################################
# Training code
###############################################################################


def evaluate(data_source_real, data_source_fake, batch_size=10):
    # Turn on evaluation mode which disables dropout.

    model_disc.eval()
    total_loss_disc = 0

    ntokens = len(corpus.dictionary)
    batches_count = 0
    for i in range(0, data_source_real.size(0) - args.bptt, args.bptt):
        data_real, targets_real = get_batch_no_msg(data_source_real,
                                                   i,
                                                   args,
                                                   evaluation=True)
        data_fake, targets_fake = get_batch_no_msg(data_source_fake,
                                                   i,
                                                   args,
                                                   evaluation=True)

        # run the real and fake to the disc model
        real_out = model_disc.forward(data_real)
        fake_out = model_disc.forward(data_fake)

        label = torch.full((data_real.size(1), 1), real_label)
        if args.cuda:
            label = label.cuda()
        errD_real = criterion(real_out, label.float())
        # get prediction (and the loss) of the discriminator on the fake sequence.

        label.fill_(fake_label)
        errD_fake = criterion(fake_out, label.float())
        errD = errD_real + errD_fake

        total_loss_disc += errD.data
        batches_count = batches_count + 1

    return total_loss_disc.item() / batches_count


def train():
    dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=AWTDiscriminatorCollator())
    device = torch.device("cuda")

    tot_loss = 0.0
    tot_acc = 0.0
    all_preds = []
    all_labels = []

    progress = tqdm(dataloader)
    for bid, (x, y) in enumerate(progress):
        x = x.to(device)
        y = y.to(device).unsqueeze(1)

        model_disc.train()
        optimizer_disc.zero_grad()

        ####### Update lr #######
        if args.scheduler:
            optimizer_disc.param_groups[0]['lr'] = learing_rate_scheduler()

        out = model_disc.forward(x)
        error = criterion(out, y.float())
        error.backward()
        optimizer_disc.step()

        preds = torch.round(torch.sigmoid(out))

        tot_acc += (preds == y).float().mean().item()
        tot_loss += error.item()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

        avg_acc = tot_acc / (bid + 1)
        avg_loss = tot_loss / (bid + 1)

        progress.set_description(
            f'EPOCH {epoch} | loss: {avg_loss:.4f} | acc: {avg_acc:.4f}')

        global step_num
        step_num += 1

    print(f'EPOCH {epoch} | F1: {f1_score(all_labels, all_preds):.4f}')


# Loop over epochs.
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer_disc = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer_disc = torch.optim.SGD(params_disc,
                                         lr=args.lr,
                                         weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer_disc = torch.optim.Adam(
            params_disc,
            lr=learing_rate_scheduler() if args.scheduler else args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.wdecay)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer_disc.param_groups[0]:
            print('Not implemented')
        else:
            val_loss_disc = evaluate(val_real_data, val_fake_data, eval_batch_size)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                  'val disc loss {:5.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss_disc))
            print('-' * 89)
            log_file_loss_val.write(str(val_loss_disc) + '\n')
            log_file_loss_val.flush()

            if val_loss_disc < stored_loss:
                model_save(args.save)
                print('Saving model (new best discriminator validation)')
                stored_loss = val_loss_disc

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)
if args.cuda:
    model_disc = model_disc.cuda()
    criterion = criterion.cuda()

test_loss_disc = evaluate(test_real_data, test_fake_data, test_batch_size)

print('-' * 89)
print('| End of training | test disc loss {:5.2f}'.format(test_loss_disc))
print('-' * 89)
