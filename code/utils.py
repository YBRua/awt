import torch
import numpy as np
from typing import Dict


def prettify_res_dict(res: Dict, prefix: str = None) -> str:
    res_str = "|"
    for k, v in res.items():
        if isinstance(v, float):
            res_str += f" {k}: {v:.4f} |"
        elif isinstance(v, int):
            res_str += f" {k}: {v:3d} |"
        else:
            res_str += f" {k}: {v} |"

    if prefix is not None:
        assert isinstance(prefix, str), "prefix must be a string"
        res_str = prefix + res_str

    return res_str


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def generate_msgs(args):
    msgs_num = args.msgs_num
    msgs = np.random.choice([0, 1], [msgs_num, args.msg_len])
    np.savetxt("msgs.txt", msgs.astype(int))
    return msgs


def word_substitute(tokens, x, p):  # substitute words with probability p
    keep = torch.rand(x.size(), device=x.device) > p
    x_ = x.clone()
    x_.random_(0, tokens)
    x_[keep] = x[keep]
    return x_


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def get_batch_no_msg(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def get_batch_noise(source, i, args, tokens, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    data_noise = word_substitute(tokens, data, args.sub_prob)
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, data_noise, target


def get_batch_different(source, i, args, all_msgs, seq_len=None, evaluation=False):
    # get a different random msg for each sentence.
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    bsz = data.size(1)
    msg = np.random.choice([0, 1], [bsz, args.msg_len])
    # assert args.msg_len == 4
    # msg = np.ones([bsz, args.msg_len])
    # single_msg = [0, 0, 0, 0]
    # msg = np.tile(single_msg, (bsz, 1))
    msg = torch.from_numpy(msg).float()
    if args.cuda:
        msg = msg.cuda()
    return data, msg, target


def get_batch_fixed(source, i, args, wm, seq_len=None, evaluation=False):
    if len(wm) != args.msg_len:
        raise ValueError("msg length should be equal to args.msg_len")
    if any([m not in [0, 1] for m in wm]):
        raise ValueError("msg should be a binary list")

    # get a different random msg for each sentence.
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    bsz = data.size(1)
    msg = np.tile(wm, (bsz, 1))
    msg = torch.from_numpy(msg).float()
    if args.cuda:
        msg = msg.cuda()
    return data, msg, target


def batchify_test(data, bsz, args_cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args_cuda:
        data = data.cuda()
    return data


def generate_msgs_test(msgs_num, msg_len):
    msgs = np.random.choice([0, 1], [msgs_num, msg_len])
    np.savetxt("msgs.txt", msgs.astype(int))
    return msgs
