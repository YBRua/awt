import torch
from typing import List
from data import Dictionary, Corpus

from .code_vocab import CodeVocab
from .data_instance import DataInstance
from .code_search_net_dataset import CodeSearchNetDataset


def _instances_to_token_ids(instances: List[DataInstance], dictionary: Dictionary):
    token_ids = []
    for instance in instances:
        token_ids.extend([
            dictionary.word2idx.get(word, dictionary.word2idx['<unk>'])
            for word in instance.tokens
        ])
    return torch.tensor(token_ids, dtype=torch.long)


def csn_dataset_to_flat_tensor(dataset: CodeSearchNetDataset):
    token_ids = []
    for instance in dataset.token_ids:
        token_ids.extend(instance)
    return torch.tensor(token_ids, dtype=torch.long)


def csn_dataset_to_corpus(train_dataset: CodeSearchNetDataset,
                          valid_dataset: CodeSearchNetDataset,
                          test_dataset: CodeSearchNetDataset, vocab: CodeVocab):
    corpus = Corpus()

    corpus.dictionary.word2idx = vocab.word2idx
    corpus.dictionary.idx2word = vocab.idx2word
    corpus.dictionary.word_freq = vocab.word_freq
    corpus.dictionary.total = sum(vocab.word_freq.values())

    corpus.train = csn_dataset_to_flat_tensor(train_dataset.token_ids)
    corpus.valid = csn_dataset_to_flat_tensor(valid_dataset.token_ids)
    corpus.test = csn_dataset_to_flat_tensor(test_dataset.token_ids)

    return corpus
