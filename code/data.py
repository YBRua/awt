import os
import json
import torch

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word_freq = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.word_freq[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path=None):
        self.dictionary = Dictionary()
        if path is not None:
            self.train = self.tokenize(os.path.join(path, "train.txt"))
            self.valid = self.tokenize(os.path.join(path, "valid.txt"))
            self.test = self.tokenize(os.path.join(path, "test.txt"))
        else:
            self.train = None
            self.valid = None
            self.test = None

    def tokenize_jsonl(self, path):
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as fi:
            lines = fi.readlines()
        objs = [json.loads(line) for line in lines]

        tokens = 0
        for obj in objs:
            words = obj["output"].split() + ["<eos>"]
            tokens += len(words)

        ids = torch.LongTensor(tokens)
        token = 0
        for obj in objs:
            words = obj["output"].split() + ["<eos>"]
            for word in words:
                if word not in self.dictionary.word2idx:
                    ids[token] = self.dictionary.word2idx["<unk>"]
                else:
                    ids[token] = self.dictionary.word2idx[word]
                token += 1

        return ids

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r") as f:
            tokens = 0
            for line in f:
                words = line.split() + ["<eos>"]
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
