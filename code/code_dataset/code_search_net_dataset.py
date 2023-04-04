import json
from torch.utils.data import Dataset

from .data_instance import DataInstance
from .code_vocab import CodeVocab, WikiTextVocabWrapper
from typing import Union, List


class CodeSearchNetDataset(Dataset):
    def __init__(
        self,
        instances: List[DataInstance],
        vocab: Union[CodeVocab, WikiTextVocabWrapper],
    ) -> None:
        super().__init__()
        self.instances = instances
        self.vocab = vocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

    def dump_json(self, file_path: str) -> None:
        json_instances = [inst.serialize() for inst in self.instances]

        json_obj = {
            'instances': json_instances,
            'vocab': self.vocab.serialize(),
        }

        with open(file_path, 'w') as f:
            json.dump(json_obj, f, indent=2)

    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, 'r') as f:
            json_obj = json.load(f)

        instances = []
        for json_instance in json_obj['instances']:
            instances.append(DataInstance.deserialize(json_instance))

        vocab = CodeVocab()
        vocab.word2idx = json_obj['vocab']['word2idx']
        vocab.idx2word = json_obj['vocab']['idx2word']
        vocab.word_freq = json_obj['vocab']['word_freq']

        return cls(instances, vocab)
