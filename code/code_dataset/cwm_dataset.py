from torch.utils.data import Dataset

from .code_vocab import CodeVocab
from .data_instance import DataInstance
from typing import List

MAX_TOKEN_LEN = 512


class CodeWatermarkDataset(Dataset):

    def __init__(self, instances: List[DataInstance], vocab: CodeVocab):
        super().__init__()
        self.instances = instances
        self.vocab = vocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index: int):
        instance = self.instances[index]
        return self.vocab.convert_tokens_to_ids(instance.tokens)[:MAX_TOKEN_LEN]
