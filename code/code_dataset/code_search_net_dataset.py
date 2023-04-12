import json
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from .data_instance import DataInstance
from .code_vocab import CodeVocab, WikiTextVocabWrapper
from typing import Union, List


def build_dataset(instances: List[DataInstance],
                  vocab: Union[CodeVocab, WikiTextVocabWrapper, RobertaTokenizer]):
    token_ids = []
    for instance in instances:
        token_ids.append(vocab.convert_tokens_to_ids(instance.tokens))

    return token_ids


class CodeSearchNetDataset(Dataset):
    def __init__(self, token_ids: List[List[int]]) -> None:
        super().__init__()
        self.token_ids = token_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, index):
        return self.token_ids[index]

    def dump_json(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(self.token_ids, f, indent=2)

    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, 'r') as f:
            json_obj = json.load(f)

        return cls(json_obj)
