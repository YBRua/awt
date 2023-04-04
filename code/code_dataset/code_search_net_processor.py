import os
import json
from tqdm import tqdm

from .code_vocab import CodeVocab
from .data_instance import DataInstance
from .code_tokenizer import split_name

from typing import List


class CodeSearchNetProcessor:
    def __init__(self) -> None:
        pass

    def _load_jsonl(self, jsonl_file: str) -> None:
        json_instances = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                json_instances.append(json.loads(line))
        return json_instances

    def _process_jsonl(self,
                       jsonl_file: str,
                       show_progress: bool = True) -> List[DataInstance]:
        instances: List[DataInstance] = []
        json_instances = self._load_jsonl(jsonl_file)

        if show_progress:
            progress = tqdm(json_instances)
            jsonl_file_basename = os.path.basename(jsonl_file)
            progress.set_description(f'Processing {jsonl_file_basename:<25}')
        else:
            progress = json_instances

        for json_instance in progress:
            source_code = json_instance['code']
            raw_code_tokens = json_instance['code_tokens']
            tokens = []
            for t in raw_code_tokens:
                tokens.extend(split_name(t))

            instances.append(
                DataInstance(source_code=source_code,
                             raw_source_tokens=raw_code_tokens,
                             tokens=tokens))

        return instances

    def process_jsonls(self,
                       jsonl_files: List[str],
                       show_progress: bool = True) -> List[DataInstance]:
        instances = []
        for jsonl_file in jsonl_files:
            instances.extend(self._process_jsonl(jsonl_file, show_progress))

        return instances

    def build_vocabulary_on_instances(self, instances: List[DataInstance]) -> None:
        vocab = CodeVocab()
        for instance in instances:
            for tok in instance.tokens:
                vocab.add_word(tok)

        return vocab
