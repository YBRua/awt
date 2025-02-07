import json
import torch
from code_dataset import CodeSearchNetDataset, CodeSearchNetProcessor, CodeVocab

TEST_FILE_PATH = './data/CodeSearchNet/java/final/jsonl/test/java_test_0.jsonl'

processor = CodeSearchNetProcessor()

token_ids = processor.process_jsonls([TEST_FILE_PATH])[:100]
vocab = processor.build_vocabulary_on_instances(token_ids)

dataset = CodeSearchNetDataset(token_ids, vocab)
dataset.dump_json('cache.json')

round_tripped = CodeSearchNetDataset.from_json('cache.json')

ori_instances = dataset.token_ids
rt_instances = round_tripped.token_ids

ori_vocab = dataset.vocab
rt_vocab = round_tripped.vocab

assert all([
    ori_inst.serialize() == rt_inst.serialize()
    for ori_inst, rt_inst in zip(ori_instances, rt_instances)
])
assert ori_vocab.serialize() == rt_vocab.serialize()

print('Yay!')
