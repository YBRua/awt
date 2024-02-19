import json
import numpy as np
from collections import Counter
from tqdm import tqdm


with open('./data/test_comments_removed.jsonl', 'r') as fi:
    json_objs = [json.loads(line) for line in fi.readlines()]


tot_toks = 0
tok_lens = []
for json_obj in tqdm(json_objs):
    tok_lens.append(len(json_obj['code_tokens']))
    tot_toks += len(json_obj['code_tokens'])
tok_lens = list(sorted(tok_lens))

assert len(json_objs) == 10955
print(tot_toks / len(json_objs))
print("quantile @0.1", np.quantile(tok_lens, 0.1))
print("quantile @0.2", np.quantile(tok_lens, 0.2))
