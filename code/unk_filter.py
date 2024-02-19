import json
from collections import Counter

if __name__ == '__main__':
    LOG_FILE = './logs/csn-eval-bert-original.log'
    THRES_1 = 0.3
    THRES_2 = 0.15
    original_test_dataset_path = './data/test_comments_removed.jsonl'

    with open(original_test_dataset_path, 'r') as fi:
        json_objs = [json.loads(line) for line in fi.readlines()]

    print(list(json_objs[0].keys()))

    with open(LOG_FILE, 'r') as fi:
        lines = fi.readlines()

    cursor = 0
    n_thres_1 = 0
    n_thres_2 = 0
    thres_1s = []
    thres_2s = []
    tot_samples = 0
    while cursor < len(lines):
        line = lines[cursor]
        if line.startswith('[ORIGINAL_ID]'):
            original_id = int(line.replace('[ORIGINAL_ID]:', '').strip())
            original_obj = json_objs[original_id]
            tot_samples += 1

            cursor += 2
            line = lines[cursor]
            assert line.startswith('[MODIFIED]')
            modified = lines[cursor].replace('[MODIFIED]', '').strip()

            toks = modified.split()
            tok_counter = Counter(toks)
            top_1_word, top_1_val = tok_counter.most_common(1)[0]

            if top_1_word == '<unk>':
                continue

            tot_toks = len(toks)

            modified_code = lines[cursor].replace('[MODIFIED]', '').strip()
            if top_1_val / tot_toks >= THRES_1:
                n_thres_1 += 1
                thres_1s.append({
                    'after_watermark': modified_code,
                    'repo': original_obj['repo'],
                    'path': original_obj['path'],
                    'func_name': original_obj['func_name']
                })
            elif top_1_val / tot_toks <= THRES_2:
                n_thres_2 += 1
                thres_2s.append({
                    'after_watermark': modified_code,
                    'repo': original_obj['repo'],
                    'path': original_obj['path'],
                    'func_name': original_obj['func_name']
                })

        cursor += 1

    print(f'Number of examples with unks >= {THRES_1}: {n_thres_1}')
    print(f'Number of examples with unks <= {THRES_2}: {n_thres_2}')
    print(f'Total number of examples: {tot_samples}')

    with open('./thres_1.jsonl', 'w') as fo:
        for line in thres_1s[:100]:
            fo.write(json.dumps(line) + '\n')

    with open('./thres_2.jsonl', 'w') as fo:
        for line in thres_2s[:100]:
            fo.write(json.dumps(line) + '\n')

    print(f'Schema: {list(thres_1s[0].keys())}')
