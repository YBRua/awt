import json
from grammar_check import join_lines, fix_join_artifacts, fix_single_quotes

if __name__ == '__main__':
    LANG = 'java'
    ORIGINAL_INPUT = 'data/test_pots.jsonl'
    OUTPUT_FILE = 'data/test_pots_csn.jsonl'
    LOG_OUTPUT = './logs/csn-eval-bert-2023-04-23-11-18-50.log'

    # load keywords
    KEYWORD_DIR = './data/keywords'
    KEYWORD_PATH = f'{KEYWORD_DIR}/{"c" if LANG in {"c", "cpp"} else LANG}.txt'
    with open(KEYWORD_PATH, 'r') as fi:
        keywords = set(w.strip() for w in fi.readlines())

    # load log
    with open(LOG_OUTPUT, 'r') as f:
        log_lines = f.readlines()

    # load original inputs
    with open(ORIGINAL_INPUT, 'r') as f:
        original_lines = f.readlines()

    # collect modified code
    cursor = 0
    id_to_modified_code = dict()
    while cursor < len(log_lines):
        line = log_lines[cursor]
        if line.startswith('sample #'):
            cursor += 2
            gt_msg_line = log_lines[cursor]
            assert gt_msg_line.startswith('gt  msg:')
            gt_msg_line = gt_msg_line.replace('gt  msg:', '').strip()
            gt_msg = json.loads(gt_msg_line)[0]

            cursor += 1
            decoded_msg_line = log_lines[cursor]
            assert decoded_msg_line.startswith('dec msg:')
            decoded_msg_line = decoded_msg_line.replace('dec msg:', '').strip()
            dec_msg = list(map(int, json.loads(decoded_msg_line)[0]))

            cursor += 4
            original_id_line = log_lines[cursor]
            assert original_id_line.startswith('[ORIGINAL_ID]:')
            original_id_line = original_id_line.replace('[ORIGINAL_ID]:', '').strip()
            original_id = int(original_id_line)

            cursor += 2
            modified_line = log_lines[cursor]
            assert modified_line.startswith('[MODIFIED]')
            tokens = modified_line.replace('[MODIFIED]', '').strip().split()
            line_val = join_lines(tokens, keywords, LANG)
            line_fixed = fix_single_quotes(fix_join_artifacts(line_val)).strip()
            id_to_modified_code[original_id] = {
                'after_watermark': line_fixed,
                'watermark': gt_msg,
                'extract': dec_msg,
            }

        cursor += 1

    repackaged = []
    for i, ori_line in enumerate(original_lines):
        original_obj = json.loads(ori_line)
        obj = dict()
        obj['docstr_token'] = original_obj['docstring_tokens']

        if i in id_to_modified_code:
            obj['output_origin_func'] = False  # flag for whether watermark succeeded
            obj['after_watermark'] = id_to_modified_code[i]['after_watermark']
            obj['watermark'] = id_to_modified_code[i]['watermark']
            obj['extract'] = id_to_modified_code[i]['extract']
        else:
            obj['output_origin_func'] = True
            obj['after_watermark'] = original_obj['code']
            obj['watermark'] = None
            obj['extract'] = None

        repackaged.append(obj)

    print(f'original: {len(original_lines)}, repackaged: {len(repackaged)}')

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(repackaged, f, indent=2)
