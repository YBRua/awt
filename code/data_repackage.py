import json


if __name__ == '__main__':
    ORIGINAL_INPUT = 'data/test_pots.jsonl'
    OUTPUT_FILE = 'data/test_pots_wt2.jsonl'
    LOG_OUTPUT = './logs/csn-eval-bert-2023-04-18-17-29-31.log'

    with open(LOG_OUTPUT, 'r') as f:
        log_lines = f.readlines()

    with open(ORIGINAL_INPUT, 'r') as f:
        original_lines = f.readlines()

    with open(OUTPUT_FILE, 'w') as fo:
        cursor = 0
        cnt = 0
        for line in log_lines:
            if line.startswith('[MODIFIED]'):
                json_obj = {'watermarked_code': line.replace('[MODIFIED]', '').strip()}
                fo.write(f'{json.dumps(json_obj)}\n')
                cnt += 1

    print(f'cnt: {cnt}')
