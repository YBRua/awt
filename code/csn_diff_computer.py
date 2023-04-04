import sys
from difflib import Differ
from collections import Counter, defaultdict
from pprint import PrettyPrinter

if __name__ == '__main__':
    # FILE_NAME_VAL = 'val_out_lm.txt'
    if len(sys.argv) < 2:
        print('Usage: python csn_diff_computer.py <log_name>')
        exit(0)
    LOG_NAME = sys.argv[1]
    FNAME = f'./logs/{LOG_NAME}'

    added_dict = defaultdict(list)
    removed_dict = defaultdict(list)
    replaced_dict = defaultdict(list)
    msg_dict = defaultdict(int)

    diff_calc = Differ()
    pp = PrettyPrinter(width=41, compact=True)

    with open(FNAME, 'r') as f:
        lines = f.readlines()

    # with open(FILE_NAME_TEST, 'r') as f:
    #     lines.extend(f.readlines())

    count = 0
    for cursor in range(len(lines)):
        cur_line = lines[cursor]

        if cur_line.startswith('sample'):
            # print(cur_line.strip())
            count += 1

        if cur_line.startswith('gt  msg:'):
            gt_msg = lines[cursor].replace('gt  msg:', '').strip()
            msg_dict[gt_msg] += 1
            cursor += 5

            assert lines[cursor].startswith('[ORIGINAL] ')
            orig = lines[cursor].replace('[ORIGINAL] ', '').strip()
            cursor += 1
            modified = lines[cursor].replace('[MODIFIED] ', '').strip()

            res = list(diff_calc.compare(orig.split(), modified.split()))
            prev = ''
            for tok in res:
                if tok.startswith(' '):
                    continue
                elif tok.startswith('-'):
                    removed_dict[gt_msg].append(tok[2:])
                elif tok.startswith('+'):
                    added_dict[gt_msg].append(tok[2:])
                    if prev.startswith('-'):
                        replaced_dict[gt_msg].append((prev[2:], tok[2:]))
                prev = tok

    for msg in sorted(removed_dict.keys()):
        removed_list = removed_dict[msg]
        added_list = added_dict[msg]

        print(f'Message {msg} ({msg_dict[msg]} samples)')
        # print('Removed')
        # pp.pprint(Counter(removed_list))
        print('Added')
        added_counter = Counter(added_list)
        most_feq = added_counter.most_common(3)
        print(f'  Most Frequent: {most_feq} ({len(added_list)} total additions)')
        print(f'  All Added: {added_counter}')
        print('Removed')
        print(f'  All Removed: {Counter(removed_list)}')
        print(f'  Total {len(removed_list)} removals')
        print()
