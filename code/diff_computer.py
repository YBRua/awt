from tqdm import trange
from difflib import Differ
from collections import Counter, defaultdict
from pprint import PrettyPrinter

if __name__ == '__main__':
    # FILE_NAME_VAL = 'val_out_lm.txt'
    FILE_NAME_TEST = 'logs/csn-eval-bert-2023-05-24-12-08-14.log'

    added_dict = defaultdict(list)
    added_pos = defaultdict(list)
    removed_dict = defaultdict(list)
    replaced_dict = defaultdict(list)
    msg_dict = defaultdict(int)

    diff_calc = Differ()
    pp = PrettyPrinter(width=41, compact=True)

    with open(FILE_NAME_TEST, 'r') as f:
        lines = f.readlines()

    # with open(FILE_NAME_TEST, 'r') as f:
    #     lines.extend(f.readlines())

    count = 0
    for cursor in trange(len(lines)):
        cur_line = lines[cursor]

        if cur_line.startswith('sample'):
            # print(cur_line.strip())
            count += 1

        if cur_line.startswith('gt  msg:'):
            gt_msg = lines[cursor].replace('gt  msg:', '').strip()
            cursor += 1

            assert lines[cursor].startswith('dec msg:')
            dec_msg = lines[cursor].replace('dec msg:', '').strip()
            cursor += 5

            msg_dict[dec_msg] += 1

            assert lines[cursor].startswith('[ORIGINAL]')
            orig = lines[cursor].replace('[ORIGINAL]', '').strip()
            cursor += 1

            assert lines[cursor].startswith('[MODIFIED]')
            modified = lines[cursor].replace('[MODIFIED]', '').strip()

            res = list(diff_calc.compare(orig.split(), modified.split()))
            prev = ''
            for tidx, tok in enumerate(res):
                if tok.startswith(' '):
                    continue
                elif tok.startswith('-'):
                    removed_dict[dec_msg].append(tok[2:])
                elif tok.startswith('+'):
                    added_dict[dec_msg].append(tok[2:])
                    added_pos[dec_msg].append(tidx)
                    if prev.startswith('-'):
                        removed_dict[dec_msg].append(prev[2:])
                prev = tok

    frequent_set = set()
    for msg in sorted(msg_dict.keys()):
        removed_list = removed_dict[msg]
        added_list = added_dict[msg]
        added_pos_list = added_pos[msg]

        print(f'Message {msg} ({msg_dict[msg]} samples)')
        # print('Removed')
        # pp.pprint(Counter(removed_list))
        print('Added')
        added_counter = Counter(added_list)
        most_freq = added_counter.most_common(5)
        frequent_set.add(most_freq[0][0])
        print(f'  Most Frequent: {most_freq} ({len(added_list)} total additions)')
        # print(f'  All Added: {added_counter}')

        # print('Added Position')
        # added_pos_counter = Counter(added_pos_list)
        # most_freq = added_pos_counter.most_common(5)
        # print(f'  Most Frequent: {most_freq} ({len(added_pos_list)} total additions)')

        # print('Removed')
        # removed_counter = Counter(removed_list)
        # most_freq = removed_counter.most_common(3)
        # print(f'  Most Frequent: {most_freq} ({len(removed_list)} total removals)')
        print()

    print(frequent_set)
    print(len(msg_dict))
