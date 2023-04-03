from difflib import Differ
from collections import Counter, defaultdict
from pprint import PrettyPrinter


if __name__ == '__main__':
    # FILE_NAME_VAL = 'val_out_lm.txt'
    FILE_NAME_TEST = 'test_out.txt'

    added_dict = defaultdict(list)
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
    for cursor in range(len(lines)):
        cur_line = lines[cursor]

        if cur_line.startswith('sample'):
            # print(cur_line.strip())
            count += 1

        if cur_line.startswith('original text:'):
            orig = cur_line.replace('original text:', '').strip()
            cursor += 1
            modified = lines[cursor].replace('modified text:', '').strip()

            cursor += 2  # skip correct bits

            assert lines[cursor].startswith('gt msg:')
            gt_msg = lines[cursor].replace('gt msg:', '').strip()
            msg_dict[gt_msg] += 1

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
