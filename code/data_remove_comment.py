import re
import json
from typing import List


def remove_comments(source: str):

    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                         re.DOTALL | re.MULTILINE)
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)


def nl_split(input: str, append_space: bool = True) -> List[str]:
    if append_space:
        pat = re.compile(r"([.()!<>{};,@\[\]])")
        input = pat.sub(" \\1 ", input)
    return input.split()


if __name__ == '__main__':
    LANG = 'java'
    ORIGINAL_INPUT = 'data/test_pots.jsonl'
    OUTPUT_FILE_PATH = 'data/test_comments_removed.jsonl'

    with open(ORIGINAL_INPUT, 'r') as f:
        original_lines = f.readlines()

    updated = []
    for i, ori_line in enumerate(original_lines):
        original_obj = json.loads(ori_line)
        removed_comment = remove_comments(original_obj['original_string'])
        if original_obj['original_string'] != removed_comment:
            print(removed_comment)
            print(original_obj['original_string'])
            print()
        original_obj['original_string'] = removed_comment
        updated.append(original_obj)

    with open(OUTPUT_FILE_PATH, 'w') as f:
        for obj in updated:
            f.write(json.dumps(obj) + '\n')
