import re
import json
import tree_sitter
import string
from tree_sitter import Parser, Language

from typing import List, Set


def fix_join_artifacts(text: str):
    # remove spaces between dots
    text = re.sub(r'\s?\.\s?(?=\w+)', '.', text)
    # remove spaces between underscores
    text = re.sub(r'_\s?(?=\w+)', '_', text)
    # replace 0X with 0x
    text = re.sub(r'0X', '0x', text)
    return text


def fix_single_quotes(input_str: str):
    # removes all spaces between single quotes to fix char pasing
    return re.sub(r"\s+(?=(?:(?:[^']*'){2})*[^']*'[^']*$)", '', input_str)


def join_lines_default(tokens: List[str], keywords: Set[str]):
    line = ''
    prev = ''
    is_member_access = False
    for tok in tokens:
        if tok == '<unk>':
            if prev[0].isupper():
                tok = 'Unk'
            else:
                tok = 'Unk'

        if tok == '.':
            is_member_access = True
        elif not tok.isidentifier():
            is_member_access = False

        if (prev.isidentifier() and tok.isidentifier() and prev not in keywords
                and tok not in keywords):
            if tok[0] == '_' or tok[0].isupper():
                line += tok
            else:
                line += f' {tok}'
        elif prev == '.' and tok == '*':
            line += tok
        elif is_member_access:
            line += tok
        elif prev == '<' or tok in {'<', '>'}:
            line += tok
        else:
            line += f' {tok}'
        prev = tok

    return line


def join_lines_java_heuristic(tokens: List[str], keywords: Set[str]):
    line = ''
    prev = ''
    is_member_access = False
    for tok in tokens:
        if tok == '<unk>':
            if len(prev) and prev[0].isupper():
                tok = 'Unk'
            else:
                tok = 'Unk'

        if tok == '.':
            is_member_access = True
        elif not tok.isidentifier():
            is_member_access = False

        if (prev.isidentifier() and tok.isidentifier() and prev not in keywords
                and tok not in keywords):
            if tok[0] == '_' or tok[0].isupper():
                line += tok
            else:
                line += f' {tok}'
        elif prev == '.' and tok == '*':
            line += tok
        elif is_member_access:
            line += tok
        elif prev == '<' or tok in {'<', '>'}:
            line += tok
        elif prev == '@':
            line += tok
        else:
            line += f' {tok}'
        prev = tok

    return line


def join_lines(tokens: List[str], keywords: Set[str], lang: str):
    if lang == 'java':
        return join_lines_java_heuristic(tokens, keywords)
    else:
        return join_lines_default(tokens, keywords)


def check_tree_validity(root: tree_sitter.Node, max_depth: int = 3):
    def _check_tree(node: tree_sitter.Node, depth: int = 0, max_depth: int = 3):
        valid = True
        if max_depth > 0 and depth > max_depth:
            return True
        if node.type == 'ERROR':
            return False
        for child in node.children:
            valid = valid and _check_tree(child, depth + 1, max_depth=max_depth)
        return valid

    return _check_tree(root, depth=0, max_depth=max_depth)


if __name__ == '__main__':
    LANG = 'cpp'
    LOG_PATH = './logs/baseline_gcj_cpp.log'

    KEYWORD_DIR = './data/keywords'
    KEYWORD_PATH = f'{KEYWORD_DIR}/{"c" if LANG in {"c", "cpp"} else LANG}.txt'

    # load keywords
    with open(KEYWORD_PATH, 'r') as fi:
        keywords = set(w.strip() for w in fi.readlines())

    MAX_DEPTH = -1
    PARSER_LANG = Language('/home/borui/code-watermarking/metrics/parser/languages.so',
                           LANG)
    parser = Parser()
    parser.set_language(PARSER_LANG)

    with open(LOG_PATH, 'r') as fi:
        lines = fi.readlines()

    valid_trees = 0
    total_trees = 0
    for line in lines:
        if line.startswith('[MODIFIED]'):
            line_tokens = line.replace('[MODIFIED]', '').strip().split()
            line = join_lines(line_tokens, keywords, LANG)
            line_fix = fix_single_quotes(fix_join_artifacts(line))
            # print(line_fix)
            # print()

            # line_fix = json.loads(line)['code']

            # if LANG == 'java':
            #     line_fix = f'public class A {{ {line_fix} }}'

            tree = parser.parse(bytes(line_fix, 'utf-8'))

            if check_tree_validity(tree.root_node, MAX_DEPTH):
                valid_trees += 1
            # else:
            #     print(line_fix)
            #     print()
            total_trees += 1

    print(f'Valid trees: {valid_trees}/{total_trees} ({valid_trees / total_trees:.4f})')
