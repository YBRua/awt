import re
import json
import tree_sitter
from argparse import ArgumentParser
from tqdm import tqdm
from tree_sitter import Parser, Language

from typing import List, Set

with open('./data/keywords/java.txt', 'r') as fi:
    java_keywords = fi.read().splitlines()
KEYWORDS = set(java_keywords)


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
    space_rm = re.sub(r"([^']*)\s+'\s+", r"\1'", input_str)
    return space_rm.replace("''", "' '")


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
        elif prev in keywords:
            line += f' {tok}'
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
    return not root.has_error


def compare_bitwise_acc(extracted: List[int], labels: List[int]):
    correct_bits = 0
    total_bits = 0
    idx = 0
    end = min(len(extracted), len(labels))
    assert end != 0
    while idx < end:
        if extracted[idx] == labels[idx]:
            correct_bits += 1
        total_bits += 1
        idx += 1

    return correct_bits / total_bits


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('json_path',
                        type=str,
                        help='path to a json file (dewatermark attack output)')
    parser.add_argument('--key',
                        type=str,
                        default='sentence',
                        help='key to code snippet items in json objects')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='if enabled, will print invalid code snippets')
    parser.add_argument('--lang', choices=['java', 'javascript', 'cpp'])
    args = parser.parse_args()

    LANG = args.lang
    KEY = args.key
    JSON_PATH = args.json_path

    MAX_DEPTH = -1
    PARSER_LANG = Language('/home/borui/code-watermarking/parser/languages.so', LANG)
    parser = Parser()
    parser.set_language(PARSER_LANG)

    with open(JSON_PATH, 'r') as fi:
        objs = json.load(fi)

    n_failed_grammar = 0
    n_passed_grammar = 0
    total_trees = 0

    for lid, json_obj in enumerate(tqdm(objs)):
        # check if the code is valid
        wm_code = json_obj[args.key]
        wm_code = join_lines(wm_code.split(), keywords=KEYWORDS, lang=LANG)
        line = fix_single_quotes(fix_join_artifacts(wm_code))

        if LANG == 'java':
            line_fix = f'public class A {{ {line} }}'
        else:
            line_fix = line

        tree = parser.parse(bytes(line_fix, 'utf-8'))

        grammar_check_ok = check_tree_validity(tree.root_node, MAX_DEPTH)

        if grammar_check_ok:
            n_passed_grammar += 1
            print(lid)
        else:
            n_failed_grammar += 1

        total_trees += 1

    assert total_trees == (n_passed_grammar + n_failed_grammar)

    print(f'Grammar check passed: {n_passed_grammar} '
          f'({n_passed_grammar / total_trees * 100:.2f}%)')
    print(f'Grammar check failed: {n_failed_grammar} '
          f'({n_failed_grammar / total_trees * 100:.2f}%)')
