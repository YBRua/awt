import re
import json
import tree_sitter
from argparse import ArgumentParser
from tqdm import tqdm
from tree_sitter import Parser, Language
from check_illegal import KeywordChecker

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
    parser.add_argument('jsonl_path',
                        type=str,
                        help='path to a jsonl file to be evaluated')
    parser.add_argument('--key',
                        type=str,
                        default='after_watermark',
                        help='key to code snippet items in json objects')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='if enabled, will print invalid code snippets')
    parser.add_argument(
        '--original_test_file',
        type=str,
        default='/home/borui/awt-orig/code/data/test_comments_removed.jsonl')
    args = parser.parse_args()

    LANG = 'java'
    # JSONL_PATH = './data/awt-modified-2023-04-28.jsonl'
    JSONL_PATH = args.jsonl_path
    ORIGINGAL_TEST_SET_PATH = args.original_test_file

    MAX_DEPTH = -1
    PARSER_LANG = Language('/home/borui/code-watermarking/metrics/parser/languages.so',
                           LANG)
    parser = Parser()
    parser.set_language(PARSER_LANG)
    secondary_checker = KeywordChecker(parser)

    with open(JSONL_PATH, 'r') as fi:
        lines = fi.readlines()

    with open(ORIGINGAL_TEST_SET_PATH, 'r') as fi:
        token_lengths = [len(json.loads(line)['code_tokens']) for line in fi.readlines()]

    n_failed_grammar = 0
    n_passed_grammar = 0
    n_failed_keyword = 0
    n_passed_keyword = 0
    n_valid = 0
    n_invalid = 0
    total_trees = 0

    total_bit_acc = 0
    total_bit_per_token = 0
    total_watermarked_samples = 0
    for lid, line in enumerate(tqdm(lines)):
        try:
            json_obj = json.loads(line)
        except json.decoder.JSONDecodeError:
            json_obj = eval(line.strip())
        except Exception as e:
            print('Unknown error')
            print(e)
            exit(1)

        # check if the code is valid
        wm_code = json_obj[args.key]
        line = fix_single_quotes(fix_join_artifacts(wm_code))

        if LANG == 'java':
            line_fix = f'public class A {{ {line} }}'

        tree = parser.parse(bytes(line_fix, 'utf-8'))
        if check_tree_validity(tree.root_node, MAX_DEPTH):
            n_passed_grammar += 1
        else:
            n_failed_grammar += 1

        if secondary_checker.are_all_keywords_legal(line_fix):
            n_passed_keyword += 1
        else:
            n_failed_keyword += 1

        if (check_tree_validity(tree.root_node, MAX_DEPTH)
                and secondary_checker.are_all_keywords_legal(line_fix)):
            n_valid += 1
        else:
            n_invalid += 1
            if args.verbose:
                print(line_fix)
        total_trees += 1

        # check bitwise accuracy
        label = json_obj['watermark']
        extracted = json_obj['extract']

        if min(len(label), len(extracted)) == 0:
            continue

        bit_acc = compare_bitwise_acc(extracted, label)
        total_bit_acc += bit_acc
        total_watermarked_samples += 1

        # compute capacity
        bpt = len(extracted) / token_lengths[lid]
        total_bit_per_token += bpt

    assert total_trees == (n_valid + n_invalid)
    assert total_trees == (n_passed_grammar + n_failed_grammar)
    assert total_trees == (n_passed_keyword + n_failed_keyword)

    print(f'Valid: {n_valid}/{total_trees} ({n_valid / total_trees * 100:.2f}%)')
    # print(f'Invalid: {n_invalid} ({n_invalid / total_trees * 100:.4f}%)')
    print(f'Passed grammar: {n_passed_grammar}/{total_trees} '
          f'({n_passed_grammar / total_trees * 100:.2f}%)')
    # print(f'Failed grammar: {n_failed_grammar} '
    #       f'({n_failed_grammar / total_trees * 100:.4f}%)')
    print(f'Passed keyword: {n_passed_keyword}/{total_trees} '
          f'({n_passed_keyword / total_trees * 100:.2f}%)')
    # print(f'Failed keyword: {n_failed_keyword} '
    #       f'({n_failed_keyword / total_trees * 100:.4f}%)')

    print(f'Bitwise accuracy: {total_bit_acc / total_watermarked_samples:.4f}')
    print(f'Bits per token: {total_bit_per_token / total_watermarked_samples:.4f}')
