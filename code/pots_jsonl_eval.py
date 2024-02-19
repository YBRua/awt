import os
import re
import json
import pickle
import warnings
import tree_sitter
from argparse import ArgumentParser
from tqdm import tqdm
from tree_sitter import Parser, Language
from check_illegal import KeywordChecker

from typing import List, Set

with open("./data/keywords/javascript.txt", "r") as fi:
    java_keywords = fi.read().splitlines()
KEYWORDS = set(java_keywords)


def fix_join_artifacts(text: str):
    # remove spaces between dots
    text = re.sub(r"\s?\.\s?(?=\w+)", ".", text)
    # remove spaces between underscores
    text = re.sub(r"_\s?(?=\w+)", "_", text)
    # replace 0X with 0x
    text = re.sub(r"0X", "0x", text)
    return text


def fix_single_quotes(input_str: str):
    # removes all spaces between single quotes to fix char pasing
    space_rm = re.sub(r"([^']*)\s+'\s+", r"\1'", input_str)
    return space_rm.replace("''", "' '")


def join_lines_default(tokens: List[str], keywords: Set[str]):
    line = ""
    prev = ""
    is_member_access = False
    for tok in tokens:
        if tok == "<unk>":
            if prev[0].isupper():
                tok = "Unk"
            else:
                tok = "Unk"

        if tok == ".":
            is_member_access = True
        elif not tok.isidentifier():
            is_member_access = False

        if (
            prev.isidentifier()
            and tok.isidentifier()
            and prev not in keywords
            and tok not in keywords
        ):
            if tok[0] == "_" or tok[0].isupper():
                line += tok
            else:
                line += f" {tok}"
        elif prev == "." and tok == "*":
            line += tok
        elif is_member_access:
            line += tok
        elif prev == "<" or tok in {"<", ">"}:
            line += tok
        else:
            line += f" {tok}"
        prev = tok

    return line


def join_lines_java_heuristic(tokens: List[str], keywords: Set[str]):
    line = ""
    prev = ""
    is_member_access = False
    for tok in tokens:
        if tok == "<unk>":
            if len(prev) and prev[0].isupper():
                tok = "Unk"
            else:
                tok = "Unk"

        if tok == ".":
            is_member_access = True
        elif not tok.isidentifier():
            is_member_access = False

        if (
            prev.isidentifier()
            and tok.isidentifier()
            and prev not in keywords
            and tok not in keywords
        ):
            if tok[0] == "_" or tok[0].isupper():
                line += tok
            else:
                line += f" {tok}"
        elif prev == "." and tok == "*":
            line += tok
        elif is_member_access:
            line += tok
        elif prev == "<" or tok in {"<", ">"}:
            line += tok
        elif prev == "@":
            line += tok
        elif prev in keywords:
            line += f" {tok}"
        else:
            line += f" {tok}"
        prev = tok

    return line


def join_lines(tokens: List[str], keywords: Set[str], lang: str):
    if lang == "java" or lang == "javascript":
        return join_lines_java_heuristic(tokens, keywords)
    else:
        return join_lines_default(tokens, keywords)


def check_tree_validity(root: tree_sitter.Node, max_depth: int = 3):
    return not root.has_error

    def _check_tree(node: tree_sitter.Node, depth: int = 0, max_depth: int = 3):
        valid = True
        if max_depth > 0 and depth > max_depth:
            return True
        if node.type == "ERROR":
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "jsonl_path", type=str, help="path to a jsonl file to be evaluated"
    )
    parser.add_argument(
        "--key",
        type=str,
        default="after_watermark",
        help="key to code snippet items in json objects",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="if enabled, will print invalid code snippets",
    )
    parser.add_argument(
        "--original_test_file",
        type=str,
        default="/home/borui/code-watermarking/datasets/csn_java/test.jsonl",
    )
    parser.add_argument("--lang", type=str, default="java")
    args = parser.parse_args()

    LANG = "javascript"
    # JSONL_PATH = './data/awt-modified-2023-04-28.jsonl'
    JSONL_PATH = args.jsonl_path
    ORIGINGAL_TEST_SET_PATH = args.original_test_file

    MAX_DEPTH = -1
    PARSER_LANG = Language("/home/borui/code-watermarking/parser/languages.so", LANG)
    parser = Parser()
    parser.set_language(PARSER_LANG)
    secondary_checker = KeywordChecker(parser)

    with open(JSONL_PATH, "r") as fi:
        lines = fi.readlines()

    with open(ORIGINGAL_TEST_SET_PATH, "r") as fi:
        ori_lines = fi.readlines()
    token_lengths = [len(json.loads(line)["code_tokens"]) for line in ori_lines]
    original_jsons = [json.loads(line) for line in ori_lines]

    n_failed_grammar = 0
    n_passed_grammar = 0
    n_failed_keyword = 0
    n_passed_keyword = 0

    n_valid = 0
    n_valid_watermarked = 0
    n_invalid = 0
    n_grammar_invalid = 0
    n_keyword_invalid = 0
    total_trees = 0

    total_bit_acc = 0
    total_bit_per_token = 0
    total_watermarked_samples = 0

    passed_idxs = []
    wmed_idxs = []

    json_warned = False
    for lid, line in enumerate(tqdm(lines)):
        try:
            json_obj = json.loads(line)
        except json.decoder.JSONDecodeError:
            if not json_warned:
                warnings.warn("这个 JSON 好像不是很对，目测应该搞成 dict 了")
                json_warned = True
            json_obj = eval(line.strip())
        except Exception as e:
            print("uncaught error")
            print(e)
            exit(1)

        # check if the code is valid
        wm_code = json_obj[args.key]
        wm_code = join_lines(wm_code.split(), keywords=KEYWORDS, lang=LANG)
        line = fix_single_quotes(fix_join_artifacts(wm_code))

        if LANG == "java":
            line_fix = f"public class A {{ {line} }}"
        else:
            line_fix = line

        # print(line_fix)

        tree = parser.parse(bytes(line_fix, "utf-8"))

        grammar_check_ok = check_tree_validity(tree.root_node, MAX_DEPTH)
        keyword_check_ok = secondary_checker.are_all_keywords_legal(line_fix)

        if grammar_check_ok:
            n_passed_grammar += 1
        else:
            n_failed_grammar += 1

        if keyword_check_ok:
            n_passed_keyword += 1
        else:
            n_failed_keyword += 1

        if grammar_check_ok:
            if keyword_check_ok:
                n_valid += 1
                passed_idxs.append(lid)
                if (
                    not json_obj["output_original_func"]
                    and len(json_obj["watermark"]) > 0
                ):
                    n_valid_watermarked += 1
                    wmed_idxs.append(lid)

                    print("====================")
                    print(original_jsons[lid]["original_string"])
                    print("--------------------")
                    print(line)
                    print("====================")

            else:
                n_invalid += 1
                n_keyword_invalid += 1
        else:
            n_invalid += 1
            n_grammar_invalid += 1
            if args.verbose:
                print(line_fix)

        total_trees += 1

        # check bitwise accuracy
        label = json_obj["watermark"]
        extracted = json_obj["extract"]

        if len(extracted) == 0 or len(label) == 0:
            bit_acc = 0
        else:
            bit_acc = compare_bitwise_acc(extracted, label)

        total_bit_acc += bit_acc
        total_watermarked_samples += 1

        # compute capacity
        bpt = len(extracted) / token_lengths[lid]
        total_bit_per_token += bpt

    assert total_trees == (n_valid + n_invalid)
    assert total_trees == (n_passed_grammar + n_failed_grammar)
    assert total_trees == (n_passed_keyword + n_failed_keyword)

    print(f"Valid: {n_valid}/{total_trees} ({n_valid / total_trees * 100:.2f}%)")
    print(f"Invalid: {n_invalid} ({n_invalid / total_trees * 100:.4f}%)")
    print(f"  {n_grammar_invalid} failed in grammar check")
    print(f"  {n_keyword_invalid} failed in keyword check")

    print(f"Bitwise accuracy: {total_bit_acc / total_watermarked_samples:.4f}")
    print(f"Bits per token: {total_bit_per_token / total_watermarked_samples:.4f}")

    # print(f'Passed idxs: {passed_idxs}')
    print(f"Num of passed idxs: {len(passed_idxs)}")

    print(f"n_valid_watermarked: {n_valid_watermarked}")

    filename_noext = os.path.splitext(os.path.basename(JSONL_PATH))[0]
    # pickle.dump(passed_idxs, open(f'passeded_idxs_{filename_noext}.pkl', 'wb'))
    pickle.dump(wmed_idxs, open(f"wmed_idxs_{filename_noext}.pkl", "wb"))
