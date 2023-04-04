import re

from typing import List


def sanitize_name(name):
    # https://github.com/eliphatfs/torch.redstone
    return re.sub(r'\W|^(?=\d)', '_', name)


def split_name(c_token: str) -> List[str]:
    # special case handling for string literals
    if c_token.startswith('"') and c_token.endswith('"'):
        return c_token[1:-1].split()

    res = []
    snake_splitted = _try_split_snake(c_token)
    for tok in snake_splitted:
        res.extend(_try_split_camel(tok))
    return res


def _try_split_snake(c_token: str) -> List[str]:
    words = c_token.split('_')
    res = ['_'] * (len(words) * 2 - 1)
    res[0::2] = words
    return res


def _try_split_camel(c_token: str) -> List[str]:
    return re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', c_token).split()


def split_string_literal(c_token: str) -> List[str]:
    # remove escape sequences
    stripped = re.sub(r'\\.', '', c_token)

    return stripped.strip().split()


def split_identifier(c_token: str) -> List[str]:
    if '/' in c_token:  # include path
        res = []
        for subtok in c_token.split('/'):
            res.extend(split_identifier(subtok))
            res.append('/')
        return res
    else:
        return split_name(c_token)
