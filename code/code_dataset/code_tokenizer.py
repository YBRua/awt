import re

from typing import List
from sctokenizer import CppTokenizer, JavaTokenizer
from sctokenizer import Token, TokenType


def sanitize_name(name):
    # https://github.com/eliphatfs/torch.redstone
    return re.sub(r'\W|^(?=\d)', '_', name)


def split_name(c_token: str) -> List[str]:
    # special case handling for string literals
    if c_token.startswith('"') and c_token.endswith('"'):
        return ['"'] + c_token[1:-1].split() + ['"']

    res = []
    snake_splitted = _try_split_snake(c_token)
    for tok in snake_splitted:
        res.extend(_try_split_camel(tok))
    return res


def _add_special_token(splitted: List[str]):
    res = [splitted[0]]
    for tok in splitted[1:]:
        res.append(f'')


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


class CodeTokenizer:
    def __init__(self, lang: str = 'c'):
        self.lang = lang
        if lang in ('c', 'cpp'):
            self.tokenizer = CppTokenizer()
        elif lang == 'java':
            self.tokenizer = JavaTokenizer()
        else:
            raise ValueError('Language must be either "c" or "java"')

    def _tokens_postprocess(self, tokens: List[Token]):
        res = []
        for token in tokens:
            if token.token_type == TokenType.COMMENT_SYMBOL:
                raise RuntimeError('No comment allowed!')
            if token.token_type == TokenType.STRING:
                # res.extend(split_string_literal(token.token_value))
                res.append('__string__')
            elif token.token_type == TokenType.CONSTANT:
                res.append('__constant__')
            elif token.token_type == TokenType.IDENTIFIER:
                res.extend(split_identifier(token.token_value))
            elif len(token.token_value) > 40:
                # the tokenizer is sometimes buggy
                # skip extremely long 'token's
                res.append('<unk>')
            else:
                res.append(token.token_value)
        return res

    def get_tokens(self, source: str):
        code_tokens = self.tokenizer.tokenize(source)
        return code_tokens, self._tokens_postprocess(code_tokens)
