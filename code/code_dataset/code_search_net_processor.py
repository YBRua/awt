import os
import json
import tree_sitter
from tqdm import tqdm
from sctokenizer import Token, TokenType
from sctokenizer import CppTokenizer, JavaTokenizer

from .code_vocab import CodeVocab
from .data_instance import DataInstance
from .code_tokenizer import split_name

from typing import List, Optional, Callable


class JavaScriptTokenizer:
    def __init__(self):
        parser = tree_sitter.Parser()
        parser_lang = tree_sitter.Language('./parser/languages.so', 'javascript')
        parser.set_language(parser_lang)
        self.parser = parser

    def collect_tokens(self, root: tree_sitter.Node) -> List[Token]:
        tokens = []

        def _collect_token(node: tree_sitter.Node):
            if node.type == 'comment':
                return
            elif node.type in {'number'}:
                tokens.append(
                    Token(node.text.decode(), TokenType.CONSTANT, node.start_point[0],
                          node.start_point[1]))
            elif node.type in {'string', 'template_string', 'regex'}:
                tokens.append(
                    Token(node.text.decode(), TokenType.STRING, node.start_point[0],
                          node.start_point[1]))
            elif node.type in {
                    'identifier', 'shorthand_property_identifier',
                    'shorthan_property_identifier_pattern'
            }:
                tokens.append(
                    Token(node.text.decode(), TokenType.IDENTIFIER, node.start_point[0],
                          node.start_point[1]))
            elif node.child_count == 0:
                tokens.append(
                    Token(node.text.decode(), TokenType.KEYWORD, node.start_point[0],
                          node.start_point[1]))
            else:
                assert node.child_count > 0
                for ch in node.children:
                    _collect_token(ch)

        _collect_token(root)
        return tokens

    def tokenize(self, code: str) -> List[Token]:
        tree = self.parser.parse(bytes(code, 'utf-8'))
        tokens = self.collect_tokens(tree.root_node)
        return tokens


class CodeSearchNetProcessor:
    def __init__(self, lang: str = 'java') -> None:
        self.lang = lang
        if self.lang == 'java':
            self.tokenizer = JavaTokenizer()
        elif self.lang == 'cpp':
            self.tokenizer = CppTokenizer()
        elif self.lang == 'javascript':
            self.tokenizer = JavaScriptTokenizer()
        else:
            print(f'Warning: unsupported language {lang}')
            self.tokenizer = None

    def _load_jsonl(self, jsonl_file: str) -> None:
        json_instances = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                json_instances.append(json.loads(line))
        return json_instances

    def _naive_token_split(self, tokens: List[str]) -> List[str]:
        new_tokens = []
        for t in tokens:
            new_tokens.extend(split_name(t))
        return new_tokens

    def _process_jsonl(self,
                       jsonl_file: str,
                       show_progress: bool = True,
                       instance_filter: Optional[Callable] = None) -> List[DataInstance]:
        instances: List[DataInstance] = []
        json_instances = self._load_jsonl(jsonl_file)

        if show_progress:
            progress = tqdm(json_instances)
            jsonl_file_basename = os.path.basename(jsonl_file)
            progress.set_description(f'Processing {jsonl_file_basename:<25}')
        else:
            progress = json_instances

        for json_instance in progress:
            source_code = json_instance['original_string']
            if 'code_tokens' in json_instance:
                raw_code_tokens = json_instance['code_tokens']
            else:
                print('no pretokenized code tokens, using built-in tokenizer')
                if self.tokenizer is None:
                    raise RuntimeError('cannot tokenize code without tokenizer')
                raw_code_tokens = self.tokenizer.tokenize(source_code)
                raw_code_tokens = [t.token_value for t in raw_code_tokens]
            tokens = self._naive_token_split(raw_code_tokens)

            instance = DataInstance(source_code=source_code,
                                    raw_source_tokens=raw_code_tokens,
                                    tokens=tokens)
            if instance_filter is None or instance_filter(instance):
                instances.append(instance)

        return instances

    def process_jsonls(self,
                       jsonl_files: List[str],
                       show_progress: bool = True,
                       instance_filter: Optional[Callable] = None) -> List[DataInstance]:
        instances = []
        for jsonl_file in jsonl_files:
            chunk_instances = self._process_jsonl(jsonl_file, show_progress,
                                                  instance_filter)
            instances.extend(chunk_instances)

        return instances

    def build_vocabulary_on_instances(self, instances: List[DataInstance]) -> CodeVocab:
        vocab = CodeVocab()
        for instance in instances:
            for tok in instance.tokens:
                vocab.add_word(tok)

        return vocab
