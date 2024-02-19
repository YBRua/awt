import os

from tqdm import tqdm
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

from .code_vocab import CodeVocab
from .data_instance import DataInstance
from .cwm_dataset import CodeWatermarkDataset
from .code_tokenizer import CodeTokenizer
from typing import List, Dict, Optional, Tuple


class CodeWatermarkProcessor:

    def __init__(self, lang: str = 'c') -> None:
        self.lang = lang
        self.code_tokenizer = CodeTokenizer(lang=lang)

    def _default_filename_splitter(self, filename: str) -> str:
        fname_no_ext = os.path.splitext(filename)[0]
        sep = fname_no_ext.rfind('_')
        assert sep != -1, f'Filename {filename} does not follow the expected format'
        return fname_no_ext[:sep], fname_no_ext[sep + 1:]  # problem, author

    def tokenize_file(self, fpath: str):
        # print(fpath)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f'No such file: {fpath}')

        with open(fpath, 'r', encoding='utf-8') as f:
            source = f.read()
        code_tokenizer = CodeTokenizer(lang=self.lang)
        code_tokens, word_tokens = code_tokenizer.get_tokens(source)
        return source, code_tokens, word_tokens

    def _load_nested_dir_mp_worker(self, args: Tuple[str, str, str]):
        dirpath, subdirpath, filename = args
        filepath = os.path.join(dirpath, subdirpath, filename)
        source, code_tokens, tokens = self.tokenize_file(filepath)
        return DataInstance(source_code=source,
                            raw_source_tokens=code_tokens,
                            tokens=tokens)

    def _load_nested_dir_mp(self, dirpath: str, num_processes: int):
        args = []
        for author_name in sorted(os.listdir(dirpath)):
            subdirpath = os.path.join(dirpath, author_name)
            if not os.path.isdir(subdirpath):
                print(f'Warning: {subdirpath} is not a directory.')
                continue

            for file in sorted(os.listdir(subdirpath)):
                filepath = os.path.join(subdirpath, file)
                if not os.path.isfile(filepath):
                    print(f'Warning: {filepath} is not a file.')
                    continue

                args.append((dirpath, author_name, file))

        with Pool(num_processes) as p:
            instances = p.map(self._load_nested_dir_mp_worker, args)

        return instances

    def load_nested_dir(self, dirpath: str, num_processes: Optional[int] = None):
        # each dir in dirpath is an author,
        # each author dir contains source code files
        self._check_dir_exist(dirpath)

        if num_processes is not None:
            return self._load_nested_dir_mp(dirpath, num_processes)

        instances = []
        for author_name in sorted(os.listdir(dirpath)):
            # subdir_base is also the author's name
            subdir = os.path.join(dirpath, author_name)

            if not os.path.isdir(subdir):
                print(f'Warning: {subdir} is not a directory.')
                continue

            for file in sorted(os.listdir(subdir)):
                fpath = os.path.join(subdir, file)
                source, code_tokens, tokens = self.tokenize_file(fpath)

                # construct data instance
                instances.append(
                    DataInstance(source_code=source,
                                 raw_source_tokens=code_tokens,
                                 tokens=tokens))

        return instances

    def _check_dir_exist(self, dir_name: str):
        if not os.path.exists(dir_name):
            raise ValueError(f'Directory {dir_name} does not exist')

    def _get_filename(self,
                      filename_noext: str,
                      transform_keys: Tuple[str],
                      ext: str = 'c'):
        transform_keys_str = '.'.join(transform_keys)
        return f'{filename_noext}.{transform_keys_str}.{ext}'

    def _load_nested_transformed_mp_worker(self, args: Tuple[str, str, str, List]):
        dirpath, author, filename, transform_keys = args
        t_filename = self._get_filename(filename, transform_keys, self.lang)
        t_filepath = os.path.join(dirpath, author, filename, t_filename)
        source, code_tokens, tokens = self.tokenize_file(t_filepath)
        if len(tokens) == 0:
            print(f'Warning: {t_filepath} is empty')
            return None
        return DataInstance(source_code=source,
                            raw_source_tokens=code_tokens,
                            tokens=tokens)

    def _load_nested_transformed_dir_mp(self,
                                        trasformed_code_dir: str,
                                        feasible_transforms: Dict,
                                        num_processes: Optional[int] = None,
                                        show_progress: bool = True) -> List[DataInstance]:
        args = []
        for author in os.listdir(trasformed_code_dir):
            author_dir = os.path.join(trasformed_code_dir, author)
            self._check_dir_exist(author_dir)

            for filename in os.listdir(author_dir):
                file_dir = os.path.join(author_dir, filename)
                self._check_dir_exist(file_dir)

                all_transform_keys = feasible_transforms[author][
                    f'{filename}.{self.lang}']
                for transform_keys in all_transform_keys:
                    t_filename = self._get_filename(filename, transform_keys, self.lang)
                    t_filepath = os.path.join(file_dir, t_filename)
                    if not os.path.exists(t_filepath):
                        raise FileNotFoundError(f'No such file: {t_filepath}')

                    args.append((trasformed_code_dir, author, filename, transform_keys))

        with Pool(num_processes) as p:
            prog = p.imap_unordered(self._load_nested_transformed_mp_worker, args)
            if show_progress:
                prog = tqdm(prog, total=len(args))

            res = []
            for item in prog:
                res.append(item)

        return res

    def load_nested_transformed_dir(self,
                                    trasformed_code_dir: str,
                                    feasible_transforms: Dict,
                                    num_processes: Optional[int],
                                    show_progress: bool = True) -> List[DataInstance]:
        if num_processes is not None:
            return self._load_nested_transformed_dir_mp(trasformed_code_dir,
                                                        feasible_transforms,
                                                        num_processes, show_progress)

        if show_progress:
            progress = tqdm(os.listdir(trasformed_code_dir))
        else:
            progress = os.listdir(trasformed_code_dir)

        instances = []
        for author in progress:
            author_dir = os.path.join(trasformed_code_dir, author)
            self._check_dir_exist(author_dir)

            for filename in os.listdir(author_dir):
                file_dir = os.path.join(author_dir, filename)
                self._check_dir_exist(file_dir)

                all_transform_keys = feasible_transforms[author][filename]
                for transform_keys in all_transform_keys:
                    t_filename = self._get_filename(filename, transform_keys)
                    t_filepath = os.path.join(file_dir, t_filename)
                    source, code_tokens, tokens = self.tokenize_file(t_filepath)
                    instance = DataInstance(source_code=source,
                                            raw_source_tokens=code_tokens,
                                            tokens=tokens)
                    instances.append(instance)
        return instances

    def train_test_split_by_proportion(self,
                                       instances: List[DataInstance],
                                       train_proportion: float = 0.8,
                                       seed: int = 42):
        train_examples, test_examples = train_test_split(instances,
                                                         train_size=train_proportion,
                                                         random_state=seed)

        return train_examples, test_examples

    def build_dataset(self, instances: List[DataInstance],
                      vocab: CodeVocab) -> CodeWatermarkDataset:
        return CodeWatermarkDataset(instances, vocab)

    def build_vocab(self, instances: List[DataInstance]) -> CodeVocab:
        vocab = CodeVocab()
        for instance in instances:
            for tok in instance.tokens:
                vocab.add_word(tok)

        return vocab
