"""CounterfactualGAN (c), 2020-2021 Marcel Robeer."""

import pandas as pd
import numpy as np
import emoji
import re
import glob
import os
import torch
import pathlib
from tqdm import tqdm
from scipy import sparse
from nltk import word_tokenize
from typing import Optional, List

from config import SEED, DATA, VECTORS

pd.options.mode.chained_assignment = None


def get_name(path):
    """Determine whether the file path is train, dev or test."""
    path = os.path.splitext(os.path.split(path)[-1])[0]
    if 'train' in path:
        return 'train'
    if 'test' in path:
        return 'test'
    return 'dev'


def maximize_by_length(d, length):
    if 'pandas' in str(type(d)):
        x = [c for c in d.columns if 'x' in str(c).lower()]
        if len(x) == 1:
            return d[d[x[0]].str.len() < length]
        l = np.vectorize(len)
        return d[(l(d[x].values.astype(str)) < length).any(axis=1)]
    raise NotImplementedError()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, target_size=1, max_test_length=130, **readargs):
        self.target_size = target_size

        # Read file(s)
        multiple_filenames = isinstance(filename, list) or '*' in filename
        __filename = filename
        if '*' in filename:
            filename = glob.glob(filename)
        self.filename = filename
        if not multiple_filenames:
            filename = [filename]
        if not filename:
            raise FileNotFoundError(f'Could not find file "{__filename}"')

        # Extract parts train/dev/test
        self.data = {get_name(f): self.preprocess(pd.read_csv(f, **readargs)) for f in filename}
        self.encoded_data = {}

        if not multiple_filenames:
            self.data = self.split(self.data)
        if isinstance(max_test_length, int) and max_test_length > 0:
            self.data['test'] = maximize_by_length(self.data['test'], max_test_length)

        # Make folder for pre-encoded vectors
        self.encoded_folder = os.path.join(VECTORS, str.lower(self.__class__.__name__))
        if not os.path.exists(self.encoded_folder):
            pathlib.Path(self.encoded_folder).mkdir(parents=True)

        self.encode_mode = None
        self.part = 'train'

    def __repr__(self):
        return self.__class__.__name__

    def __len__(self):
        """Length of dataset part."""
        if not self.encode_mode or self.encode_mode is None:
            len(self.data[self.part])
        return self.encoded_data[self.encode_mode][self.part].shape[0]

    def __getitem__(self, index):
        """Get a single (pre-encoded) instance."""
        if self.encode_mode is None:
            return self.data[self.part].iloc[index].values
        X = self.encoded_data[self.encode_mode][self.part]
        y = self.data[self.part]['y'].iloc[index]
        return ({i: x[index, :] for i, x in enumerate(X)} if isinstance(X, list) else X[index, :],
                np.float32(y) if isinstance(y, np.float64) else y)

    def preprocess(self, X):
        """Optionally preprocess each file:
        - Remove obsolete columns
        - Rename columns
        - Remove NaN values
        - etc.
        """
        return X

    def split(self, data, train=0.7, dev=0.1, test=0.2, reset_index=True):
        """Split data into train/dev/test."""
        assert train + dev + test == 1., 'parts should sum to one'

        data_ = data[list(data.keys())[0]]
        np.random.seed(SEED)
        to_select = np.random.rand(len(data_))  # TO-DO seed

        parts = {}
        parts['train'] = data_[to_select < train]
        parts['dev'] = data_[(to_select >= train) & (to_select < (dev + train))]
        parts['test'] = data_[to_select > (dev + train)]
        if reset_index:
            for part in ['train', 'dev', 'test']:
                parts[part] = parts[part].reset_index(drop=True)
        return parts

    def describe(self):
        """Print dataset descriptives."""
        print(f'Dataset "{str(self)}"')
        print(f'Columns: {self.data["train"].columns.values}')
        print(f'Target size: {self.target_size} ({"regression" if self.target_size == 1 else "classification"})')
        if len(self.encoded_data.keys()) > 0:
            print(f'Encoded: {", ".join(list(self.encoded_data.keys()))}')
        print(f'Parts:')
        dataset_len = 0
        for k in ['train', 'dev', 'test']:
            if k in self.data.keys():
                dataset_len += self.data[k].shape[0]
                print(f'  {k:5} | {self.data[k].shape[0]}')
        print(' ' * 7, '|', str(dataset_len), '+')
        print('Column lengths (str, tokens):')
        X = self.get('all')
        for col in [col for col in X.columns if 'X' in col]:
            print(f'  {col}: {X[col].str.len().mean():.3f}, {sum(len(word_tokenize(str(w))) for w in X[col]) / len(X):.3f}')

    def get(self, part='all', return_type='pd', encode_method=None):
        """Get (part of) data.

        Args:
            part: name of part (train/dev/test/all)
            return_type: type to return for part
            encode_method: return encoded data (None is standard data)
        """
        assert part in ['all'] + list(self.data.keys())
        assert return_type in ['pd', 'pandas', 'pt', 'torch', 'pytorch']
        assert encode_method in [None] + list(self.encoded_data.keys())

        if return_type in ['pd', 'pandas']:
            data = pd.concat(self.data.values()) if part == 'all' else self.data[part]
        elif return_type in ['pt', 'torch', 'pytorch']:
            data = torch.vstack(self.encoded_data[encode_method].values()) \
                if part == 'all' else self.encoded_data[encode_method]
        else:
            raise NotImplementedError('')
        return data

    def encode(self,
               encode_fn: callable,
               encode_name: str,
               attempt_load: bool = True,
               filetype: str = 'pt',
               parts: Optional[List[str]] = None):
        """Encode data, later retrieveable by `encode_name`.

        Args:
            encode_fn: function to encode part of the data
            encode_name: unique name for pre-encoded data
            attempt_load: whether to attempt loading the pre-encoded data from the `VECTORS` folders (if exists)
            filetype: type of file to save pre-encoded data to (`pt` for `torch`,
                `np` for `numpy`, `npz` for `scipy.sparse`)
        """
        allowed_filetypes = {'pt': (torch.load, lambda loc, data: torch.save(data.clone(), loc)),
                             'npy': (np.load, np.save),
                             'npz': (sparse.load_npz, sparse.save_npz)}
        assert filetype in allowed_filetypes.keys(), f'Unknown filetype "{filetype}"'
        self.encode_mode = encode_name

        load_fn, save_fn = allowed_filetypes[filetype]

        if parts is None:
            parts = self.data.keys()
   
        # Load if exists
        path = os.path.join(self.encoded_folder, str.lower(encode_name))
        if attempt_load:
            files = [os.path.split(p)[-1] for p in glob.glob(path + '/*')]
            if all(k + '.' + filetype in files for k in parts):
                self.encoded_data[encode_name] = {}
                for k in parts:
                    self.encoded_data[encode_name][k] = load_fn(os.path.join(path, k) + '.' + filetype)
                if all(self.data[k].shape[0] == self.encoded_data[encode_name][k].shape[0]
                       for k in parts):
                    return self

        # Encode
        part_items = [(k, v) for k, v in self.data.items() if k in parts]
        self.encoded_data[encode_name] = {part_name: self._encode(encode_fn, part)
                                          for part_name, part in
                                          tqdm(part_items, desc=f'Encoding {"/".join(parts)} data')} 

        # Save
        if not os.path.exists(path):
            pathlib.Path(path).mkdir(parents=True)
        for part_name, encoded_part in self.encoded_data[encode_name].items():
            save_fn(os.path.join(path, part_name) + '.' + filetype, encoded_part)
        return self

    def _encode(self, encode_fn, part):
        """Encoding implementation; applies `encode_fn` to a part."""
        return encode_fn(part['X'])

    def target(self, part='test', seed=SEED, return_type='np'):
        assert return_type in ['pt', 'np']
        size = len(self.get(part=part))
        np.random.seed(seed)
        y_target = np.random.rand(size) if self.target_size == 1 else \
            np.random.randint(0, self.target_size, size)

        return torch.tensor(y_target) if return_type == 'pt' else y_target


class Hatespeech(Dataset):
    def __init__(self):
        """Hatespeech dataset from `https://data.world/thomasrdavidson/hate-speech-and-offensive-language`."""
        self.mention_pattern = re.compile(r'@\w*')
        self.tco_pattern = re.compile(r'http://t.co/\w*')
        self.label_map = {0: 0.0, 1: 1.0, 2: 0.4}
        super().__init__(os.path.join(DATA, 'hatespeech_data.csv'), target_size=1, index_col=0)

    def preprocess(self, X):
        """Selects the `tweet` (X) and `hate_speech` (y) columns and replace @mention with @user."""
        def clean_tweet(s):
            return emoji.demojize(self.tco_pattern.sub('http://t.co/', self.mention_pattern.sub('@user', s)))

        X = X[['tweet', 'class']]
        X['tweet'] = X['tweet'].apply(clean_tweet)
        X['class'] = X['class'].map(self.label_map).astype(float)
        return X.rename({'tweet': 'X', 'class': 'y'}, axis=1)


class SST(Dataset):
    def __init__(self):
        """SST-2 dataset from `https://github.com/clairett/pytorch-sentiment-classification/tree/master/data/SST2`."""
        super().__init__(os.path.join(DATA, 'SST2') + '/*.tsv', target_size=2, sep='\t', header=None)

    def preprocess(self, X):
        """Renames columns to X and y."""
        return X.rename({0: 'X', 1: 'y'}, axis=1)


class SNLI(Dataset):
    def __init__(self):
        """SNLI dataset from `https://archive.nyu.edu/handle/2451/41728`."""
        self.label_map = {'entailment': 0, 'neutral': 1, '-': 1, 'contradiction': 2}
        super().__init__(os.path.join(DATA, 'snli_1.0/snli_1.0_') + '*.txt', target_size=3, sep='\t')

    def preprocess(self, X):
        """..."""
        X = X[['sentence2', 'sentence1', 'gold_label']]
        X['gold_label'] = X['gold_label'].map(self.label_map).astype(int)
        return X.rename({'sentence1': 'X_premise', 'sentence2': 'X_hypothesis', 'gold_label': 'y'}, axis=1)  

    def _encode(self, encode_fn, part):
        """Encoding implementation; applies `encode_fn` to `X_premise` and `X_hypothesis` in a part."""
        return encode_fn(part[['X_premise', 'X_hypothesis']].astype(str).values)
