"""CounterfactualGAN (c), 2020-2021 Marcel Robeer.

Prepare the folder structure and required files when running experiments."""

import pathlib
import os
import warnings
import glob
import requests

from transformers import BertTokenizer
from config import (GLOVE, INFERSENT, INFERSENT_URL, BERT, DATA,
                    EXTERNAL_MODELS, TRAINED_MODELS, RESULTS)


# General filters
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def make_folder(full_path, exists_path=None):
    """Make all folders for full path."""
    exists_path = full_path if not exists_path else exists_path
    if not os.path.exists(exists_path):
        try:
            pathlib.Path(full_path).mkdir(parents=True)
        except FileExistsError:
            pass

# Prepare folders TRAINED_MODELS
make_folder(TRAINED_MODELS)
make_folder(RESULTS)
make_folder(INFERSENT, exists_path=os.path.split(INFERSENT)[0])
make_folder(DATA)

# Search or download model.py for Infersent
if not os.path.exists(os.path.join(EXTERNAL_MODELS, INFERSENT_URL.split('/')[-1])):
    try:
        pathlib.Path(os.path.split(EXTERNAL_MODELS)[0]).mkdir(parents=True)
    except FileExistsError:
        pass

    r = requests.get(INFERSENT_URL)
    if r.status_code == 200:
        with open(os.path.join(EXTERNAL_MODELS, INFERSENT_URL.split('/')[-1]), 'w') as f:
            f.write(r.text)
    else:
        raise Exception(f'Cannot download "{INFERSENT_URL}" (returned status {r.status_code})')

if not os.path.exists(GLOVE):
    raise FileNotFoundError(f'Download GloVe embeddings from https://nlp.stanford.edu/projects/glove/ ',
                            f'and save to "{GLOVE}"')

if not glob.glob(INFERSENT):
    raise FileNotFoundError(f'Unable to find any infersent model in "{INFERSENT}"')

try:
    BertTokenizer.from_pretrained(BERT)
except OSError:
    raise FileNotFoundError(f'Unable to locate model "{BERT}"')

print('All required assets have been downloaded.')
