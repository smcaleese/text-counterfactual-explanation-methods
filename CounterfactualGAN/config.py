"""Configuration file.

Args:
    SEED: seed for reproduciblity
    FOLDER: location of data, vectors and external models
    DATA: location of data
    VECTORS: location to load/save vectors
    GLOVE: location of GloVe vectors
    COUNTER_FITTED_EMBEDDINGS: counter fitted embeddings used by TextFooler
    COS_SIM: file two write cosine similarities between embeddings in COUNTER_FITTED_EMBEDDINGS
    EXTERNAL_MODELS2: location of external models (e.g. InferSent, bert-base-uncased) on Disk
    INFERSENT: location of InferSent file
    INFERSENT_URL: external models.py file of InferSent
    BERT: location of BERT model (or its name if an internet connection is available)
    UNIVERSAL_SENTENCE_ENCODER: location of USE (or an url if an internet connection is available)
    DEFAULT_HPARAMS: default hparams for training
    EXTERNAL_MODELS: location for local save of external model files
    TRAINED_MODELS: location to save trained models
"""

import os

SEED = 0
FOLDER = '/home/ACCOUNTNAME'

DATA = FOLDER + '/data'

VECTORS = FOLDER + '/vectors/'
GLOVE = VECTORS + 'glove.840B.300d.txt'
COUNTER_FITTED_EMBEDDINGS = VECTORS + 'counter-fitted-vectors.txt'
COS_SIM = VECTORS + 'cos_sim_counter_fitting.npy'

EXTERNAL_MODELS2 = FOLDER + '/models/'
INFERSENT = EXTERNAL_MODELS2 + 'infersent1.pkl'
INFERSENT_URL = 'https://raw.githubusercontent.com/facebookresearch/InferSent/master/models.py'
BERT = EXTERNAL_MODELS2 + 'bert-base-uncased/'
BERT_MED = EXTERNAL_MODELS2 + 'bert-medium-uncased/'
BERT_SMALL = EXTERNAL_MODELS2 + 'bert-small-uncased/'
BERT_TINY = EXTERNAL_MODELS2 + 'bert-tiny-uncased/'
UNIVERSAL_SENTENCE_ENCODER = EXTERNAL_MODELS2 + 'universal-sentence-encoder/'

DEFAULT_HPARAMS = {
    'batch_size': 32,
    'max_epochs': 5,
    'gpus': 1,
    'learning_rate': 2e-5,
    'adam_epsilon': 1e-8,
    'warmup_steps': 0,
    'weight_decay': 0.0,
    'num_workers': 8
}

EXTERNAL_MODELS = os.path.join(os.getcwd(), 'models/external')
RESULTS = os.path.join(os.getcwd(), 'results/')
TRAINED_MODELS = EXTERNAL_MODELS2 + 'trained/'
