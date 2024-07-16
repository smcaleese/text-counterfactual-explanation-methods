"""Infersent model for finetuning."""

import os
import torch
import torch.nn as nn
import numpy as np

from models.model import Model, as_np
from models.external.models import InferSent  # https://github.com/facebookresearch/InferSent
from config import GLOVE, INFERSENT


class InfersentModel(Model):
    def __init__(self,
                 hparams,
                 dataset,
                 embeddings=GLOVE,
                 infersent=INFERSENT):
        """Finetunable Infersent model for sequence classification.

        Args:
            hparams: hyperparameters
            dataset: dataset containing train/dev/test data
            embeddings: location of embeddings (GloVe or fastText)
            infersent: location of external Infersent model code
        """
        super().__init__(hparams, dataset)
        self.__load_infersent(embeddings, infersent,
                              self.hparams.batch_size)
        out_dim = self.infersent.enc_lstm_dim * 2
        if str(self.dataset) == 'SNLI':  # handle combined sentence embeddings
            out_dim *= 4
        self.head = nn.Sequential(nn.Linear(out_dim, out_dim),
                                  nn.Linear(out_dim, self.dataset.target_size))

    def __load_infersent(self, embeddings, infersent, batch_size):
        word_emb_dim = int(os.path.splitext(embeddings)[0].split('.')[-1].replace('d', ''))
        version = int(os.path.splitext(infersent)[0][-1])
        params_model = {'bsize': batch_size,
                        'word_emb_dim': word_emb_dim,
                        'enc_lstm_dim': 2048,
                        'pool_type': 'max',
                        'dpout_model': 0.0,
                        'version': version}
        self.infersent = InferSent(params_model)
        self.infersent.load_state_dict(torch.load(infersent))
        self.infersent.set_w2v_path(embeddings)
        data = self.dataset.get()
        X = data['X'].values if 'X' in data.columns \
            else np.stack(data[c].values for c in data.columns
                          if c.startswith('X')).reshape(1, -1)[0]
        self.infersent.build_vocab(np.unique(X.astype(str)), tokenize=True)

    def __repr__(self):
        return 'infersent'

    def encode(self, X):
        """Encode instances."""
        def encode_fn(x):
            return torch.tensor(self.infersent.encode(x, tokenize=True)).float()
        if isinstance(X, list):
            X = as_np(X)
        if X.ndim == 2 and X.shape[-1] == 2:
            u, v = encode_fn(X[:, 0]), encode_fn(X[:, 1])
            return torch.stack((u, v, torch.abs(u - v), u * v), dim=1).reshape(u.size(0), -1)
        return encode_fn(X)

    def forward_pass(self, X):
        """Single forward pass of pre-encoded instances."""
        return self.head(X)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate)
