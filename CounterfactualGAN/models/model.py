"""Generic implementation for all models.

Provides interfaces to access and encode data, and train and test models.
"""

import os
import argparse
import pytorch_lightning as pl
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from config import EXTERNAL_MODELS, SEED, TRAINED_MODELS, DEFAULT_HPARAMS


def as_np(X):
    """As numpy array."""
    if isinstance(X, list):
        X = np.array(X)
    return X.values if 'pandas' in str(type(X)) else X


class ModelWithData:
    def __init__(self, hparams, dataset, attempt_load=True):
        """Construct a model with access to data through dataloaders.

        Args:
            hparams: hyperparameters
            dataset: dataset to encode, containing train/dev/test data
            attempt_load: whether to attempt loading pre-encoded data
        """
        self.hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.dataset = dataset
        self.target_size = dataset.target_size
        self.attempt_load = attempt_load

    def encode(self, X):
        """Optionally encode dataset or batch."""
        return X

    def _dataloader(self, part: str, name: str, encode_fn: callable, **kwargs):
        """Dataloader for part. If not encoded, encode first.

        Args:
            part: train/dev/test
            name: unique name for pre-encoded vectors
            encode_fn: function to encode each part with
            **kwargs: optional arguments to pass to the PyTorch `DataLoader`
        """
        if name not in self.dataset.encoded_data.keys():
            self.dataset.encode(encode_fn, name, attempt_load=self.attempt_load)
        self.dataset.encode_mode = name
        self.dataset.part = part
        sample = kwargs.pop('sample', False)
        if sample:  # Sample up to 20,000 samples
            d = self.dataset.encoded_data[name][part]
            self.dataset.encoded_data[name][part] = d[:min(20000, len(d))]
        return DataLoader(self.dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=part == 'train',
                          num_workers=self.hparams.num_workers if hasattr(self.hparams, 'num_workers') else 0,
                          **kwargs)

    def dataloader(self, part: str, **kwargs):
        """Iterable over a part of the data."""
        return self._dataloader(part, name=str(self), encode_fn=self.encode, **kwargs)

    def train_dataloader(self, **kwargs):
        """Iterable over training data."""
        return self.dataloader('train', **kwargs)

    def val_dataloader(self, **kwargs):
        """Iterable over validation (dev) data."""
        return self.dataloader('dev', **kwargs)

    def test_dataloader(self, **kwargs):
        """Iterable over test data."""
        return self.dataloader('test', **kwargs)


class Model(ModelWithData, pl.LightningModule):
    def __init__(self, hparams, dataset, attempt_load=True):
        """General PyTorch-Lightning based model.

        Args:
            hparams: hyperparameters
            dataset: dataset containing train/dev/test data
            attempt_load: whether to attempt to load a previously finetuned model
                and pre-encoded data from disk

        Example:
            Finetune the model for four epochs, then test its performance.

            >>> import pytorch_lightning as pl
            >>> from models import Model
            >>> from config import DEFAULT_HPARAMS
            >>> from dataset import SST
            >>> model = Model(DEFAULT_HPARAMS, SST())
            >>> trainer = pl.Trainer(max_epochs=4)
            >>> trainer.fit(model)
            >>> trainer.test(model)
        """
        ModelWithData.__init__(self, hparams, dataset, attempt_load=attempt_load)
        pl.LightningModule.__init__(self)
        self.loss = nn.MSELoss() if self.target_size == 1 else nn.CrossEntropyLoss()
        self.metric = nn.MSELoss() if self.target_size == 1 else pl.metrics.classification.F1()
        self.metric_name = 'mse' if self.target_size == 1 else 'f1_score'
        self.test = None

    def __repr__(self):
        return 'model'

    @property
    def batch_size(self):
        """Size of batch."""
        if not self.training:
            return self.hparams.batch_size * 4
        return self.hparams.batch_size

    def training_step(self, batch, batch_nb):
        """Single training step."""
        X, y = batch
        loss = self.loss(self(X), y if self.dataset.target_size > 1 else y.unsqueeze(1))
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        X, y = batch
        loss = self.loss(self(X), y if self.dataset.target_size > 1 else y.unsqueeze(1))
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        """Combine validation steps to calculate the total loss."""
        loss = torch.stack([o['val_loss'] for o in outputs]).mean()
        return {'avg_val_loss': loss, 'log': {'val_loss': loss}}

    def forward_pass(self, X):
        """Single forward pass for one batch."""
        raise Exception('Implemented in subclasses.')

    def forward(self, X):
        """Single forward call."""
        if 'torch' not in str(type(X)):
            X = self.encode(X)
        if X.shape[0] > self.batch_size:
            #if self.on_gpu:
            #    print(f'  [INFO] Running {str(self)} on "{self.device}" with',
            #          f'batch size {self.batch_size}.')
            X.requires_grad = False
            with torch.no_grad():
                return torch.cat([self.forward_pass(x.to(self.device) if self.on_gpu else x)
                                  for x in DataLoader(X, batch_size=self.batch_size)])
        return self.forward_pass(X.to(self.device) if self.on_gpu else X)

    def test_step(self, batch, batch_idx):
        """Single test step"""
        X, y = batch
        y_ = self(X)
        if self.target_size > 1:
            y_ = torch.argmax(y_, dim=1)
        return {self.metric_name: self.metric(y, y_), 'len': y_.shape[0]}

    def test_epoch_end(self, outputs):
        """Average batch test metrics to calculate overall performance."""
        scores = torch.stack([o[self.metric_name] * o['len'] for o in outputs]).sum()
        avg_score = (scores / torch.tensor([o['len'] for o in outputs], device=scores.device).sum()).cpu().item()
        self.test = {self.metric_name: avg_score}
        return {self.metric: avg_score, 'log': self.test}


def train_test(model_fn,
               dataset,
               force_finetune=False,
               calculate_performance=False,
               gpu_index=0):
    """Train and (optionally) test a model.

    Args:
        model_fn: model function to instantiate
        dataset: instantiated dataset
        force_finetune: whether to force training/finetuning
        calculate_performance: whether to calculate model performance
        gpu_index: which gpu to cast model to (if available)

    Returns:
        Tuple containing (1) the model name, (2) test score and performance measure
        (if calculate_performance was True), and (3) model instance
    """
    test_score = None
    if 'hparams' not in model_fn.__init__.__code__.co_varnames:
        model = model_fn(dataset, attempt_load=not force_finetune)
        if calculate_performance:
            test_score = model.test()
    else:
        pl.seed_everything(SEED)
        model_name = str(model_fn).split(".")[-2].lower()
        model_file = os.path.join(TRAINED_MODELS, f'{model_name}_{str(dataset).lower()}.ckpt')
        load_from_disk = os.path.isfile(model_file) and not force_finetune
        if load_from_disk:
            print(f'|--> Loading model checkpoint "{model_file}"')
            model = model_fn.load_from_checkpoint(model_file, dataset=dataset)
        else:
            print(f'|--> Training a new model and saving to "{model_file}"')
            model = model_fn(DEFAULT_HPARAMS, dataset)
        trainer = pl.Trainer(max_epochs=model.hparams.max_epochs,
                             gpus=[gpu_index] if model.hparams.gpus > 0 else 0,
                             checkpoint_callback=False,
                             progress_bar_refresh_rate=10)
        if not load_from_disk:
            trainer.fit(model)
        if calculate_performance and hasattr(model, 'dataset'):
            print('|--> Testing model')
            trainer.test(model)
        if not load_from_disk:
            del model.dataset  # reduce space, faster saving
            trainer.save_checkpoint(model_file)
        test_score = model.test
    return str(model), test_score, model
