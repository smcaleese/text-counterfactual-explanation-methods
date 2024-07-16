"""CounterfactualGAN (c), 2020-2021 Marcel Robeer.

This module trains a CounterfactualGAN with the user-specified
hyperparameters.

Example:
    To run CounterfactualGAN on SST-2 with `infersent` as a black-box model allowing
    three options for each test instance:
        $ python3 counterfactual_gan.py --data sst --black_box infersent --K 3 --n_trials 1

Attributes:
    DEFAULT_HPARAMS (dict): Default values for hyperparameters (if not specified by user)
    DATASET_MAP (dict): Dataset instances and corresponding classes of whitebox models
"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
import optuna
import numpy as np

from transformers import (AutoModelWithLMHead, AutoTokenizer,
                          get_linear_schedule_with_warmup,
                          top_k_top_p_filtering)
from transformers.modeling_bert import BertOnlyMLMHead
from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.utilities.distributed import rank_zero_only
from time import time as current_time
from collections import defaultdict

from config import (SEED, BERT, BERT_MED, BERT_SMALL, BERT_TINY,
                    TRAINED_MODELS, RESULTS)
from dataset import Hatespeech, SST, SNLI
from models import (train_test, ModelWithData, HatespeechWhitebox,
                    SSTWhitebox, SNLIWhitebox, InfersentModel)
from models import BERT as BERTModel
from explanation_methods import ExplanationMethod
from utils import split_hparam, GeLU

DEFAULT_HPARAMS = {
    'data': 'sst',
    'black_box': 'whitebox',
    'batch_size': 32,
    'n_epochs': 50,
    'finetune_epochs': 4,
    'finetune_task': 'mask,swap,target_impute',
    'finetune_lm_task': 'mask',
    'training_target': 'ones',
    'lr': 0.0002,
    'g_warmup_steps': 300,
    'd_warmup_steps': 150,
    'b1': 0.9,
    'b2': 0.999,
    'max_length': 32,
    'K': 3,
    'lambda_lm': 20.0,
    'lambda_tgt_lm': 2.0,
    'lambda_tgt_d': 100.0,
    'lambda_tgt_g': 8.0,
    'lambda_rec': 1.0,
    'label_smoothing': 0.95,
    'seed': SEED,
    'gpus': 1,
    'model': 'bert-medium',
    'examples_per_epoch': 8,
    'verbose': 0,
    'dev_run': False,
    'generator_depth': 2,
    'discriminator_type': 'rnn',
    'discriminator_steps': 3,
    'gpu_offset': 0,
    'limit_vocab': True,
    'force_rerun': False,
    'n_trials': 30,
}


DATASET_MAP = {
    'hatespeech': (Hatespeech, HatespeechWhitebox),
    'sst': (SST, SSTWhitebox),
    'snli': (SNLI, SNLIWhitebox)
}


class StudentModel(ModelWithData, ExplanationMethod, pl.LightningModule):
    def __init__(self, hparams, black_box, dataset, tokenizer, generate_fn):
        ModelWithData.__init__(self, hparams, dataset)
        ExplanationMethod.__init__(self, tokenizer, dataset.target_size)
        pl.LightningModule.__init__(self)
        self.black_box = black_box
        self.generate_fn = generate_fn
        self.batch_size = self.hparams.batch_size * max(1, self.hparams.gpus)
        self.max_length = self.hparams.max_length

        self.tgt_loss = nn.MSELoss() if self.target_size == 1 else nn.BCEWithLogitsLoss()

        self.check_hparams()

    def __repr__(self):
        return 'cfgan_student'

    def paired_inputs(self):
        return str(self.dataset).lower() == 'snli'

    def check_hparams(self):
        h = self.hparams
        assert h.n_epochs >= 1, 'Training should be a mininum of 1 epochs'
        assert h.finetune_epochs >= 0, 'Finetuning should be a minimum of 0 epochs'
        assert h.training_target in ['ones', 'random'], f'Unknown training method "{h.training_target}"'
        assert h.K > 0, 'K should be a positive integer (1+)'
        assert h.lr > 0, 'Requires positive learning rate'
        assert h.g_warmup_steps > 0, 'Generator requires positive number of warmup steps'
        assert h.d_warmup_steps > 0, 'Discriminator requires positive number of warmup steps'
        assert h.b1 >= 0. and h.b1 <= 1., 'Beta 1 should be between 0 and 1'
        assert h.b2 >= 0. and h.b2 <= 1., 'Beta 1 should be between 0 and 1'
        assert h.max_length > 2, 'Maximum length should be > 2'
        assert h.label_smoothing >= 0.5 and h.label_smoothing <= 1., 'Label smoothing should be between 0.5 and 1 (no smoothing)'
        assert h.n_finetune >= 1, 'Supply at least one finetuning strategy'
        assert h.finetune_lm_task in split_hparam(h.finetune_task), 'Finetune LM tasks shoud be in the finetuning strategies'
        for finetuning in split_hparam(h.finetune_task):
            assert finetuning in ['mask', 'swap', 'random', 'target_impute'], f'Unknown finetuning strategy "{finetuning}"'
        assert h.generator_depth > 0 and h.generator_depth <= 4, 'Generator must be between 1 and 4 layers deep'
        assert h.discriminator_type in ['conv', 'bert', 'rnn'], f'Unknown decoder type "{h.discriminator_type}"'
        assert h.discriminator_steps >= 1, 'Discriminator must be trained for at least one step each epoch'
        
    def encode(self, X):
        if 'pandas' in str(type(X)):
            X = X.values
        if X.ndim == 2 and X.shape[1] == 2:  # Pairs of inputs
            X = [(x1, x2) for x1, x2 in X]
        return self.tokenizer.batch_encode_plus(X, return_tensors='pt',
                                                pad_to_max_length=True)['input_ids']
    
    def embed(self, X, add_positions=True, normalize=True):
        s = self.lm if hasattr(self, 'lm') else self
        hidden = s.encoder(X)[0]
        if add_positions:
            position_ids = torch.arange(X.size(1), dtype=torch.long, device=hidden.device)
            position_ids = position_ids.unsqueeze(0).expand(X.size())
            hidden += s.position_embeddings(position_ids)
        if normalize:
            hidden = s.norm(hidden)
        return hidden

    def decode(self, X, replace_special=False, skip_sep=True, replace_after_sep=True):
        s = [self.detokenizer(self.tokenizer.convert_ids_to_tokens(x)) for x in X]
        special = self.tokenizer.all_special_tokens
        sep_token = self.tokenizer.sep_token
        if replace_after_sep:
            # Find SEP in list, replace all characters after SEP
            s = [s_.split() for s_ in s]
            if self.paired_inputs():
                seps = [s_.index(sep_token, s_.index(sep_token) + 1) if s_.count(sep_token) >= 2 else None for s_ in s]
            else:
                seps = [s_.index(sep_token) if sep_token in s_ else None for s_ in s]
            for i, sep in enumerate(seps):
                if sep is not None:
                    s_ = ['' if j >= sep else c for j, c in enumerate(s[i])]
                    s[i] = s_
            s = [' '.join(s_) for s_ in s]
        if skip_sep:
            special = [t for t in special if t != sep_token]
        if replace_special:
            res_, s = s, []
            for s_ in res_:
                for t in special:
                    s_ = s_.replace(t, '')
                s.append(' '.join(s_.split()))
        return s

    def get_str(self, X, k=1, lm_decode=True, replace_special=False, skip_sep=False, replace_after_sep=True, combine_words=True):
        if lm_decode:  
            X_ = self.lm.decoder(X)
            X_ = torch.stack([top_k_top_p_filtering(X_[:, i], top_k=3, top_p=0.9) for i in range(X_.size(1))])
            distr = torch.distributions.Categorical(logits=X_)
            if k > 1:  # sample K and choose best later
                X_ = distr.sample((self.hparams.K, )).permute(2, 0, 1).reshape(-1, self.hparams.max_length)
            else:  # take top-1
                X_ = distr.sample().T
        else:
            X_ = X
        if self.hparams.limit_vocab and lm_decode:
            X_ = self.lm.new2vocab[X_].detach()
        s = self.decode(X_,
                        replace_special=replace_special,
                        skip_sep=skip_sep,
                        replace_after_sep=replace_after_sep)
        if combine_words:
            s = [s_.replace(' ##', '') for s_ in s]
        return s

    def black_box_fn(self, X, k=1, return_X=False):
        """Call black-box function and obtain y values.

        Args:
            X: inputs (embeddings or indices of tokens)
            k: top-k predictions to make (in case of embeddings)
            return_X: return decoded indices after decoding
                (may be useful when using top-k decoding to see the
                 actual strings used for black-box values y)

        Returns:
            Predictive values y for X if return_X is False, else (X, y)
        """
        # Prepare for call to black-box
        device = X.device.index
        X = self.get_str(X,
                         k=k,
                         lm_decode=hasattr(self, 'lm'),
                         replace_special=True,
                         replace_after_sep=True,
                         skip_sep=self.paired_inputs())
        if self.paired_inputs():
            X_ = []
            for x in X:
                x_ = x.split(self.tokenizer.sep_token)
                if len(x_) == 1:
                    x_ += ['']
                X_.append([x_[0].strip(), x_[1].strip()])
            X = X_
        
        if hasattr(self.black_box, 'device') and 'torch' in str(type(X)):
            X = X.to(self.black_box.device)

        # Do actual call
        y = self.black_box(X)

        # Return to format to be used by CounterfactualGAN
        if 'torch' not in str(type(y)):
            y = torch.tensor(y)
        if self.on_gpu:
            y = y.to(device)
        if self.target_size == 1:
            y = y.T
            if y.ndim > 1:
                y = y.squeeze()
        if y.dtype == torch.float64:
            y = y.float()

        # Group if K > 1
        if hasattr(self, 'lm') and k > 1: 
            y = y.reshape(-1, self.hparams.K) if self.target_size == 1 else y.reshape(-1, self.hparams.K, y.size(-1))
            X = [x for x in zip(*(iter(X),) * self.hparams.K)]

        if return_X:
            return X, y
        return y

    def to_max_length(self, X):
        """Cut-off or pad to maximum length."""
        if X.size(-1) < self.max_length:
            X = nn.functional.pad(X, (0, self.max_length - X.size(-1)))
        elif X.size(-1) > self.max_length:
            X = X[:, :self.max_length]
        return X

    def get_bin(self, y, n_bins=4, single_instance=False):
        """Find which bin (class or range of regression values) an y value is in."""
        if self.target_size > 1:
            if y.ndim > 1 or single_instance:
                return torch.argmax(y, axis=-1)
            return y
        if not hasattr(self, 'bins') or self.bins[0] != n_bins:
            self.bins = (n_bins, torch.linspace(y.min(), y.max(), steps=n_bins + 1)[1:])
        mask = torch.zeros_like(y, dtype=int)
        for _min in self.bins[1]:
            mask = mask.add((y >= _min).long())
        if mask.ndim > 1:
            mask = mask.squeeze()
        return mask

    def train_dataloader(self):
        """Load training data."""
        torch.manual_seed(self.hparams.seed)

        def collate_fn(data):
            X = self.to_max_length(torch.stack([d[0] for d in data]))

            if hasattr(self, 'lm'):
                X_ = X.clone()
                if self.on_gpu:
                    X_ = X_.to(self.lm.encoder.device)
                X_ = self.embed(X_)
                y = self.black_box_fn(X_)
            else:
                y = self.black_box_fn(X)
                X_tasks = {}

                # Predict mask
                if 'mask' in self.hparams.finetune_task:
                    X_ = X.clone()
                    X_.masked_fill_(torch.FloatTensor(X.shape).uniform_() < 0.2,
                                    self.tokenizer.mask_token_id)
                    X_tasks['mask'] = X_

                # Swap two values
                if 'swap' in self.hparams.finetune_task:
                    X_ = X.clone()
                    seps = (X_ == self.tokenizer.sep_token_id).nonzero()
                    for i, j in enumerate(seps[:, 1]):
                        if j > 2:
                            a, b = torch.randperm(j - 1)[:2] + torch.tensor([1, 1])
                            try:
                                X[i, a], X[i, b] = X[i, b], X[i, a].clone() 
                            except:
                                pass
                    X_tasks['swap'] = X_

                # Random target
                if 'random' in self.hparams.finetune_task:
                    X_ = X.clone()
                    X_.masked_scatter_(torch.FloatTensor(X.shape).uniform_() < 0.15,
                                       torch.randint(0, self.vocab_size, X.shape))
                    X_tasks['random'] = X_

                # Impute with word in other target (bin or class)
                if 'target_impute' in self.hparams.finetune_task:
                    X_ = X.clone()
                    for i in range(X_.size(0)):
                        choices = torch.randperm(self.probs[-1].shape[0])
                        tgt = choices[choices != self.get_bin(y[i].cpu(), single_instance=True)][0]
                        X_[i].masked_scatter_(torch.FloatTensor(X_[i].shape).uniform_() < 0.3,
                                              torch.multinomial(probs[tgt], X_[i].shape[0]))
                    X_tasks['target_impute'] = X_

                return (X, X_tasks, y)
            return (X, X_, y)

        dl = super().train_dataloader(sample=self.paired_inputs(), collate_fn=collate_fn)

        # Reduce vocab size
        self.__reduce_vocab_size()
        
        # Make bins for target impute
        if not hasattr(self, 'lm') and 'target_impute' in self.hparams.finetune_task:
            X = self.dataset.encoded_data[str(self)]['train']
            if X.size(0) > 8000:  # Limit huge datasets
                X = X[torch.randperm(X.size(0))[:8000], :]
            y = self.get_bin(self.black_box_fn(X))
            ids, inv, cnt = X.unique(return_inverse=True, return_counts=True)

            res = []
            for y_ in y.unique():
                inv_ids, cnt_ids_ = inv[y == y_].unique(return_counts=True)
                cnt_ids = torch.zeros_like(cnt)
                cnt_ids[inv_ids] = cnt_ids_
                res.append(cnt_ids / cnt.float())

            res = torch.stack(res)

            # Exclude special ids
            for i, id in enumerate(ids):
                if id in self.tokenizer.all_special_ids:
                    res[:, i] = 0.0

            probs = (res.T / res.sum(dim=-1)).T
            self.probs = (ids, probs)

        return dl
    
    def __reduce_vocab_size(self):
        """Reduce vocab size and create mappings from and to the new vocab."""
        if hasattr(self, 'lm') and hasattr(self.lm, 'vocab2new'):
            self.vocab2new = self.lm.vocab2new
            self.new2vovab = self.lm.new2vocab
        else:
            all_ids = self.dataset[:][0]
            new_ids = torch.cat((torch.tensor(self.tokenizer.all_special_ids),
                                 all_ids.reshape(-1))).unique()
            self.vocab_size = len(new_ids)
            print(f'Reduced vocab size from {self.tokenizer.vocab_size} to {self.vocab_size}',
                  f'({(self.vocab_size / self.tokenizer.vocab_size) * 100:.2f}%)')

            self.vocab2new = torch.full((self.tokenizer.vocab_size,),
                                        self.tokenizer.unk_token_id,
                                        dtype=int)
            self.vocab2new[new_ids] = torch.arange(len(new_ids))
            self.new2vocab = new_ids
            if self.on_gpu:
                self.vocab2new = self.vocab2new.to(self.device)
                self.new2vocab = self.new2vocab.to(self.device)


class LanguageModel(StudentModel):
    def __init__(self, hparams, black_box, dataset, tokenizer, model):
        super().__init__(hparams, black_box, dataset, tokenizer, model._generate_no_beam_search)
        self.encoder = model.base_model
        self.encoder.train()
        self.config = self.encoder.config
        self.position_embeddings = nn.Embedding(self.config.max_position_embeddings,
                                                self.config.hidden_size)
        self.norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.is_trained = False
        self.__lm_head = model.cls if hasattr(model, 'cls') else model.lm_head
  
        # Define discriminator
        self.discriminator = Discriminator(self.hparams,
                                           self.config,
                                           self.tgt_loss,
                                           target_size=self.target_size)

    def on_train_start(self, *args, **kwargs):
        """Only define decoder when we know the vocab size."""
        cfg = self.config
        if self.hparams.limit_vocab:
            cfg.vocab_size = self.vocab_size
        vocab_map = self.vocab2new if self.hparams.limit_vocab else None
        vocab_map_inv = self.new2vocab if self.hparams.limit_vocab else None
        self.decoder = Decoder(cfg, self.__lm_head, vocab_map, vocab_map_inv)
        if self.on_gpu:
            self.decoder = self.decoder.to(self.device)
        self._start_time = current_time()

    def __repr__(self):
        return 'cfgan_lm'

    def forward(self, X, X_true=None, y=None, valid_or_fake=None):
        """Single forward pass."""
        hidden = self.embed(X)

        # L_lm
        if X_true is not None:
            _, y_lm = self.decoder(hidden, lm_labels=X_true)
        else:
            y_lm = 0.0

        # L_adv, L_tgt
        if y is not None and y.ndim > 2:
            y = y[0]
        _, (y_adv, y_tgt) = self.discriminator(hidden, y_adv=valid_or_fake, y_tgt=y)

        return (y_lm, y_adv, y_tgt)

    def configure_optimizers(self):
        """Configure optimizers for use by PyTorch Lightning."""
        no_decay = ["bias", "LayerNorm.weight"]
        named_params = self.encoder.named_parameters()
        self.optimizer_grouped_parameters = [
            {
                "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {"params": [p for n, p in named_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        self.opt = torch.optim.AdamW(self.optimizer_grouped_parameters,
                                     lr=self.hparams.lr)
        return self.opt

    def training_step(self, batch, batch_nb):
        """Single training step for a batch."""
        instances, task_instances, labels = batch

        def to_gpu(x):
            return x.to(instances.device) if self.on_gpu else x

        def adv_tensor(offset=0.0):
            ls = offset - self.hparams.label_smoothing
            return to_gpu(torch.Tensor(instances.size(0), 1).fill_(ls))

        valid = adv_tensor()
        fake = adv_tensor(1.0)

        _, adv_valid, tgt_loss = self.forward(instances,
                                              y=labels,
                                              valid_or_fake=valid)
        adv_fake = []
        for task in task_instances.keys():
            y = self.black_box_fn(task_instances[task]) if task == 'target_impute' else None
            if task == self.hparams.finetune_lm_task:
                lm_loss, adv_fake_, tgt_loss_ = self.forward(task_instances[task],
                                                             y=y,
                                                             X_true=instances,
                                                             valid_or_fake=fake)
            else:
                _, adv_fake_, tgt_loss_ = self.forward(task_instances[task],
                                                       y=y,
                                                       valid_or_fake=fake)
            adv_fake.append(adv_fake_)
            if y is not None:
                tgt_loss = (tgt_loss + tgt_loss_) / 2

        adv = [adv_valid] + adv_fake
        adv_loss = sum(adv) / len(adv)

        log = {'lm_loss': lm_loss, 'adv_loss': adv_loss, 'tgt_loss': tgt_loss}
        loss = self.hparams.lambda_lm * lm_loss + adv_loss + self.hparams.lambda_tgt_lm * tgt_loss
        return {'loss': loss, 'progress_bar': log, 'log': log}

    def on_train_end(self):
        """Called when training ends. Fixes encoder in current state."""
        self.encoder.eval()
        self.is_trained = True


class InitModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=self.config.initializer_range)


class Decoder(InitModule):
    def __init__(self, config, lm_head,
                 vocab_map=None,
                 vocab_map_inv=None):
        """Decoder (hidden embedding to predictions for each token)."""
        super().__init__(config)
        if vocab_map is not None:
            self.lm_head = BertOnlyMLMHead(config)
            self.head_teacher = lm_head
        else:
            self.lm_head = lm_head
        self.lm_loss = nn.CrossEntropyLoss()
        self.vocab_map = vocab_map
        self.vocab_map_inv = vocab_map_inv

    def forward(self, X, lm_labels=None):
        out = self.lm_head(X)

        if lm_labels is not None:
            if self.vocab_map is not None:  # Replace with new vocab IDs
                lm_labels = self.vocab_map[lm_labels].detach()
                lm_teacher = self.head_teacher(X).detach()[:, :, self.vocab_map_inv]
            out = out, self.lm_loss(out.view(-1, self.config.vocab_size), lm_labels.flatten())
            if self.vocab_map is not None:
                out = out[0], 0.5 * out[1] + 0.5 * nn.MSELoss()(out[0], lm_teacher)

        return out


class Discriminator(InitModule):
    def __init__(self,
                 hparams,
                 config,
                 tgt_loss,
                 target_size=2):
        """Discriminator, with an output `tgt` (probability of a class or regression values),
        and `adv` (likelihood that the instance is real)."""
        super().__init__(config)
        self.hparams = hparams
        self.target_size = target_size

        hidden = self.config.hidden_size
        if self.hparams.discriminator_type == 'conv':
            self.shared = nn.Sequential(
                nn.Conv2d(self.hparams.max_length, 2, 2, padding=1),
                GeLU(),
                nn.Conv2d(2, 1, 2)
            )
        elif self.hparams.discriminator_type == 'rnn':
            self.shared = nn.GRU(hidden,
                                 hidden,
                                 num_layers=2,
                                 batch_first=True,
                                 dropout=0.1)
        else:
            shared_layer = nn.TransformerEncoderLayer(hidden, nhead=config.num_attention_heads, activation='gelu')
            self.shared = nn.TransformerEncoder(shared_layer, 2)
        self.adv_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            GeLU(),
            nn.Linear(hidden, 1)
        )
        modules = [nn.Linear(hidden, hidden),
                   GeLU(),
                   nn.Linear(hidden, self.target_size)]
        if self.target_size > 1:
            modules.append(nn.Softmax(dim=-1))
        self.cls_head = nn.Sequential(*modules)

        self.adv_loss = nn.MSELoss()
        self.tgt_loss = tgt_loss

    def forward(self, Z, y_adv=None, y_tgt=None):
        if self.hparams.discriminator_type == 'conv':
            Z_ = self.shared(Z.unsqueeze(2)).squeeze(2).squeeze(1)
        elif self.hparams.discriminator_type == 'rnn':
            Z_ = self.shared(Z)[-1][0, :]
        else:
            Z_ = self.shared(Z)[:, 0]

        out = (self.adv_head(Z_),
               self.cls_head(Z_))

        if y_adv is not None or y_tgt is not None:
            adv_loss, tgt_loss = 0.0, 0.0
            if y_adv is not None:
                adv_loss = self.adv_loss(out[0], y_adv)
            if y_tgt is not None:
                if self.target_size == 1 and y_tgt.ndim == 1:
                    y_tgt = y_tgt.unsqueeze(1)
                tgt_loss = self.tgt_loss(out[1], y_tgt)
            out = (out, (adv_loss, tgt_loss))

        return out


class Generator(nn.Module):
    def __init__(self, hparams, config, target_size=2, margin=0.5):
        """Generator: takes hidden layer and transforms it to a new equal size hidden layer of target."""
        super().__init__()
        self.max_length = hparams.max_length

        decoder_layer = nn.TransformerDecoderLayer(config.hidden_size,
                                                   nhead=max(config.num_attention_heads // 3, 1),
                                                   activation='gelu')
        self.model = nn.TransformerDecoder(decoder_layer, hparams.generator_depth)
        self.target_size = target_size
        self.target_encoder = nn.Linear(self.target_size, config.hidden_size)  
        self.dist = nn.CosineEmbeddingLoss(margin=margin)

    def encode_target(self, y_target):
        """Encode the target."""
        if self.target_size > 1:
            y_target.squeeze_()
        else:
            if y_target.ndim == 1:
                y_target = y_target.unsqueeze(1)
            elif y_target.shape[0] == 1:
                y_target = y_target.T
        return self.target_encoder(y_target.float())

    def forward(self, Z, y_target=None, Z_compare=None):
        """Single forward pass."""
        Z_targeted = Z.clone()

        if y_target is None:  # random target
            y_target = torch.empty(Z.size(0), self.target_size, device=Z.device).uniform_()
        Z_targeted[:, 0] = self.encode_target(y_target)
        out = self.model(Z, Z_targeted)

        if Z_compare is not None:
            eq = torch.ones(out.size(0), device=out.device)
            out = (out, self.dist(out.view(out.size(0), -1), Z_compare.view(out.size(0), -1), eq))

        return out


class CounterfactualGAN(StudentModel):
    def __init__(self, hparams, black_box, tokenizer, dataset, lm):
        super().__init__(hparams, black_box, tokenizer, dataset, lm.generate_fn)
        self.lm = lm

        _config = self.lm.encoder.config
        self.generator = Generator(self.hparams,
                                   _config,
                                   target_size=self.target_size)
        self.discriminator = self.lm.discriminator
        self.val_target = self.dataset.target(part='dev',
                                              return_type='pt',
                                              seed=self.hparams.seed)
        self.test_target = self.dataset.target(part='test',
                                               return_type='pt',
                                               seed=self.hparams.seed)

    def __repr__(self):
        return 'counterfactualgan'

    def forward(self, Z, y, Z_=None):
        """Forward pass of generator G.

        Args:
            Z: embedding of original instance
            y: target values for each instance
            Z_: ground-truth embedding after generation
        """
        return self.generator.forward(Z, y_target=y, Z_compare=Z_)

    def backward(self, use_amp, loss, optimizer, opt_idx):
        """Backward pass."""
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    def configure_optimizers(self):
        """Configure optimizers and schedulers for generator G and discriminator D."""
        lr = self.hparams.lr
        betas = (self.hparams.b1, self.hparams.b2)

        # Optimizers
        optimizer = torch.optim.Adam
        opt_d = optimizer(self.discriminator.parameters(),
                          lr=lr,
                          betas=betas)
        opt_g = optimizer(self.generator.parameters(),
                          lr=lr,
                          betas=betas)

        # Schedulers
        scheduler = get_linear_schedule_with_warmup
        sched_d = scheduler(opt_d,
                            num_warmup_steps=self.hparams.d_warmup_steps,
                            num_training_steps=self.hparams.n_epochs)
        sched_g = scheduler(opt_g,
                            num_warmup_steps=self.hparams.g_warmup_steps,
                            num_training_steps=self.hparams.n_epochs)
        return [opt_d, opt_g], [sched_d, sched_g]

    def generate_targets(self, labels):
        """Generate targets based on label and training target strategy (ones/random)."""
        if self.target_size == 1:
            if self.hparams.training_target == 'ones':
                target_labels = 1.0 - labels.clone()
                target_labels[target_labels < 0.5] = 0.0
                target_labels[target_labels >= 0.5] = 1.0
            else:
                target_labels = torch.rand(labels.size(0), device=labels.device)
        else:
            if self.hparams.training_target == 'ones':
                random_tgt = torch.randint(0, self.target_size, (labels.size(0),))
                target_labels = torch.eye(self.target_size, device=labels.device)[random_tgt]
                target_labels *= self.hparams.label_smoothing
                target_labels[target_labels == 0.] = (1. - self.hparams.label_smoothing) / (self.target_size - 1)
            else:
                target_labels = labels[torch.randperm(labels.size(0))].clone()
        return target_labels
    
    def d_loss(self, instances, labels, target_labels, valid, fake):
        """Discriminator loss."""
        if self.hparams.discriminator_type != 'rnn':
            self.discriminator.train()
            self.generator.eval()

        # Generate counterfactual
        cf = self.generator(instances, target_labels).detach()

        adv_loss, tgt_loss = 0.0, 0.0
        n_steps = self.hparams.discriminator_steps
        for _ in range(n_steps):
            _, (x_adv, x_tgt) = self.discriminator(instances,
                                                   y_adv=valid,
                                                   y_tgt=labels)
            _, (y_adv, _) = self.discriminator(cf, y_adv=fake)

            # Kullback-Leibler
            tgt_loss += x_tgt / n_steps
            adv_loss += ((x_adv + y_adv) / 2) / n_steps

        return adv_loss, tgt_loss

    def g_loss(self, instances, target_labels, fake):
        """Generator loss."""
        if self.hparams.discriminator_type != 'rnn':
            self.discriminator.eval()
            self.generator.train()

        # Single pass with generator
        new_instances = self.forward(instances, target_labels)
        
        # L_tgt
        y_target = self.black_box_fn(new_instances)
        if self.on_gpu:
            y_target = y_target.to(target_labels.device)
        tgt_loss = self.tgt_loss(y_target, target_labels)

        # L_adv
        _, (adv_loss, _) = self.discriminator(new_instances,
                                              y_adv=fake) 

        return new_instances, adv_loss, tgt_loss

    def training_step(self, batch, batch_nb, optimizer_idx):
        """Single training step for optimizer (discriminator/generator)."""
        instances, encoded_instances, labels = batch
        if self.hparams.limit_vocab:
            instances = self.lm.decoder.vocab_map[instances].detach()

        def adv_tensor(offset=0.0):
            ls = offset - self.hparams.label_smoothing
            return torch.full((instances.size(0), 1), ls, device=instances.device)

        # Make tensors for valid and fake instances
        valid = adv_tensor()
        fake = adv_tensor(1.0)

        # D: Discriminator
        if optimizer_idx == 0:
            # Create target labels
            self.target_labels = self.generate_targets(labels)

            # Calculate discriminator loss
            adv_loss, tgt_loss = self.d_loss(encoded_instances,
                                             labels,
                                             self.target_labels,
                                             valid,
                                             fake)
            loss = adv_loss + self.hparams.lambda_tgt_d * tgt_loss
            log = {'d_tgt': tgt_loss,
                   'd_adv': adv_loss,
                   'd_loss': loss}
        # G: Generator
        else:
            # Use pre-generated target labels and instances to map to target
            # and cycle back afterward s
            self.last_labels, self.last_instances = labels, encoded_instances
            self.counterfactual_instances, adv_loss_cf, tgt_loss_cf = self.g_loss(encoded_instances,
                                                                                  self.target_labels,
                                                                                  fake)
            self.cycled_instances, adv_loss_rec, tgt_loss_rec = self.g_loss(self.counterfactual_instances,
                                                                            self.last_labels,
                                                                            fake)
            adv_loss = (adv_loss_cf + adv_loss_rec) / 2
            tgt_loss = (tgt_loss_cf + tgt_loss_rec) / 2

            # Regularization loss
            cycled = self.lm.decoder(self.cycled_instances).view(-1, self.lm.vocab_size)
            rec_loss = 0.5 * nn.CrossEntropyLoss()(cycled, instances.flatten()) + \
                       0.5 * nn.MSELoss()(encoded_instances, self.cycled_instances)
            loss = adv_loss +  self.hparams.lambda_tgt_g * tgt_loss + self.hparams.lambda_rec * rec_loss
            log = {'g_adv': adv_loss,
                   'g_tgt': tgt_loss,
                   'g_rec': rec_loss,
                   'g_loss': loss}
        return {'loss': loss, 'progress_bar': log, 'log': log}

    def __print_examples(self, n, replace_special=True):
        if VERBOSE > 0 and hasattr(self, 'last_instances'):
            n = min(len(self.last_labels), max(0, n))

            o = self.get_str(self.last_instances[:n], replace_special=replace_special)
            c = self.get_str(self.counterfactual_instances[:n], replace_special=replace_special)

            def get_label(x, i):
                if self.target_size > 1:
                    return int(torch.argmax(x[i]).item())
                return round(x[i].item(), 3)

            print('')
            for i in range(n):
                original = get_label(self.last_labels, i)
                target = get_label(self.target_labels, i)
                to_print = f'  {i} "{o[i]}" ({original} to {target}) '
                if original == target:
                    to_print += f'-copy-> "{c[i]}"'
                else:
                    to_print += f'-CF-> "{c[i]}"'
                print(to_print)

    def on_epoch_end(self):
        if VERBOSE > 0:
            print(f'\n{"#" * 10}\nEPOCH {self.current_epoch}:\n{"#" * 10}')
        self.__print_examples(self.hparams.examples_per_epoch)

    def on_train_end(self):
        self.training_time = current_time() - self.lm._start_time
        self.__print_examples(self.hparams.batch_size)

    def __val_test_dataloader(self, part):
        """Dataloader for validation/test set."""
        assert part in ['val', 'test']
        def collate_fn(data):
            return self.to_max_length(torch.stack([d[0] for d in data]))
        dl = super().val_dataloader if part == 'val' else super().test_dataloader
        return dl(collate_fn=collate_fn)

    def val_dataloader(self):
        return self.__val_test_dataloader('val')

    def test_dataloader(self):
        return self.__val_test_dataloader('test')

    def __val_test_step(self, batch, batch_idx, part='test'):
        """Single step for validation/testing."""
        assert part in ['val', 'test']

        def to_gpu(x):
            return x.to(batch.device) if self.on_gpu else x

        # Get targets
        tgt = self.val_target if part == 'val' else self.test_target
        target = to_gpu(tgt[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size].clone())
        target = target if self.target_size == 1 else torch.eye(self.target_size, device=target.device)[target]

        # Encode batch and generate counterfactuals        
        Z_ = self.forward(self.embed(batch), target)

        # Get a top-k decoded counterfactuals X_ and their corresponding predictions y_
        X_, y_ = self.black_box_fn(Z_, k=self.hparams.K, return_X=True)
        y_ = to_gpu(y_)
        
        if self.hparams.K == 1:
            return target, X_, y_

        # Pick the highest fidelity counterfactual from the top-k (k > 1) counterfactuals
        pdist = nn.PairwiseDistance(p=2)
        tgt = target.unsqueeze(1) if self.target_size == 1 else target
        best_ys = torch.stack([pdist(tgt, y_[:, k]) for k in range(self.hparams.K)]).T.argmin(-1)
        best_xs = [x[i] for x, i in zip(X_, best_ys)]
        return target, best_xs, y_[torch.arange(y_.size(0)), best_ys]

    def validation_step(self, batch, batch_idx):
        target, _, y_ = self.__val_test_step(batch, batch_idx, part='val')
        return {'batch_loss': self.discriminator.tgt_loss(y_, target),
                'batch_len': batch.size(0)}

    def validation_epoch_end(self, outputs):
        total_len = sum(output['batch_len'] for output in outputs)
        total_loss = sum(output['batch_loss'] * output['batch_len'] for output in outputs)
        loss = 1.0 - (total_loss / total_len)
        if VERBOSE > 1:
            print('CURRENT LOSS:', loss.item())
        return {'log': {'val_loss': loss}}

    def repeats(self, X):
        """Number of repeats per instance."""
        if self.paired_inputs():
            X = [' '.join(x) for x in X]
        res = []
        for x in X:
            r = 0
            x_ = x.split(' ')
            for i in range(len(x_) - 1):
                if x_[i] == x_[i + 1]:
                    r += 1
            res.append(r)
        res = np.array(res)
        return np.mean(res), res
    
    def test_step(self, batch, batch_idx):
        start_time = current_time()
        target, cfs, y_cf = self.__val_test_step(batch, batch_idx, part='test')

        if self.on_gpu:
            batch = batch.cpu()
            target = target.cpu()
            y_cf = y_cf.cpu()
        X = self.get_str(batch, lm_decode=False, replace_special=True, combine_words=True)
        if batch_idx == 0:
            print(cfs, X, target)
        similarity, sims = self.similarity(cfs, X)
        repeats, reps = self.repeats(cfs)
        return {'similarity': similarity * batch.size(0),
                'repeats': repeats,
                'sims': sims,
                'reps': reps,
                'cfs': cfs,
                'target': target if self.target_size == 1 else torch.argmax(target, axis=1),
                'y_cf': y_cf,
                'inference_time': current_time() - start_time}        

    def test_epoch_end(self, outputs):
        """Combine all test batches to a final result, write to DataFrame .json file."""
        import pandas as pd

        def total(i):
            return sum(output[i] for output in outputs)
        sims = np.concatenate([output['sims'] for output in outputs])
        reps = np.concatenate([output['reps'] for output in outputs])

        y_cf = torch.cat([output['y_cf'] for output in outputs])
        target = torch.cat([output['target'] for output in outputs])
        cfs = [i for l in [output['cfs'] for output in outputs] for i in l]

        y_cf_ = torch.argmax(y_cf, axis=-1) if self.target_size > 1 else y_cf
        print(torch.unique(y_cf_, return_counts=True))

        k = self.hparams.K
        fidelity = self.fidelity(target, y_cf)
        similarity = total('similarity') / len(self.dataset)
        repeats = total('repeats') / len(self.dataset)
        model = str(self.hparams.black_box).lower()
        dataset = str(self.hparams.data).lower()
        res = {'model': model,
               'dataset': dataset,
               'explanation_method': f'counterfactualgan (top-{k})',
               'seed': self.hparams.seed,
               'target_seed': self.hparams.seed,
               'similarity': similarity,
               'similarity_std': np.std(sims),
               'repeats': repeats,
               'repeats_std': np.std(reps),
               'X_sim': sims,
               'performance_measure': 'MSE' if self.target_size == 1 else 'F1-score',
               'fidelity': fidelity,
               'training_time': self.training_time,
               'inference_time': total('inference_time'),
               'counterfactuals': cfs,
               'y_target': np.array(target),
               'y_cf': np.array(y_cf),
               'hparams': self.hparams}
        if fidelity > 0.0:
            pd.DataFrame([res]).to_json(RESULTS + f'counterfactuals_{dataset}_{model}_top-{k}_{fidelity}_counterfactualgan.json')
        ret = {k: res[k] for k in ['fidelity', 'similarity', 'similarity_std', 'repeats', 'repeats_std']}
        ret['log'] = {'test_fidelity': fidelity,
                      'test_similarity': similarity,
                      'test_repeats': repeats}
        return ret


class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

    def on_test_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def get_config():
    """Configuration from terminal."""
    parser = ArgumentParser(description='CounterfactualGAN. (c) 2020-2021 ANON')
    def arg(name, type, values, description, **kwargs):
        default = DEFAULT_HPARAMS[name]
        return parser.add_argument(f'--{name}',
                                   type=type,
                                   default=default,
                                   metavar=values,
                                   help=f'{description} (default={default})',
                                   **kwargs)

    # General options
    arg('data', str, '(hatespeech, sst, snli)', 'dataset to use')
    arg('black_box', str, '(whitebox, infersent, bert)', 'black-box model to explain')
    arg('model', str, '(bert, bert-medium, bert-small, bert-tiny, ...)',
        'name of language model from Transformers package')
    arg('K', int, '[1+]', 'number of times to sample final counterfactuals')
    arg('examples_per_epoch', int, '[0+]',
        'number of examples to print after each training epoch')
    arg('gpu_offset', int, '[0-3]', 'which gpu to use if using a single gpu')
    arg('seed', int, None, 'seed for reproducibility')
    arg('verbose', int, '[0-2]',
        'verbosity level (0=no print, 1=somewhat verbose, 2=full verbose)')
    arg('dev_run', bool, 'True/False',
        'set to True for dev run (one epoch for each phase)')
    arg('force_rerun', bool, 'True/False',
        'set to True to prohibit loading a pretrained model from disk')
    arg('n_trials', int, '[1+]',
        'number of trials for hyperparameter search (1 = no search)')

    # Training options
    arg('n_epochs', int, '[1+]', 'number of training epochs')
    arg('finetune_epochs', int, '[0+]',
        'number of finetuning epochs for language model')
    arg('finetune_task', str, '(mask, swap, random, target-impute)',
        'types of tasks during the finetuning step') 
    arg('finetune_lm_task', str, '(mask, swap, random)',
        'which finetune task to fit the language model head on')
    arg('training_target', str, '(ones, random)',
        'type of goal for GAN training task')
    arg('batch_size', int, '[1+]', 'batch size per CPU/GPU')
    arg('max_length', int, '[1+]', 'maximum length of each instance')
    arg('gpus', int, '[0+]', 'number of GPUs to train on')
    arg('label_smoothing', int, '[0-1]',
        'label smoothing to prevent overconfidence, where 1.0 is no smoothing')

    # Optimizer
    arg('lr', float, '[0-1]', 'Adam: learning rate')
    arg('g_warmup_steps', int, '[0+]', 'Adam: Number of warmup steps for the generator')
    arg('d_warmup_steps', int, '[0+]', 'Adam: Number of warmup steps for the discriminator')
    arg('b1', float, '[0+]', 'Adam: beta 1')
    arg('b2', float, '[0+]', 'Adam: beta 2')

    # Relative weights of losses
    arg('lambda_lm', float, '[0+]', 'relative weight of language model objective')
    arg('lambda_tgt_lm', float, '[0+]', 'relative weight of target loss (language model)')
    arg('lambda_tgt_g', float, '[0+]', 'relative weight of target loss (generator)')
    arg('lambda_tgt_d', float, '[0+]', 'relative weight of target loss (discriminator)')
    arg('lambda_rec', float, '[0+]', 'relative weight of change reconstruction loss')

    # Generator/discriminator/decoder options
    arg('generator_depth', int, '1-4', 'Generator: number of layers')
    arg('discriminator_type', str, '(conv, bert, rnn)',
        'Discriminator: type of shared layer')
    arg('discriminator_steps', int, '[1+]',
        'Discriminator: number of steps to train in each epoch')
    arg('limit_vocab', bool, 'True/False',
        'Decoder: limit vocab to only tokens present in training data')

    return parser.parse_args()


def main(hparams=DEFAULT_HPARAMS):
    if not isinstance(hparams, Namespace):
        hparams = Namespace(**hparams)

    hparams.n_finetune = len(split_hparam(hparams.finetune_task))
    hparams.n_trials = max(1, hparams.n_trials)
    global VERBOSE
    VERBOSE = hparams.verbose
    del hparams.verbose
    if VERBOSE < 2:
        rank_zero_only.rank = -1

    # Set seeds (reproducibility)
    pl.seed_everything(hparams.seed)

    if hparams.model == 'bert':
        hparams.model = BERT
    elif hparams.model == 'bert-medium':
        hparams.model = BERT_MED
    elif hparams.model == 'bert-small':
        hparams.model = BERT_SMALL
    elif hparams.model == 'bert-tiny':
        hparams.model = BERT_TINY

    # Printing
    def print_phase(i, name, epochs=None):
        if VERBOSE > 0:
            if epochs is None:
                print(f'[{i}] {name}')
            else:
                ep = 'a single development run' if hparams.dev_run else f'{epochs} epochs'
                print(f'[{i}] {name} for {ep}')

    print_phase(1, f'Loading dataset "{hparams.data}" and model {hparams.black_box}"')
    assert str(hparams.data).lower() in DATASET_MAP.keys(), 'Unknown dataset'
    assert str(hparams.black_box).lower() in ['whitebox', 'infersent', 'bert'], 'Unknown model'
    dataset, black_box = DATASET_MAP[str(hparams.data).lower()]
    dataset = dataset()
    if str(hparams.black_box).lower() != 'whitebox':
        black_box = InfersentModel if str(hparams.black_box).lower() == 'infersent' else BERTModel
    _, _, black_box = train_test(black_box, dataset)
    if hasattr(black_box, 'hparams.batch_size'):  # Align batch_size of CFGAN and black-box
        black_box.hparams.batch_size = hparams.batch_size
    tokenizer = AutoTokenizer.from_pretrained(hparams.model)

    def objective(trial=None):
        # Seed
        pl.seed_everything(hparams.seed)
        torch.random.manual_seed(hparams.seed)

        # Choices
        if trial is not None:
            hparams.finetune_epochs = trial.suggest_int('finetune_epochs', 7, 10)
            hparams.n_epochs = trial.suggest_int('n_epochs', 30, 100)
            ct = [1.0, 3.0, 5.0, 10.0, 15.0, 25.0, 50.0]
            hparams.lambda_lm = trial.suggest_categorical('lambda_lm', ct + [100.0])
            hparams.lambda_tgt_lm = trial.suggest_categorical('lambda_tgt_lm', ct)
            hparams.lambda_tgt_d = trial.suggest_categorical('lambda_tgt_d', ct)
            hparams.lambda_tgt_g = trial.suggest_categorical('lambda_tgt_g', ct)
            hparams.lambda_rec = trial.suggest_categorical('lambda_rec', ct)

        # Set save path
        model_path = os.path.join(TRAINED_MODELS, f'{str(dataset).lower()}_{str(black_box)}_')
        exclude = ['force_rerun', 'verbose', 'n_epochs', 'gpus', 'examples_per_epoch', 'gpu_offset']

        def get_hash(to_exclude):
            return hash(frozenset([i for i in vars(hparams).items() if i[0] not in to_exclude]))

        id_lm = get_hash(exclude + ['lambda_tgt_g', 'lambda_rec', 'generator_depth',
                                    'warmup_steps', 'n_epochs'])
        id_cf = get_hash(exclude)

        # Phase 1: Language model finetuning on dataset
        model_file = model_path + f'LM{id_lm}.ckpt'
        print_phase(2, 'Finetuning LM', hparams.finetune_epochs)
        progress_bar_refresh_rate = 50 if VERBOSE == 0 else 10

        model_pretrained = AutoModelWithLMHead.from_pretrained(hparams.model)
        if os.path.isfile(model_file) and not hparams.force_rerun:
            print(f'> Loading LM from file "{model_file}"')
            lm = LanguageModel.load_from_checkpoint(model_file,
                                                    black_box=black_box,
                                                    dataset=dataset,
                                                    tokenizer=tokenizer,
                                                    model=model_pretrained)
        else:
            lm = LanguageModel(hparams,
                               black_box=black_box,
                               dataset=dataset,
                               tokenizer=tokenizer,
                               model=model_pretrained)
            trainer_lm = pl.Trainer(gpus=[int(hparams.gpu_offset)] if hparams.gpus == 1 else hparams.gpus,
                                    max_epochs=hparams.finetune_epochs,
                                    fast_dev_run=hparams.dev_run,
                                    progress_bar_refresh_rate=progress_bar_refresh_rate,
                                    val_check_interval=1 if hparams.dev_run else 5,
                                    checkpoint_callback=False)
            trainer_lm.fit(lm)
            if trial is not None and not hparams.dev_run:
                print(f'> Saving LM to file "{model_file}"')
                trainer_lm.save_checkpoint(model_file)

        # Phase 2: Train CounterfactualGAN
        print_phase(3, 'Training CounterfactualGAN', hparams.n_epochs)
        logger = MetricsCallback()
        cf_gan = CounterfactualGAN(hparams,
                                black_box,
                                dataset,
                                tokenizer,
                                lm)
        trainer_cf = pl.Trainer(gpus=[int(hparams.gpu_offset)] if hparams.gpus == 1 else hparams.gpus,
                                max_epochs=hparams.n_epochs,
                                fast_dev_run=hparams.dev_run,
                                progress_bar_refresh_rate=progress_bar_refresh_rate,
                                early_stop_callback=False,
                                callbacks=[logger],
                                checkpoint_callback=False)
        trainer_cf.fit(cf_gan)
        if trial is not None and not hparams.dev_run:
            model_file = model_path + f'CFGAN{id_cf}.ckpt'
            print(f'> Saving CounterfactualGAN to file "{model_file}"')
            trainer_cf.save_checkpoint(model_file)

        # Test its performance
        print_phase(4, 'Testing model')
        trainer_cf.test(cf_gan)

        if not hparams.dev_run:
            print(f'LM = LM{id_lm}.ckpt')
            print(f'CF = CFGAN{id_cf}.ckpt')
        if len(logger.metrics) == 0:  # aborted before end trial
            return 0.0
        res_fid = logger.metrics[-1]['test_fidelity']
        res_sim = logger.metrics[-1]['similarity'] / hparams.max_length
        res_rep = logger.metrics[-1]['repeats'] / hparams.max_length
        return (res_fid - res_sim * res_fid - res_rep
                if dataset.target_size == 1
                else res_fid + (1.0 - max(res_sim, 1.0)) + res_rep)

    if hparams.n_trials > 1:  # Hyperparameter optimization with optuna
        direction = 'minimize' if dataset.target_size == 1 else 'maximize'
        study = optuna.create_study(direction=direction,
                                    pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=hparams.n_trials)

        print(f'Number of finished trials: {len(study.trials)}')
        print('Best trial:')
        print(f'  Value: {study.best_trial.value}')
        print(f'  Params:')
        for k, v in study.best_trial.params.items():
            print(f'\t{k}={v}')
        if len(study.trials) > 1:
            print('Most important variables:')
            print(optuna.importance.get_param_importances(study))
    else:
        objective()

if __name__ == '__main__':
    main(get_config())
