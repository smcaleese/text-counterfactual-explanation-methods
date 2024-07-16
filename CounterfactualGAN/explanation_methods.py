"""CounterfactualGAN (c), 2020-2021 Marcel Robeer."""

import re
import nltk
import numpy as np
import torch
import sklearn.metrics as metrics
import itertools
import tensorflow_hub as hub
import tensorflow as tf
import math
from tqdm import tqdm
from time import time as current_time
from nltk.metrics.distance import edit_distance
from nltk.tokenize import word_tokenize
from transformers import BertForMaskedLM, BertTokenizer
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize

from config import SEED, BERT, UNIVERSAL_SENTENCE_ENCODER, COUNTER_FITTED_EMBEDDINGS, COS_SIM
from utils import flatten, np_cache, get_pos, pos_filter, STOPWORDS
from models.external.use import USE


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

DETOKENIZER = nltk.tokenize.treebank.TreebankWordDetokenizer()

def generic_detokenizer(x):
    x = str(DETOKENIZER.detokenize(x)
            .replace(' & amp;', ' &amp;')
            .replace(' & nbsp;', ' &nbsp;')
            .replace(' @ ', ' @')
            .replace('@ user', '@user')
            .replace(' # ', ' #')
            .replace('do n\' t', 'don\'t')
            .replace('ca n\' t', 'can\'t')
            .replace('/ /', '//')
            .replace('http: /', 'http:/')
            .replace('https: /', 'https:/')
            .replace(' t . co / ', 't.co/')
            .replace('& # ', '&#'))
    return str(re.sub(r'\&(\s)?\#(\s)?([\d]+)(\s)?;', r'&#\3;', x))


class USEWrapper:
    def __init__(self, file=UNIVERSAL_SENTENCE_ENCODER):
        self.encoder = hub.Module(file)
        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def __call__(self, sentences):
        if ((isinstance(sentences, np.ndarray)
             and sentences.ndim == 2 and sentences.shape[1] == 2) or
            (isinstance(sentences, list) and len(sentences) > 0
             and isinstance(sentences[0], list))):
            sentences = np.array([' '.join(s) for s in sentences])
        

        return self.sess.run(self.encoder(sentences))


USE_FOR_SEMANTIC_SYM = USEWrapper()


class ExplanationMethod:
    def __init__(self, tokenizer, target_size, detokenizer=None, use_semantic=None, provide_true_labels=False):
        """Explanation methods all use the same evaluation.

        Args:
            tokenizer: tokenizer to split strings into tokens
            target_size: size of target (1 for regression or number of classes)
            detokenizer: detokenizer to convert lists of tokens to strings
            use_semantic: function to call Universal Sentence Encoder (USE)
            provide_true_labels: whether __call__ can take true labels
        """
        self.tokenizer = tokenizer
        if detokenizer is None:
            detokenizer = generic_detokenizer
        self.detokenizer = detokenizer
        self.target_size = target_size
        if use_semantic is None:
            use_semantic = USE_FOR_SEMANTIC_SYM
        self.use_semantic = use_semantic
        self.provide_true_labels = provide_true_labels

    def __call__(self, X, predict_fn, y=None, return_y=False):
        raise Exception('Implemented in subclasses.')
        
    def similarity(self, l1, l2):
        """Word edit distance similarity between pairs of strings."""
        l1 = [l1] if isinstance(l1, str) else l1
        l2 = [l2] if isinstance(l2, str) else l2
        assert len(l1) == len(l2), 'inputs must be equal length'

        sims = np.array([edit_distance(word_tokenize(str(s1)), word_tokenize(str(s2)))
                         for s1, s2 in zip(l1, l2)])
        if len(sims) == 1:
            return sims[0], sims
        return np.nanmean(sims), sims

    def semantic_similarity(self, l1, l2, l1_encoded=False, l2_encoded=False):
        """Semantic similarity (according to USE + cosine) between pairs of strings."""
        if not l1_encoded:
            l1 = [l1] if isinstance(l1, str) else l1
            if isinstance(l1, np.ndarray) and l1.ndim > 1 or (len(l1) > 0 and isinstance(l1[0], np.ndarray)):
                l1 = [' '.join(l) for l in l1]
            l1 = self.use_semantic(l1)
        if not l2_encoded:
            l2 = [l2] if isinstance(l2, str) else l2
            if isinstance(l2, np.ndarray) and l2.ndim > 1 or (len(l2) > 0 and isinstance(l2[0], np.ndarray)):
                l2 = [' '.join(l) for l in l2]
            l2 = self.use_semantic(l2)

        def cos_dist(a, b):
            return 1.0 - np.arccos(np.clip(np.inner(a, b), -1.0, 1.0) / (np.linalg.norm(a) * np.linalg.norm(b))) / math.pi
        cos_dists = np.array([cos_dist(l1[i], l2[i]) for i in range(len(l1))])
        
        if len(cos_dists) == 1:
            return cos_dists[0], cos_dists
        return np.nanmean(cos_dists), cos_dists

    def fidelity(self, y_target, y_cf):
        """Calculate fidelity of the method."""
        y_target, y_cf = np.array(y_target), np.array(y_cf)

        if self.target_size == 1:
            metric = metrics.mean_squared_error
        else:
            if y_target.ndim > 1:
                y_target = np.argmax(y_target, axis=-1)
            if y_cf.ndim > 1:
                y_cf = np.argmax(y_cf, axis=-1)
            average = 'binary' if self.target_size == 2 else 'macro'
            metric = lambda x, y: metrics.f1_score(x, y, average=average)
        return metric(y_target, y_cf)

    def evaluate(self, X, X_, y, y_, training_time, inference_time):
        sim_avg, sim = self.similarity(X, X_)
        sem_avg, sem = self.semantic_similarity(X, X_)
        return {'similarity': sim_avg,
                'X_sim': sim,
                'semantic': sem_avg,
                'X_sem': sem.reshape(-1),
                'performance_measure': 'MSE' if self.target_size == 1 else 'F1-score',
                'fidelity': self.fidelity(y, y_),
                'training_time': training_time,
                'inference_time': inference_time}


class BaselineMethod(ExplanationMethod):
    def __init__(self, tokenizer=None, target_size=1, seed=SEED, **kwargs):
        """Generic implementation of shared functions for baseline methods.

        Args:
            tokenizer: tokenizer to split strings into tokens
            target_size: size of target (1 for regression or number of classes)
        """
        if tokenizer is None:
            tokenizer = nltk.tokenize.word_tokenize
        super().__init__(tokenizer, target_size, **kwargs)
        self.seed = seed

    def distance_fn(self, x, y):
        """Distance function depending on target_size between two vectors x and y."""
        if self.target_size == 1:  # regression analysis
            if x.ndim > 1:
                x = x.squeeze()
            if y.ndim > 1:
                y = y.squeeze()
            dist = (x - y) ** 2
        else:  # classification
            if x.ndim == 1:
                x = np.eye(self.target_size)[x]
            if y.ndim == 1:
                y = np.eye(self.target_size)[y]
            dist = metrics.pairwise.paired_distances(x, y, metric='euclidean')
        return normalize(dist.reshape(-1, 1), norm='max', axis=0).squeeze()

    def __repr__(self):
        seed = f'seed={self.seed}' if self.seed is not None else ''
        return f'{self.__class__.__name__}({seed})'

    def permute(self, X):
        """Create potential counterfactuals by applying perturbations."""
        return X

    def _with_gpu(self, predict_fn, X):
        """Handle (optional) GPU run of predict function."""
        if hasattr(predict_fn, 'eval'):
            predict_fn = predict_fn.eval()
        gpu_enabled_device = torch.cuda.is_available() and hasattr(predict_fn, 'device')
        if gpu_enabled_device:
            if hasattr(self, 'model'):
                self.model = self.model.to('cpu')
            predict_fn = predict_fn.to('cuda')
        y = predict_fn(X)
        if 'numpy' not in str(type(y)):
            if 'torch' in str(type(y)):
                if y.device.type != 'cpu':
                    y = y.cpu()
                y = y.detach()
            y = np.array(y)
        if gpu_enabled_device and hasattr(self, 'model'):
            self.model = self.model.to('cuda')
        return y

    def __call__(self, X, predict_fn, y, return_y=False, verbose=False):
        """Generate neighborhood data, and select counterfactuals from these.

        Args:
            X: instances
            predict_fn: prediction function to obtain y
            y: target values for CF(x)
            return_y: also return original y values and y values
                for the counterfactuals
        """
        assert len(X) == len(y), 'X and y should have the same number of elements'
        if 'pandas' in str(type(X)):
            X = X.values

        # Optionally, split and keep the second the same
        X, X_other = (X[:, 0], X[:, 1:]) if X.ndim > 1 else (X, None)

        # Create permutations and serialize
        __time_start = current_time()
        if hasattr(self.permute, 'cache_info'):
            hits = self.permute.cache_info().hits
        X_ = [x for x in tqdm(self.permute(X), desc='Permuting instances', total=len(X), disable=not verbose)]
        if hasattr(self.permute, 'cache_info') and self.permute.cache_info().hits > hits:  # cache was used, add time
            __time_start -= self.cache_time
        X_ind = np.array(flatten([[i] * len(x) for i, x in enumerate(X_)]))
        X_ = np.array(flatten(X_))

        # Apply predictions
        if X_other is not None:
            X_ = np.stack((X_.reshape(-1, 1), X_other[X_ind]), axis=1).reshape(X_.shape[0], -1)
        y_ = self._with_gpu(predict_fn, X_)

        # Select each corresponding to target
        y_target = np.array(y)[np.array(X_ind)]
        candidates = self.distance_fn(y_target, y_)

        res_x = np.empty_like(X)
        res_y = np.empty((y.shape[0], y_.shape[1])) if self.target_size > 1 and y_.ndim > 1 else np.empty_like(y) 
        for ind in tqdm(np.unique(X_ind), desc='Selecting instances', disable=not verbose):
            selected = self.select(candidates, X_ind, ind)
            res_x[ind] = X_[X_ind == ind][selected]
            res_y[ind] = y_[X_ind == ind][selected]

        time = max(current_time() - __time_start, 0.)
        if self.target_size > 1:
            res_y = np.stack(res_y, axis=-1).T
        res_eval = self.evaluate(X, res_x, y, res_y, 0., time)

        if return_y:
            return res_eval, res_x, res_y
        return res_eval, res_x

    def select(self, candidates, X_ind, ind):
        c = candidates[X_ind == ind]
        if len(c) <= 5:
            return np.argmin(c)
        return np.argpartition(c, min(5, len(c)))[0]


class SEDC(BaselineMethod):
    def __init__(self, tokenizer=None, target_size=1, max_combs=20, max_diff=5, **kwargs):
        """Find counterfactuals by dropping tokens.

        Args:
            tokenizer: tokenizer to split strings into tokens
            target_size: size of target (1 for regression or number of classes)
            max_combs: maximum number of combinations to sample
            max_diff: maximum number of dropout tokens
        """
        super().__init__(tokenizer, target_size, **kwargs)
        self.max_combs = max_combs
        self.max_diff = max_diff

    def permute(self, X):
        """Created permuted versions of X by dropping tokens up to length max_diff."""
        np.random.seed(self.seed)

        for x in X:
            x_ = self.tokenizer(x)
            md = min(self.max_diff, len(x_))
            mc = min(self.max_combs // md, len(x_))
            all_combs = []
            for i in range(1, 1 + md):
                all_combs += np.random.binomial(1, 1. - i / len(x_), (mc, len(x_))).tolist()
            yield [self.detokenizer([x_[c_] for c_ in range(len(c)) if c[c_] == 1]) for c in all_combs]


class PWWSAntonym(BaselineMethod):
    def __init__(self, tokenizer=None, target_size=1, max_combs=20, max_diff=5, **kwargs):
        """Find counterfactuals by searching WordNet antonyms, and using LFO as a fall-back.

        Args:
            tokenizer: tokenizer to split strings into tokens
            target_size: size of target (1 for regression or number of classes)
            max_combs: maximum number of combinations to sample
            max_diff: maximum number of dropout tokens
        """
        super().__init__(tokenizer, target_size, **kwargs)
        self.max_combs = max_combs
        self.max_diff = max_diff

    def get_antonyms(self, word):
        """Get antonyms for word from WordNet."""
        for syn in nltk.corpus.wordnet.synsets(word): 
            for lemma in syn.lemmas(): 
                if lemma.name() != syn.name():
                    yield lemma.name()
                if lemma.antonyms(): 
                    yield lemma.antonyms()[0].name()

    def permute(self, X):
        """Created permuted versions of X by searching for antonyms, based on PWWS
        Algorithm 1, except adding line (3) with synonyms + antonyms and (14) with
        targeted classification/regression/NLI."""
        np.random.seed(self.seed)

        for x in X:
            x_ = self.tokenizer(x)
            all_antonyms = {}
            for i, w in enumerate(x_):
                antonyms = [a for a in self.get_antonyms(w)]
                if len(antonyms) > 0:
                    all_antonyms[i] = antonyms
            md = min(self.max_diff, len(x_))
            mc = min(self.max_combs // md, len(x_))
            all_combs = []
            for i in range(1, 1 + md):
                all_combs += np.random.choice(len(x_), (mc, i)).tolist()
            res = []
            for c in all_combs:
                r_ = []
                for i in range(len(x_)):
                    if i in c:
                        if i in all_antonyms.keys():
                            r_.append(np.random.choice(all_antonyms[i]))
                    else:
                        r_.append(x_[i])
                res.append(self.detokenizer(r_))
            yield res


class eBERT(BaselineMethod):
    def __init__(self, bert_model=BERT, target_size=1, batch_size=48, **kwargs):
        """Explain by finding replacement words using BERT.

        Args:
            bert_model: name of BERT model to use
            target_size: size of target (1 for regression or number of classes)
            batch_size: size of batches when predicting [MASK] tokens
        """
        self.batch_size = batch_size
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.model = BertForMaskedLM.from_pretrained(bert_model)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        self.model.eval()
        super().__init__(tokenizer, target_size, **kwargs)

    def encode(self, X):
        """Transform strings into tokenized sequences of equal length."""
        return self.tokenizer.batch_encode_plus(
            X,
            return_tensors='pt',
            pad_to_max_length=True
        )['input_ids']

    def decode(self, X):
        """Transform tokenized sequences into single strings."""
        def dec(x):
            return self.tokenizer.decode(x).replace('[CLS]', '') \
                                           .replace('[SEP]', '') \
                                           .replace('[PAD]', '') \
                                           .replace('  ', ' ') \
                                           .strip() 
        return np.array([dec(x) for x in X])

    @np_cache(maxsize=3)
    def permute(self, X):
        """Created permuted versions of X by predicting replacements for [MASK] tokens."""
        tok = self.tokenizer
        mask = tok.mask_token_id
        pad = tok.pad_token_id
        start_time = current_time()

        # Encode
        X_enc = self.encode(X)
        repeats = torch.tensor([[x_ not in tok.all_special_ids for x_ in x] for x in X_enc]).sum(dim=-1)

        xs = []
        x_origs = []

        # Construct masks and trace replaced tokens
        for i in range(len(X_enc)):
            x_orig = []
            x_ = X_enc[i].repeat(repeats[i], 1)
            for j in range(0, repeats[i]):
                x_orig.append(x_[j][j + 1].clone())
                x_[j][j + 1] = mask
            xs.append(x_)
            x_origs.append(torch.tensor(x_orig))

        xs, x_origs = torch.cat(xs), torch.cat(x_origs)

        def predict_fn(batch):
            with torch.no_grad():
                return torch.topk(self.model(batch.to(self.model.device) if torch.cuda.is_available
                                                                        else batch)[0],
                                2, dim=-1)[-1]

        top = torch.cat([predict_fn(batch) for batch in tqdm(DataLoader(xs, batch_size=self.batch_size),
                                                             desc='Predicting instances')])
                                                             
        if torch.cuda.is_available():
            top = top.to('cpu')
        masked_positions = top.T[(xs == mask).repeat(2, 1).reshape(2, xs.shape[1], -1)].reshape(-1, 2)

        # Pick the most probable instance that is not the original, assign it to [MASK]
        has_changed = (masked_positions.T != x_origs).T
        xs_pad = xs.clone()
        xs_pad[xs == mask] = pad
        xs[xs == mask] = torch.tensor([pos[has_changed[i]][0] if len(pos[has_changed[i]]) > 0
                                                              else pad
                                       for i, pos in enumerate(masked_positions)])

        # Decode, group and return
        decoded = self.decode(torch.stack((xs, xs_pad), dim=1).view(xs.size(0) * 2, xs.size(1)))
        X_ind = np.array(flatten([[i] * 2 * x for i, x in enumerate(repeats)]))

        self.cache_time = current_time() - start_time

        return [decoded[X_ind == ind].copy() for ind in np.unique(X_ind)]

    def select(self, candidates, X_ind, ind):
        probs = candidates[X_ind == ind]
        if probs.sum() == 0:
            probs = 1. - probs
        return np.random.choice(np.arange(probs.size), p=probs / probs.sum())
    

class TextFooler(BaselineMethod):
    def __init__(self,
                 target_size=1,
                 stop_words=STOPWORDS,
                 embeddings=COUNTER_FITTED_EMBEDDINGS,
                 cos_sim=COS_SIM,
                 universal_sentence_encoder=USE(),
                 verbose=True,
                 **kwargs):
        super().__init__(target_size=target_size, provide_true_labels=True, seed=None, **kwargs)
        self.verbose = verbose
        self.stop_words = STOPWORDS
        self.word2idx, self.idx2word = self.__build_vocab(embeddings)
        self.sim_mat = self.__cos_sim_matrix(cos_sim, embeddings)
        self.use = universal_sentence_encoder
        self.seed = None

    def __build_vocab(self, emb):
        """https://github.com/jind11/TextFooler/blob/master/attack_classification.py"""
        word2idx = {}
        idx2word = {}
        with open(emb, 'r') as ifile:
            for line in tqdm(ifile, 'Creating vocabulary', disable=not self.verbose):
                word = line.split()[0]
                if word not in idx2word:
                    idx2word[len(idx2word)] = word
                    word2idx[word] = len(idx2word) - 1
        return word2idx, idx2word
    
    def __cos_sim_matrix(self, cos_sim, embedding_path):
        try:
            product = np.load(cos_sim)
        except FileNotFoundError:
            """https://github.com/jind11/TextFooler/blob/master/comp_cos_sim_mat.py"""
            embeddings = []
            with open(embedding_path, 'r') as ifile:
                for line in tqdm(ifile, 'Creating cosine similarity matrix', disable=not self.verbose):
                    embedding = [float(num) for num in line.strip().split()[1:]]
                    embeddings.append(embedding)
            embeddings = np.array(embeddings)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.asarray(embeddings / norm, 'float32')
            product = np.dot(embeddings, embeddings.T)
            np.save((cos_sim), product)
        return product
    
    def pick_most_similar_words_batch(self, src_words, ret_count=10, threshold=0.0):
        """https://github.com/jind11/TextFooler/blob/master/attack_nli.py"""
        sim_order = np.argsort(-self.sim_mat[src_words, :])[:, 1:1 + ret_count]
        sim_words, sim_values = [], []
        for idx, src_word in enumerate(src_words):
            sim_value = self.sim_mat[src_word][sim_order[idx]]
            mask = sim_value >= threshold
            sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
            sim_word = [self.idx2word[id] for id in sim_word]
            sim_words.append(sim_word)
            sim_values.append(sim_value)
        return sim_words, sim_values

    def not_approx_equal(self, y, y_):
        y = y.numpy() if 'torch' in str(type(y)) else y
        y = y_.numpy() if 'torch' in str(type(y_)) else y_
        if self.target_size > 1:
            y = np.argmax(y, axis=-1) if isinstance(y, np.ndarray) and y.ndim > 1 else y
            y = np.int(y) if isinstance(y, int) else y
            y_ = np.argmax(y_, axis=-1) if isinstance(y_, np.ndarray) and y_.ndim > 1 else y_
            y_ = np.int(y_) if isinstance(y_, int) else y_
            return y != y_
        return np.abs(y - y_) > 0.2

    def attack_one(self, text, predict_fn, y, y_true, sim_score_threshold=0.7, sim_score_window=15,
                   synonym_num=50, import_score_threshold=-1.0, text_other=None):
        """https://github.com/jind11/TextFooler/blob/master/attack_classification.py and
        https://github.com/jind11/TextFooler/blob/master/attack_nli.py"""
        if y_true is not None and self.not_approx_equal(y, y_true):
            return ''
        if text_other is not None:
            if isinstance(text_other, np.ndarray):
                text_other = text_other.tolist()
            elif isinstance(text_other, str):
                text_other = [text_other]

        orig_y = y_true if y_true is not None else y_pred
        orig_prob = np.array([1.0] * self.target_size) if isinstance(orig_y, (int, np.int, np.int64)) else orig_y
        len_text = len(text)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_ssw = (sim_score_window - 1) // 2
        
        # get pos
        pos_ls = get_pos(text)
        
        # get importance score
        leave_1_texts = [self.detokenize(text[:ii] + ['<oov>'] + text[min(ii + 1, len_text):]) for ii in range(len_text)]
        if text_other is not None:
            leave_1_pred = self._with_gpu(predict_fn, [[t] + text_other for t in leave_1_texts])
        else:
            leave_1_pred = self._with_gpu(predict_fn, leave_1_texts)
        correct = leave_1_pred[:, orig_y.argmax(axis=-1)] if self.target_size > 1 else leave_1_pred
        maxim = leave_1_pred.max(axis=-1) if self.target_size > 1 else leave_1_pred
        select = orig_prob[leave_1_pred.argmax(axis=-1)] if self.target_size > 1 else np.array([orig_prob] * len(leave_1_pred))
        import_scores = y.max() - correct + self.not_approx_equal(leave_1_pred, orig_y).astype(int) * (maxim - select)

        # get words to perturb ranked by importance score for word in words_perturb
        words_perturb = []
        for idx in np.argsort(import_scores)[::-1]:
            try:
                if import_scores[idx] > import_score_threshold and text[idx] not in self.stop_words:
                    words_perturb.append((idx, text[idx]))
            except:
                pass

        # find synonyms
        words_perturb_idx = [self.word2idx[word] for idx, word in words_perturb if word in self.word2idx]
        synonym_words, _ = self.pick_most_similar_words_batch(words_perturb_idx, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in self.word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text[:]
        text_cache = text_prime[:]
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            if text_other is not None:
                new_probs = self._with_gpu(predict_fn, [[t] + text_other for t in self.detokenize(new_texts)])
            else:
                new_probs = self._with_gpu(predict_fn, self.detokenize(new_texts))

            # compute semantic similarity
            if idx >= half_ssw and len_text - idx - 1 >= half_ssw:
                txt_min, txt_max = idx - half_ssw, idx + half_ssw + 1
            elif idx < half_ssw and len_text - idx - 1 >= half_ssw:
                txt_min, txt_max = 0, sim_score_window
            elif idx >= half_ssw and len_text - idx - 1 < half_ssw:
                txt_min, txt_max = len_text - sim_score_window, len_text
            else:
                txt_min, txt_max = 0, len_text
            semantic_sims = self.use([self.detokenize(text_cache[txt_min:txt_max])] * len(new_texts),
                                       list(map(lambda x: self.detokenize(x[txt_min:txt_max]), new_texts)))[0]

            new_probs_mask = self.not_approx_equal(orig_y, new_probs)
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                break
            else:
                n = new_probs[:, np.argmax(orig_y)] if self.target_size > 1 else new_probs
                new_label_probs = n + (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)
                p_ = orig_prob[orig_y] if self.target_size > 1 else orig_prob
                if np.min(new_label_probs, axis=-1) < p_:
                    text_prime[idx] = synonyms[np.argmin(new_label_probs, axis=-1)]
            text_cache = text_prime[:]
        return self.detokenize(text_prime)
    
    def tokenize(self, X):
        """Tokenize instance(s)."""
        if type(X) is str:
            return self.tokenizer(X)
        return [self.tokenizer(x) for x in X]

    def detokenize(self, X):
        """Detokenize instance(s)."""
        detokenize_single = lambda x: ' '.join(x).replace(' ##', '')
        if isinstance(X, (list, np.ndarray)) and len(X) > 0 and isinstance(X[0], str):
            return detokenize_single(X)
        return np.array([detokenize_single(x) for x in X])

    def __call__(self, X, predict_fn, y, y_true=None, return_y=False,
                 sim_score_threshold=0.7, sim_score_window=15,
                 synonym_num=50, import_score_threshold=-1.0):
        assert len(X) == len(y), 'X and y should have the same number of elements'
        if 'pandas' in str(type(X)):
            X = X.values
        if y_true is not None and 'pandas' in str(type(y_true)):
            y_true = y_true.values

        __time_start = current_time()

        # Speed up SNLI, otherwise it takes hours
        if X.ndim > 1: #978
            si = [i for i in range(978)] + [1289, 2039, 2483, 2584, 2886, 4311, 4323,
                                            4792, 5507, 6168, 6206, 6390, 6513, 6689,
                                            6952, 7120, 7464, 7836, 8050, 8613, 8699, 9187]
            X = X[si]
            y = y[si]
 
        y_pred = self._with_gpu(predict_fn, X)
        X, X_other = (X[:, 0], X[:, 1:]) if X.ndim > 1 else (X, None)
        
        # Perform attack
        X_ = [self.attack_one(self.tokenize(X[i]),
                              predict_fn,
                              y_pred[i],
                              y_true[i] if y_true is not None else None,
                              sim_score_threshold=sim_score_threshold,
                              sim_score_window=sim_score_window,
                              synonym_num=synonym_num,
                              import_score_threshold=import_score_threshold,
                              text_other=X_other[i] if X_other is not None else None)
              for i in tqdm(range(len(X)), desc='Attacking instances', total=len(X), disable=not self.verbose)]
        
        time = max(current_time() - __time_start, 0.)

        # Apply predictions
        if X_other is not None:
            X_ = np.array(X_)
            X_ = np.stack((X_.reshape(-1, 1), X_other), axis=1).reshape(X_.shape[0], -1)
        y_ = self._with_gpu(predict_fn, X_)

        # Evaluate and return
        res_eval = self.evaluate(X, X_, y, y_, 0., time)  # TO-DO add load time?

        if return_y:
            return res_eval, X_, y_
        return res_eval, X_
