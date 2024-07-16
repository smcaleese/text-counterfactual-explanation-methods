"""CounterfactualGAN (c), 2020-2021 Marcel Robeer.

This module contains utility functions used by other modules.
"""

import numpy as np
import torch.nn as nn
import nltk

from transformers.activations import gelu_new
from functools import lru_cache, wraps

    
try:
    nltk.data.find('taggers/universal_tagset')
except LookupError:
    nltk.download('universal_tagset')


def split_hparam(s):
    """Split a comma-seperated hyperparameter string into a list."""
    return s.replace(' ', '').split(',')


def flatten(li):
    """Flatten list of lists."""
    return [i for subli in li for i in subli]


def np_cache(*args, **kwargs):
    """lru_cache decorater able to handle np.arrays."""
    def decorator(function):
        def np_to_hashable(a, level=0):
            if 'numpy' in str(type(a)) and hasattr(a, '__iter__'):
                v = tuple(np_to_hashable(a_) for a_ in a)
                return ('<np', v, 'np>') if level == 0 else v
            return a

        def hashable_to_np(a):
            if isinstance(a, tuple) and len(a) == 3 and a[0] == '<np' and a[-1] == 'np>':
                return np.array(a[1])
            return a

        @wraps(function)
        def wrapper(*args, **kwargs):
            args = [np_to_hashable(a) for a in args]
            kwargs = {k: np_to_hashable(v) for k, v in kwargs.items()}
            return cached_wrapper(*args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(*args, **kwargs):
            args = [hashable_to_np(a) for a in args]
            kwargs = {k: hashable_to_np(v) for k, v in kwargs.items()}
            return function(*args, **kwargs)

        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper
    return decorator


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return gelu_new(X)

    
def get_pos(sent, tagset='universal'):
    """https://github.com/jind11/TextFooler/blob/master/criteria.py"""
    if tagset == 'default':
        word_n_pos_list = nltk.pos_tag(sent)
    elif tagset == 'universal':
        word_n_pos_list = nltk.pos_tag(sent, tagset=tagset)
    _, pos_list = zip(*word_n_pos_list)
    return pos_list


def pos_filter(ori_pos, new_pos_list):
    """https://github.com/jind11/TextFooler/blob/master/criteria.py"""
    return [ori_pos == new_pos or (set([ori_pos, new_pos]) <= set(['NOUN', 'VERB']))
            for new_pos in new_pos_list]

    
STOPWORDS = set(['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both',  'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used',  've', 'was', 'wasn', "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])