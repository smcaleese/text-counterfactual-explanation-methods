"""White-box model for each dataset (SST, Hatespeech, SNLI)."""

import numpy as np
import afinn
import re
import pickle
import os

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, f1_score
from textstat import syllable_count
from scipy import sparse
from models.model import as_np
from config import SEED, VECTORS


class WhiteboxModel:
    def __init__(self,
                 assumed_dataset_name: str,
                 dataset,
                 sentiment_analyzer: callable,
                 attempt_load: bool = True,
                 **kwargs):
        """White-box prediction.

        Args:
            assumed_dataset_name: name to check dataset against
            dataset: dataset to train and test on
            sentiment_analyzer: sentiment analyzer function
            attempt_load: attempt to load pretrained vectors and model from disk
            **kwargs: optional arguments
        """
        self.dataset = dataset
        assert str(self.dataset).lower() == assumed_dataset_name, \
            f'This model is unable to function with dataset "{self.dataset}"'
        self.sentiment_analyzer = sentiment_analyzer
        self.attempt_load = attempt_load

    def __repr__(self):
        return 'whitebox'

    def __call__(self, X):
        """Predict a set of samples X."""
        return self.predict(X)

    def predict(self, X):
        """Predict a set of samples X."""
        raise NotImplementedError('Implemented in subclasses')

    def test(self, X_cols='X'):
        """Calculate performance measure on test set.

        Args:
            X_cols: name of columns that contain X values
        """
        print('|--> Testing model')
        tgt_size = self.dataset.target_size

        y_true = self.dataset.data['test']['y'].values
        y_pred = self.predict(self.dataset.data['test'][X_cols])
        if tgt_size == 1:
            return {'mse': mean_squared_error(y_true, y_pred)}
        y_pred = np.argmax(y_pred, axis=-1) if y_pred.ndim > 1 else y_pred
        return {'f1_score': f1_score(y_true, y_pred, average='binary' if tgt_size == 2 else 'macro')}


class SSTWhitebox(WhiteboxModel):
    def __init__(self, dataset, **kwargs):
        """White-box prediction using Afinn for SST dataset.

        Args:
            dataset: dataset containing train/dev/test data

        Raises:
            Exception: dataset does not correspond to model

        Example:
            Train the model and test its performance.

            >>> from models import SSTWhitebox
            >>> from dataset import SST
            >>> model = SSTWhitebox(SST())
            >>> model.test()
            {'f1_score': 0.6790986790986792}
        """
        super().__init__('sst', dataset, afinn.Afinn(), **kwargs)

    def predict(self, X):
        """Predict instance(s)."""
        ys = []
        for x in as_np(X):
            score = min(max(-10, self.sentiment_analyzer.score(x)), 10) / (2 * 10.)
            ys.append([.5 - score, .5 + score])
        return np.array(ys)


class HatespeechWhitebox(WhiteboxModel):
    def __init__(self, dataset, attempt_load=True):
        """White-box predictor by `t-davidson` for Hatespeech dataset.

        Args:
            dataset: dataset containing train/dev/test data
            attempt_load: whether to load pre-trained encoded
                data from disk (if exists)

        Raises:
            Exception: dataset does not correspond to model

        Example:
            Train the model and predict on the first three instances of the
            development set.

            >>> from models import HatespeechWhitebox
            >>> from dataset import Hatespeech
            >>> data = Hatespeech()
            >>> model = HatespeechWhitebox(data)
            >>> model(data.get('dev')['X'][:3])
            array([0.90115203, 0.81466604, 0.89845699])
        """
        super().__init__('hatespeech', dataset, SIA(), attempt_load)
        self.dataset.encode(self.encode, str(self), filetype='npy', attempt_load=self.attempt_load)
        self.attempt_load = True
        self.model = LinearRegression().fit(self.dataset.encoded_data[str(self)]['train'],
                                            self.dataset.data['train']['y'].values)

    # https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/classifier/classifier.py
    def get_features(self, x):
        """This function takes a string and returns a list of features.
        These include Sentiment scores, Text and Readability scores,
        as well as Twitter specific features.
        This is modified to only include those features in the final
        model."""

        # Sentiment (NLTK Vader)
        sentiment = self.sentiment_analyzer.polarity_scores(x)['compound']

        def preprocess(text_string):
            """Preprocess a twitter string (URLs, mentions, and fix whitespace)."""
            spaces = '\s+'
            url = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                   '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            mention = '@[\w\-]+'
            parsed_text = re.sub(spaces, ' ', text_string)
            parsed_text = re.sub(url, 'URLHERE', parsed_text)
            parsed_text = re.sub(mention, 'MENTION', parsed_text)
            return parsed_text

        words = preprocess(x)

        # Number of words, terms, syllables, characters (per word and total)
        n_words = len(words.split())
        syllables = syllable_count(words)
        n_chars = sum(len(w) for w in words)  # num chars in words
        n_chars_total = len(x)
        n_terms = len(x.split())

        # Average number of syllables, number of unique terms
        avg_syl = round(float((syllables + 0.001)) / float(n_words + 0.001), 4)
        n_unique_terms = len(set(words.split()))

        # Modified Flesch-Kinkaide (FK) grade w/ average sentence length ommitted
        FKRA = round(0.39 * n_words + (11.8 * avg_syl) - 15.59, 1)

        # Modified Flesch Reading Ease (FRE) score, where sentence fixed to 1
        FRE = round(206.835 - (1.015 * n_words) - (84.6 * avg_syl), 2)

        # Count number of URLs, @mentions and #hashtags
        hashtag_pattern = '#[\w\-]+'
        twitter_objs = (x.count('URLHERE'),
                        x.count('MENTION'),
                        len(re.findall(hashtag_pattern, x)))

        return [FKRA, FRE, syllables, n_chars, n_chars_total, n_terms, n_words,
                n_unique_terms, sentiment, *twitter_objs]

    def encode(self, X):
        """Pre-encode dataset part."""
        return np.array([self.get_features(x) for x in as_np(X)])

    def predict(self, X):
        """Predict instance(s)."""
        return self.model.predict(self.encode(X))


class SNLIWhitebox(WhiteboxModel):
    def __init__(self, dataset, attempt_load=True, seed=SEED):
        """White-box prediction with TF-IDF and Logistic Regression for SNLI.

        Args:
            dataset: dataset containing train/dev/test data
            attempt_load: whether to load pre-trained encoded
                data from disk (if exists)
            seed: seed for reproducibility

        Raises:
            Exception: dataset does not correspond to model

        Example:
            Train the model and predict on the first three instances of
            the training set.

            >>> from models import SNLIWhitebox
            >>> from dataset import SNLI
            >>> data = SNLI()
            >>> model = SNLIWhitebox(data, seed=1994, attempt_load=False)
            >>> model(data.get('test')[['X_premise', 'X_hypothesis]][:3])
            array([[0.28439253, 0.27085086, 0.4447566 ],
                   [0.13814729, 0.62595744, 0.23589527],
                   [0.23283758, 0.40971037, 0.35745205]])
        """
        super().__init__('snli', dataset, None, attempt_load)
        self.encoder = self.__get_encoder()  
        self.dataset.encode(self.encode, str(self), filetype='npz', attempt_load=self.attempt_load)
        self.model = LogisticRegression(random_state=seed).fit(self.dataset.encoded_data[str(self)]['train'],
                                                               self.dataset.data['train']['y'].values)

    def __get_encoder(self):
        X = self.dataset.data['train'][['X_premise', 'X_hypothesis']]
        tfidf_file = VECTORS + 'tfidf_snli.pkl'

        X = as_np(X)
        if not os.path.isfile(tfidf_file) or not self.attempt_load:
            X_ = np.array([' '.join(x) for x in X.astype(str)])
            encoder = TfidfVectorizer(ngram_range=(1, 2)).fit(X_)
            with open(tfidf_file, 'wb') as f:
                pickle.dump(encoder, f)
        else:
            print(f'|--> Loading model checkpoint "{tfidf_file}"')
            with open(tfidf_file, 'rb') as f:
                encoder = pickle.load(f)
        return encoder

    def encode(self, X):
        """Pre-encode dataset part."""
        X = as_np(X)
        return sparse.hstack((self.encoder.transform(X[:, 0]),
                              self.encoder.transform(X[:, 1])))

    def predict(self, X):
        """Predict instance(s)."""
        return self.model.predict_proba(self.encode(X))

    def test(self):
        """Evaluate performance on the test set."""
        return super().test(['X_premise', 'X_hypothesis'])
