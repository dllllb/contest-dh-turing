import re
from nltk.corpus import stopwords
from nltk.collocations import TrigramCollocationFinder, TrigramAssocMeasures
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.util import ngrams
import numpy as np
import pandas as pd
from functools import partial, update_wrapper

def tokenize(text, stopwords):
    clean_text = re.sub('[^a-zA-Z\']', ' ', text)
    sentence = clean_text.lower().split()
    sentence = [w for w in sentence if w not in stopwords]
    sentence = [w for w in sentence if len(w) > 3]
    return sentence

def dialogs2sentences(dialogs, stopwords):
    sentences = []
    for d in dialogs:
        for post in d['thread']:
            sentence = tokenize(post['text'], stopwords)
            sentences.append(sentence)
    return sentences

def dialog_ngrams_count(d, stopwords, freq_ngrams, ngram_len):
    total_count_a = 0
    total_count_b = 0
    for post in d['thread']:
        sentence = tokenize(post['text'], stopwords)
        sent_ngrams = ngrams(sentence, ngram_len)
        count = sum([1 for e in sent_ngrams if e in freq_ngrams])
        if post['userId'] == 'user1':
            total_count_a += count
        else:
            total_count_b += count
    return total_count_a, total_count_b

def find_bigrams(sentences, n_ngrams):
    cf = BigramCollocationFinder.from_documents(sentences)
    fng = cf.nbest(BigramAssocMeasures.likelihood_ratio, n_ngrams)
    return fng

def find_trigrams(sentences, n_ngrams):
    cf = TrigramCollocationFinder.from_documents(sentences)
    fng = cf.nbest(TrigramAssocMeasures.likelihood_ratio, n_ngrams)
    return fng

class FreqNgrams(BaseEstimator, TransformerMixin):
    def __init__(self, n_ngrams=30, ngram_len=2):
        self.n_ngrams = n_ngrams
        self.stopwords = set(stopwords.words('english'))
        self.ngram_fidner = find_bigrams if ngram_len == 2 else find_trigrams
        self.ngram_len = ngram_len

    def fit(self, dialogs, y=None):
        sentences = dialogs2sentences(dialogs.dialog, self.stopwords)
        self.fng = set(self.ngram_fidner(sentences, self.n_ngrams))
        return self

    def transform(self, dialogs):
        counts = [
            dialog_ngrams_count(d, self.stopwords, self.fng, self.ngram_len)
            for d in dialogs.dialog
        ]
        c_arr = np.array(counts)
        columns=['u1_ngram_{}'.format(self.ngram_len), 'u2_ngram_{}'.format(self.ngram_len)]
        return pd.DataFrame(c_arr, columns=columns, index=dialogs.index)
