import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

def simple_extractors(dialogs_df, extractors):
    features = []
    for ex in extractors:
        feat = dialogs_df['dialog'].apply(ex).apply(pd.Series, 1)
        # naming with consciousness
        feat_len = int(feat.shape[1] / 2)
        if feat_len > 1:
            feat.columns = ['{ex}_{side}_{i}'.format(ex=ex.__name__, side=s, i=i)
                            for s in ['self', 'that'] for i in range(1, feat_len + 1)]
        else:
            feat.columns = ['{ex}_{side}'.format(ex=ex.__name__, side=s)
                            for s in ['self', 'that']]
        features.append(feat)

    features_df = pd.concat(features, axis=1)

    return features_df

def simple_transfromer(extractors):
    return FunctionTransformer(partial(simple_extractors, extractors=extractors), validate=False)

def get_texts(d, side):
    alice, bob = [], []
    for post in d['thread']:
        text = post['text']
        if post['userId'] == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    if side =='fit':
        return ' '.join(alice)+ ' '.join(bob)
    elif side=='user1':
        return ' '.join(alice)
    elif side=='user2':
        return ' '.join(bob)


class PCACharFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, side):
        self.side = side
        self.pipe = Pipeline(steps=[
            ('cv', CountVectorizer(analyzer='char_wb', ngram_range=(1, 2), binary=True, dtype=np.uint8)),
            ('pca', TruncatedSVD(20))])

    def fit(self, dialogs_df, y=None):
        texts = dialogs_df['dialog'].apply(get_texts, side='fit')
        self.pipe = self.pipe.fit(texts)
        return self

    def transform(self, dialogs_df):
        texts = dialogs_df['dialog'].apply(get_texts, side=self.side)
        feats = self.pipe.transform(texts)
        feats_df = pd.DataFrame(feats, columns=['pca_char_{side}_{i}'.format(side=self.side, i=i)
                                                for i in range(1, feats.shape[1]+1)],
                                index=dialogs_df.index)
        return feats_df

class PCACharFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, side):
        self.side = side
        self.pipe = Pipeline(steps=[
            ('cv', CountVectorizer(analyzer='char_wb', ngram_range=(2, 5), binary=True, dtype=np.uint8)),
            ('pca', TruncatedSVD(20))])

    def fit(self, dialogs_df, y=None):
        texts = dialogs_df['dialog'].apply(get_texts, side='fit')
        self.pipe = self.pipe.fit(texts)
        return self

    def transform(self, dialogs_df):
        texts = dialogs_df['dialog'].apply(get_texts, side=self.side)
        feats = self.pipe.transform(texts)
        feats_df = pd.DataFrame(feats, columns=['char_pca_{side}_{i}'.format(side=self.side, i=i)
                                                for i in range(1, feats.shape[1]+1)],
                                index=dialogs_df.index)
        return feats_df

class PCAWordFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, side):
        self.side = side
        self.pipe = Pipeline(steps=[
            ('cv', CountVectorizer(ngram_range=(1, 2), binary=True, dtype=np.uint8)),
            ('pca', TruncatedSVD(20))])

    def fit(self, dialogs_df, y=None):
        texts = dialogs_df['dialog'].apply(get_texts, side='fit')
        self.pipe = self.pipe.fit(texts)
        return self

    def transform(self, dialogs_df):
        texts = dialogs_df['dialog'].apply(get_texts, side=self.side)
        feats = self.pipe.transform(texts)
        feats_df = pd.DataFrame(feats, columns=['pca_word_{side}_{i}'.format(side=self.side, i=i)
                                                for i in range(1, feats.shape[1]+1)],
                                index=dialogs_df.index)
        return feats_df

class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that unites several DataFrame transformers

    Fit several DataFrame transformers and provides a concatenated
    Data Frame

    Parameters
    ----------
    list_of_transformers : list of DataFrameTransformers

    """

    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers

    def transform(self, X, **transformparamn):
        """ Applies the fitted transformers on a DataFrame

        Parameters
        ----------
        X : pandas DataFrame

        Returns
        ----------
        concatted :  pandas DataFrame

        """

        concatted = pd.concat([transformer.transform(X)
                               for transformer in
                               self.fitted_transformers_], axis=1).copy()
        return concatted

    def fit(self, X, y=None, **fitparams):
        """ Fits several DataFrame Transformers

        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement

        Returns
        ----------
        self : object
        """

        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self
