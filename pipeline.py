import json
import numpy as np
import pandas as pd
from functools import partial
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier

import freq_ngrams as fng
import metric
import utils
import w2v_features as w2v
import features


def update_model_stats(stats_file, params, results):
    import json
    import os.path
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = []
        
    stats.append({**results, **params})
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

        
def run_experiment(evaluator, params, stats_file):    
    import time
    
    params = init_params(params)
    start = time.time()
    scores = evaluator(params)
    exec_time = time.time() - start
    update_model_stats(stats_file, params, {**scores, 'exec-time-sec': exec_time})

    
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
    
    
def prepare_pairs(features, target):
    slices_x, slices_y = [], []
    for e in features.index:
        _features = features.drop(e, axis=0)
        _target = target.drop(e, axis=0)
        row = features.loc[e]
        row_y = target.loc[e]
        sample_x, _x, sample_y, _y = train_test_split(
            _features,
            _target,
            train_size=100,
            test_size=None,
            stratify=_target)
        sample_x = sample_x.copy()
        for col, val in row.items():
            sample_x['sample_{}'.format(col)] = val
        sample_dialog_id, sample_id = e
        sample_x['sample_dialog_id'] = sample_dialog_id
        sample_x['sample_id'] = sample_id
        sample_x = sample_x.reset_index().set_index(['dialogId', 'Id', 'sample_dialog_id', 'sample_id'])
        slices_x.append(sample_x)

        sample_y = sample_y.copy().to_frame()
        sample_y['sample_dialog_id'] = sample_dialog_id
        sample_y['sample_id'] = sample_id
        sample_y = sample_y.reset_index().set_index(['dialogId', 'Id', 'sample_dialog_id', 'sample_id']).squeeze()
        slices_y.append(sample_y < row_y)

    pairs = pd.concat(slices_x)
    pairs_y = pd.concat(slices_y)
    return pairs, pairs_y


def load_data(files, rec_limit=None):
    dialogs = []
    for path in files:
        with open(path) as f:
            dialogs += json.load(f)

    if rec_limit is not None:
        dialogs = dialogs[:rec_limit]
    return utils.dialogs_preproc(dialogs)


def init_params(overrides):
    defaults = {
        'valid_type': 'holdout',
        'pairwise': False,
        'use_fuzzy_match': True,
        'use_freq_ngrams': True,
        'use_pca_wc': True,
        
    }
    return {**defaults, **overrides}


def validate(params):    
    # days 24-27 was avalilable for training
    # day 28 and final holdout was not avalilable
    train_files = [
        'datasets/train_20170724.json', 
        'datasets/train_20170725.json',
        'datasets/train_20170726.json',
        'datasets/train_20170727.json'
    ]
    test_files = [
        'datasets/train_final.json'
    ]
    
    rec_limit = params.get('rec_limit')
    
    data_tr, target_tr = load_data(train_files, rec_limit)
    data_tst, target_tst = load_data(test_files, rec_limit)
    
    extractors = [
        features.min_len,
        features.max_len,
        features.avg_len,
        features.min_words,
        features.max_words,
        features.avg_words,
        features.ngram_intersection,
        features.get_human_words(),
        features.get_human_words_norm(),
        features.get_bot_words(),
        features.get_bot_words_norm(),
        features.get_stop_words_share(),
        features.min_partial_ratio,
        features.max_partial_ratio,
        features.avg_partial_ratio,
        features.dialogue_len,
        features.dialog_len_share,
        features.max_in_row,
        features.started_dialogue,
        features.finished_dialogue,
        features.min_reply_time,
        features.avg_reply_time,
        features.two_phrases_in_a_row,
        features.capital_letters_ratio,
        features.question_marks,
        features.has_a_question,
        features.punctuation_marks,
    ]
    
    if params['use_fuzzy_match']:
        extractors.extend([
            features.min_QRatio,
            features.max_QRatio,
            features.avg_QRatio,
            features.min_WRatio,
            features.max_WRatio,
            features.avg_WRatio,
        ])
        
    if params['use_w2v']:
        extractors.append(
            w2v.get_words_not_in_w2v()
        )

    transfs = [
        simple_transfromer(extractors),
    ]
    
    if params['use_freq_ngrams']:
        transfs.extend([
            fng.FreqNgrams(ngram_len=2),
            fng.FreqNgrams(ngram_len=3),
        ])
        
    if params['use_pca_wc']:
        transfs.extend([
            PCAWordFeatures(side='user1'),
            PCAWordFeatures(side='user2'),
            PCACharFeatures(side='user1'),
            PCACharFeatures(side='user2'),
        ])
        
    tr = DataFrameFeatureUnion(transfs)
    
    features_tr = tr.fit_transform(data_tr)
    features_tst = tr.transform(data_tst)
    
    if params['pairwise']:
        scorer = partial(metric.pairwise_spearman_scorer, direct_target=target_tst)
        features_tr, target_tr = prepare_pairs(features_tr, target_tr)
        features_tst, target_tst = prepare_pairs(features_tst, target_tst)
        lgbm_class = LGBMClassifier
    else:
        scorer = metric.spearman_scorer
        lgbm_class = LGBMRegressor
        
    est_type = params['est_type']
    if est_type == 'rf':
        est = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf']
        )
    elif est_type == 'xgb':
        if params['objective'] == 'poisson':
            objective = 'count:poisson'
        else:
            objective = 'reg:linear'
        est = XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            objective=objective)
    elif est_type == 'catb':
        if params['objective'] == 'poisson':
            loss = 'Poisson'
        else:
            loss = 'RMSE'
        est = CatBoostRegressor(loss_function=loss)
    elif est_type == 'lgbm':
        est = lgbm_class(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            objective=params['objective'])
        
    est.fit(features_tr, target_tr)
        
    score = scorer(est, features_tst, target_tst)
    return {'spearman': score}


def test_validate():
    params = {
        'est_type': 'lgbm',
        'n_estimators': 10,
        'max_depth': 3,
        'learning_rate': .3,
        'objective': 'poisson',
        'rec_limit': 50,
        'use_fuzzy_match': False,
        'use_w2v': False,
        'use_pca_wc': False,
        'use_freq_ngrams': False,
    }
    
    print(validate(init_params(params)))


def test_validate_pairwise():
    params = {
        'est_type': 'lgbm',
        'n_estimators': 10,
        'max_depth': 3,
        'learning_rate': .3,
        'objective': 'binary',
        'pairwise': True,
        'rec_limit': 100,
        'use_fuzzy_match': False,
        'use_w2v': False,
        'use_pca_wc': False,
        'use_freq_ngrams': False,
    }
    
    print(validate(init_params(params)))
