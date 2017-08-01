import pandas as pd
from random import sample
import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold


def dialogs_preproc(dialogs):
    dia_lst = []
    for dia in dialogs:
        for u in dia['users']:
            d = deepcopy(dia)
            uname = u['id']
            for phrase in d['thread']:
                phrase['userId'] = 'user1' if phrase['userId'] == uname else 'user2'

            if 'evaluation' in d:
                evl = [e for e in d['evaluation'] if e['userId'] == uname][0]
                row = [d['dialogId'], d, uname, evl['quality']]
            else:
                row = [d['dialogId'], d, uname]
            dia_lst.append(row)

    if 'evaluation' in dialogs[0]:
        df = pd.DataFrame(
            dia_lst,
            columns=['dialogId', 'dialog', 'Id', 'quality']
        ).set_index(['dialogId', 'Id'])
        X = df.drop('quality', axis=1)
        y = df['quality']
        return X, y
    else:
        X = pd.DataFrame(
            dia_lst,
            columns=['dialogId', 'dialog', 'Id']
        ).set_index(['dialogId', 'Id'])
        return X


def train_test_split(features, target):
    dialog_ids = features.reset_index()["dialogId"].unique().tolist()
    train_dialogs = sample(list(dialog_ids), int(len(dialog_ids) * 0.8))
    train_feats = features.sort_index().loc[train_dialogs]
    test_feats = features.reset_index(level=1).drop(train_dialogs).set_index('id')
    if target is not None:
        train_target = target.sort_index().loc[train_dialogs]
        test_target = features.reset_index(level=1).drop(train_dialogs).set_index('id').iloc[:,0]
        return train_feats, test_feats, train_target, test_target
    return train_feats, test_feats


def dialog_kfold(X, n_folds=5, random_seed=0):
    # sklearn cv format
    arr = X.copy().reset_index().dialogId.unique()
    n_samples = len(arr)
    np.random.RandomState(random_seed).shuffle(arr)
    step = int(n_samples / n_folds) + (n_samples % n_folds > 0)
    folds = []
    for i in range(0, n_samples, step):
        folds.append((X.reset_index().dialogId.isin(arr[i:i + step]).values,
                     ~X.reset_index().dialogId.isin(arr[i:i + step]).values))
    return folds



def stratified_dialog_kfold(X, dialogs, n_folds=5, random_seed=0):
    # sklearn cv format
    skf = StratifiedKFold(n_splits=n_folds, random_state = random_seed)
    dialogs_dict = { dialog['dialogId']: dialog for dialog in dialogs}
    
    arr = X.copy().reset_index().dialogId.unique()
    isbot = [
            dialogs_dict[dialogId]['users'][0]['userType']=='Bot' or dialogs_dict[dialogId]['users'][1]['userType']=='Bot'\
            for dialogId in arr
            ]
    folds = []
    for _, test_index in skf.split(arr, isbot):
        folds.append((X.reset_index().dialogId.isin(arr[test_index]).values,
                     ~X.reset_index().dialogId.isin(arr[test_index]).values))
    return folds
