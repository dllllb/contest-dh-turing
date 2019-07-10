from scipy.stats import spearmanr
import numpy as np
import pandas as pd


def spearman(y_true, y_pred):
    if y_true.std() == 0 or y_pred.std() == 0:
        return 0
    return spearmanr(y_true, y_pred)[0]


def spearman_scorer(est, features, target):
    pred = est.predict(features)
    return spearman(target, pred)


def pairwise_spearman_scorer(est, features, target, direct_target):
    pred = est.predict_proba(features)
    
    dialogue_preds = pd.Series(pred[:,1], index=features.index, name='scores')
    dialogue_preds = dialogue_preds.to_frame().reset_index()[['sample_dialog_id', 'sample_id', 'scores']]

    scores = dialogue_preds.groupby(['sample_dialog_id', 'sample_id']).sum()

    pred_true_df = pd.DataFrame({'target': direct_target, 'rank': scores.scores})

    return spearman(pred_true_df.target, pred_true_df['rank'])

    
def test_spearman_scorer():
    from sklearn.linear_model import LinearRegression
    from sklearn.dummy import DummyRegressor

    rng = np.random.RandomState(1)
    # Generate sample data
    y_pred = (5 * rng.rand(10)).reshape(-1,1)
    y_true = np.sin(y_pred[:, 0]).ravel()
    # regression
    regr = LinearRegression()
    regr = regr.fit(X=y_pred, y=y_true)
    print(spearman_scorer(regr, X=y_pred, y_true=y_true))
    # constant
    regr = DummyRegressor(strategy='mean')
    regr = regr.fit(X=y_pred, y=y_true)
    print(spearman_scorer(regr, X=y_pred, y_true=y_true))
