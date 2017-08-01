from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
import numpy as np


def sp_scorer(model, x, y):
    pred = model.predict(x)
    if pred.std() == 0 or y.std() == 0:
        return 0
    return spearmanr(y, pred)

def spearman(y_true, y_pred):
    if y_true.std() == 0 or y_pred.std() == 0:
        return 0
    return spearmanr(y_true, y_pred)[0]


spearman_scorer = make_scorer(spearman, greater_is_better=True)

if __name__ == '__main__':
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

