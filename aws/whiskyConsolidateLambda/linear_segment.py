from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np


class SegmentedLinearRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, n_seg=4, min_segment_length=200):
        self.m = min_segment_length
        self.n_seg = n_seg
        self.dtr = DecisionTreeRegressor(max_leaf_nodes=n_seg)

    def fit(self, X, y):
        # First fit decision tree regressor on slopes
        dy = np.gradient(y, X.ravel())
        self.dtr.fit(X, dy.reshape(-1, 1))
        self.dys_dt = self.dtr.predict(X).flatten()

        # Initialize and get sorted unique gradients
        self.lr = []
        _, idx = np.unique(self.dys_dt, return_index=True)
        unique = self.dys_dt[np.sort(idx)]

        # For each group of gradients add a linear regressor
        at_least_one_regressor = False
        for dy in unique:
            msk = self.dys_dt == dy
            # Only add to our regressors if segment is long enough
            if np.sum(msk) >= self.m:
                self.lr.append(LinearRegression())
                self.lr[-1].fit(X[msk], y[msk])
                at_least_one_regressor = True

        # Fallback for small datasets - linear regression
        if not at_least_one_regressor:
            self.lr.append(LinearRegression())
            self.lr[-1].fit(X, y)

        # Return last coefficient and intercept features
        self.coef_ = self.lr[-1].coef_[0]
        self.intercept_ = self.lr[-1].intercept_
        return self

    def predict(self, X, y=None):
        # Assumes we're predicting the future so only use the latest regressor
        return self.lr[-1].predict(X)

    def score(self, X, y):
        # Get score for latest segment
        return self.lr[-1].score(X,y)