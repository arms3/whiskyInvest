from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
import numpy as np


class OutlierLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, outlierpct, reg=LinearRegression()):
        self.outlierpct = outlierpct
        self.reg = reg

    def fit(self, X, y):
        reg = self.reg
        reg.fit(X, y)
        errors = np.abs(y - reg.predict(X))
        number_to_remove = int(self.outlierpct * len(X))
        try:
            keepidx = np.argsort(errors)[:-number_to_remove].values
        except:
            keepidx = np.argsort(errors)[:-number_to_remove]
        keepidx.sort()
        X_, y_ = X[keepidx], y[keepidx]
        reg = self.reg
        self.reg = reg.fit(X_, y_)

        # Return coef and intercept features
        self.coef_ = self.reg.coef_
        self.intercept_ = self.reg.intercept_
        return self

    def predict(self, X, y=None):
        return self.reg.predict(X)

    def score(self, X, y):
        return self.reg.score(X, y)


class SplineRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, order=2, smooth_factor = None):
        self.k = order
        self.s = smooth_factor

    def _spline_tangent(self, spl, x0):
        d = spl.derivatives(x0)
        self.coef_ = d[1] *1.0
        self.intercept_ = np.sum(d[0] - (d[1] * x0))

        def pred_func(x):
            return d[1] * (x - x0) + d[0]
        return pred_func

    def fit(self, X, y):
        self.spl = UnivariateSpline(X, y, k=self.k, s=self.s)
        self.maxx = max(X)
        self.future = self._spline_tangent(self.spl, self.maxx)
        return self

    def predict(self, X, y=None):
        in_train = X < self.maxx
        return np.concatenate([self.spl(X[in_train]), self.future(X[~in_train])])