import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


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


def get_hourly():
    # Load up full data
    # Regroup to hourly
    # Calculate spreads
    # Dump results to csv
    # run_regression
    # Regroup to daily
    # Save daily to csv
    return


def run_regression(df):
    lr = OutlierLinearRegression(0.5, SplineRegressor(smooth_factor=11, order=1))
    linreg = {}
    for pitch, spread in df.groupby('pitchId'):
        X = spread.max_buy.dropna().index.astype(np.int64).values.reshape(-1, 1)
        y = spread.max_buy.dropna().values
        lr.fit(X, y)
        linreg[pitch] = {'intercept': lr.intercept_, 'slope': lr.coef_, 'r_value': lr.score(X, y)}

    linreg = pd.DataFrame.from_dict(linreg, orient='index')

    # Fees
    market_fee = 1.75 / 100.0  # % per transaction, rounded to nearest pence
    storage_fee = 0.15  # pence per annum (charged monthly)
    dayseconds = 24 * 3600 * 1e9  # nanoseconds in a day

    # Calculate returns
    pitches = df.sort_index(ascending=True).dropna().reset_index().groupby('pitchId').last()
    pitches = pitches.join(linreg)
    pitches = pitches.dropna()
    pitches['adjusted_slope'] = (dayseconds * pitches.slope) - (storage_fee / 365.25)
    pitches['spread_fee_adj'] = np.round(pitches.min_sell * (1 + market_fee), 2) - pitches.max_buy
    pitches['days_to_close_spread'] = pitches.spread_fee_adj / pitches.adjusted_slope
    pitches['fee_adjusted_purchase_cost'] = np.round(pitches.min_sell * (1 + market_fee), 2)
    pitches['annual_return'] = 100 * 365.25 * pitches.adjusted_slope / pitches.fee_adjusted_purchase_cost

    return pitches

if __name__ == '__main__':
