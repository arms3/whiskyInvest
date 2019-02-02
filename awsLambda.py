#!/usr/bin/env python
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
import pandas as pd
from dateutil import parser
import dask.dataframe as dd
import gc
import boto3


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
    tzinfos = {'BST': 3600,
               'GMT': 0}

    df = dd.read_csv('s3://whisky-pricing/20*.csv', parse_dates=False)
    unique_time = df.time.unique()
    fix_time = unique_time.map(lambda x: pd.Timestamp(parser.parse(x, tzinfos=tzinfos)) \
                               .tz_convert('UTC'), meta=pd.Series([], dtype='datetime64[ns, UTC]', name='fix_time'))

    # Regroup to hourly
    fix_time = fix_time.dt.floor('h')  # Get hour only
    unique_time = dd.concat([unique_time, fix_time], axis=1, )
    unique_time.columns = ['time', 'fixed_time']
    unique_time = unique_time.groupby('fixed_time').first().reset_index()
    df = df.merge(unique_time, left_on='time', right_on='time', how='inner')
    df = df.compute()

    # Calculate spreads
    max_buy = df[df['buy'] == 1].groupby(['pitchId', 'fixed_time'])[['limit']].max()
    min_sell = df[df['buy'] == 0].groupby(['pitchId', 'fixed_time'])[['limit']].min()
    del df
    gc.collect()
    spreads = max_buy.join(min_sell, on=['pitchId', 'fixed_time'], how='outer', rsuffix='r')
    spreads.columns = ['max_buy', 'min_sell']
    spreads = spreads.reset_index()
    spreads.columns = ['pitchId', 'time', 'max_buy', 'min_sell']
    spreads.set_index('time', inplace=True)

    # Free some RAM
    del max_buy, min_sell
    gc.collect()

    # Dump results to csv - this need to be on aws using boto3
    # TODO: Assess if this is necessary
    print('Uploading spreads...')
    spreads.to_csv('spreads.csv')
    bucket.upload_file('spreads.csv', 'spreads.csv')

    # Regroup to daily # Save daily to csv
    print('Uploading daily pricing...')
    spreads.groupby('pitchId').resample('D')[['min_sell', 'max_buy']].mean().to_csv('mean_daily_spread.csv')
    bucket.upload_file('mean_daily_spread.csv', 'mean_daily_spread.csv')

    return spreads


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

    # Save csv
    print('Uploading model...')
    pitches.to_csv('pitch_models.csv')
    bucket.upload_file('pitch_models.csv', 'pitch_models.csv')

    return pitches


def lambda_function(context, event):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('whisky-pricing')
    df = get_hourly()
    pitches = run_regression(df)
    print(pitches.head(3))


if __name__ == '__main__':
    print('Reticulating splines...')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('whisky-pricing')
    print('Loading hourly data...')
    df = get_hourly()
    print('Running regression...')
    pitches = run_regression(df)
    print(pitches.head(3))
