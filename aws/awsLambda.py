#!/usr/bin/env python
import gc
import boto3
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dateutil import parser
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
    print('Loading hourly data...')
    tzinfos = {'BST': 3600,
               'GMT': 0}

    df = dd.read_csv('s3://whisky-pricing/20*.csv', parse_dates=False)
    # debug local
    # df = dd.read_csv(r'C:\\Users\\sincl\\OneDrive\\Desktop\\days\\2018-12-2*.csv', parse_dates=False)
    unique_time = df.time.unique()
    fix_time = unique_time.map(lambda x: pd.Timestamp(parser.parse(x, tzinfos=tzinfos))
                               .tz_convert('UTC'), meta=pd.Series([], dtype='datetime64[ns, UTC]', name='fix_time'))

    # Regroup to hourly
    print('Regrouping to hourly...')
    fix_time = fix_time.dt.floor('h')  # Get hour only
    unique_time = dd.concat([unique_time, fix_time], axis=1, )
    unique_time.columns = ['time', 'fixed_time']
    unique_time = unique_time.groupby('fixed_time').first().reset_index()
    df = df.merge(unique_time, left_on='time', right_on='time', how='inner')
    df = df.compute()
    df.drop('time', axis=1, inplace=True)
    df.rename({'fixed_time': 'time'}, axis=1, inplace=True)

    # Calculate spreads
    print('Calculating spreads...')
    max_buy = df[df['buy'] == 1].groupby(['pitchId', 'time'])[['limit']].max()
    min_sell = df[df['buy'] == 0].groupby(['pitchId', 'time'])[['limit']].min()
    del df
    gc.collect()
    spreads = max_buy.join(min_sell, on=['pitchId', 'time'], how='outer', rsuffix='r')
    spreads.columns = ['max_buy', 'min_sell']
    spreads = spreads.reset_index()
    spreads.columns = ['pitchId', 'time', 'max_buy', 'min_sell']
    spreads.set_index('time', inplace=True)
    return spreads


def index2int(index):
    """Helper function to convert timeindexes to ints"""
    return index.astype(np.int64).values.reshape(-1,1)


def int2dt(array):
    """Helper function to convert int64 to timeseries[ns]"""
    return pd.to_datetime(pd.Series(array.squeeze(), name='timestamp'),utc=True)


def run_regression(df):
    lr = OutlierLinearRegression(0.5, SplineRegressor(smooth_factor=11, order=1))
    linreg = {}
    preds = []
    print('Running regression...')
    for pitch, spread in df.groupby('pitchId'):
        X = index2int(spread.max_buy.dropna().index)
        y = spread.max_buy.dropna().values
        lr.fit(X, y)
        linreg[pitch] = {'intercept': lr.intercept_, 'slope': lr.coef_, 'r_value': lr.score(X, y)}
        # Predict within timeseries (for plotting)
        preds.append(pd.DataFrame({'time': int2dt(X), 'predict': lr.predict(X), 'pitchId': np.full(len(X), pitch)})\
                     .set_index('pitchId'))

    linreg = pd.DataFrame.from_dict(linreg, orient='index')

    # Combine preds append to df and save
    print('Combining daily predictions and uploading...')
    preds = pd.concat(preds, axis=0).reset_index()
    df = df.reset_index().merge(preds, on=['pitchId','time']).reset_index().set_index('time')
    df.drop('index',axis=1, inplace=True) # Remove the rangindex artifact created in merge
    df.to_csv('spreads.csv')
    bucket.upload_file('spreads.csv', 'spreads.csv')

    # Regroup to daily # Save daily to csv
    print('Uploading daily pricing...')
    df.groupby('pitchId').resample('D')[['min_sell', 'max_buy', 'predict']].mean().to_csv('mean_daily_spread.csv')
    bucket.upload_file('mean_daily_spread.csv', 'mean_daily_spread.csv')

    # Fees
    market_fee = 1.75 / 100.0  # % per transaction, rounded to nearest pence
    storage_fee = 0.15  # pence per annum (charged monthly)
    dayseconds = 24 * 3600 * 1e9  # nanoseconds in a day

    # Calculate returns
    print('Calculating returns...')
    pitches = df.sort_index(ascending=True).dropna().reset_index().groupby('pitchId').last() # Warning here about converting to naive datetime not sure why
    pitches = pitches.join(linreg)
    pitches = pitches.dropna()

    pitches['adjusted_slope'] = (dayseconds * pitches.slope) - (storage_fee / 365.25)
    pitches['spread_fee_adj'] = pitches.min_sell * (1 + market_fee) - pitches.max_buy
    pitches['days_to_close_spread'] = pitches.spread_fee_adj / pitches.adjusted_slope
    pitches['fee_adjusted_purchase_cost'] = pitches.min_sell * (1 + market_fee)
    pitches['predicted1y'] = pitches.intercept + (pitches.time.astype(np.int64) * pitches.slope) \
                             + (pitches.adjusted_slope * 365.25)  # Use model price here
    pitches['fee_adjusted_sell_cost'] = pitches.predicted1y * (1 - market_fee)
    pitches['annual_return'] = -100 + 100 * pitches.fee_adjusted_sell_cost / pitches.fee_adjusted_purchase_cost

    # Save csv
    print('Uploading model...')
    pitches.to_csv('pitch_models.csv')
    bucket.upload_file('pitch_models.csv', 'pitch_models.csv')

    return pitches


if __name__ == '__main__':
    print('Reticulating splines...')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('whisky-pricing')
    df = get_hourly()
    pitches = run_regression(df)
    print(pitches.head(3))
