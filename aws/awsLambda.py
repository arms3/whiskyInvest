#!/usr/bin/env python
import gc
import s3fs
import numpy as np
import pandas as pd
from dateutil import parser
from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

# Initialize S3 connection
s3 = s3fs.S3FileSystem(anon=False)

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


def index2int(index):
    """Helper function to convert timeindexes to ints"""
    return index.astype(np.int64).values.reshape(-1,1)


def int2dt(array):
    """Helper function to convert int64 to timeseries[ns]"""
    return pd.to_datetime(pd.Series(array.squeeze(), name='timestamp'),utc=True)


def regroup_df_hourly(df, time_column='time'):
    # Get list of unique timestamps
    unique_time = df.time.unique()

    # Ensure we have UTC aware timestamps
    tzinfos = {'BST': 3600, 'GMT': 0}
    fix_time = pd.Series([pd.Timestamp(parser.parse(x, tzinfos=tzinfos)).tz_convert('UTC') for x in unique_time],dtype='datetime64[ns, UTC]', name='fix_time')

    fix_time = fix_time.dt.floor('h')  # Get hour only
    unique_time = pd.concat([pd.Series(unique_time, name=time_column), fix_time], axis=1)
    unique_time = unique_time.groupby('fix_time').first().reset_index()
    df = df.merge(unique_time, left_on=time_column, right_on=time_column, how='inner') # Drop all timestamps not on the hour
    df.drop(time_column, axis=1, inplace=True)
    df.rename({'fix_time': time_column}, axis=1, inplace=True)
    return df


def get_hourly():
    # Load up full data
    print("Reading existing hourly spread data...")
    df = pd.read_csv('s3://whisky-pricing/spreads.csv', parse_dates=['time'])
   
   # Get latest imported date
    print("Importing non consolidated dates...")
    last_date = pd.to_datetime(df.time.max()).date()
    today = pd.to_datetime('today').date()

    # Get missing days
    missing_days = []
    delta = today - last_date
    for i in range(delta.days):
        day = last_date + pd.Timedelta(i+1, unit='D')
        try:
            df_day = pd.read_csv(f's3://whisky-pricing/{day}.csv', parse_dates=False)
            print(f'Imported day: {day}')
            missing_days.append(df_day)
        except:
            print(f'Day not found: {day}')

    if len(missing_days) == 0:
        return df, False # Return flag so that further processing can be skipped
    
    # Concat all the missing days
    missing = pd.concat(missing_days,axis=0)

    # Regroup to hourly
    missing = regroup_df_hourly(missing, time_column='time')

    # Calculate spreads
    print('Calculating spreads...')
    max_buy = missing[missing['buy'] == 1].groupby(['pitchId', 'time'])[['limit']].max()
    min_sell = missing[missing['buy'] == 0].groupby(['pitchId', 'time'])[['limit']].min()
    missing = max_buy.join(min_sell, on=['pitchId', 'time'], how='outer', rsuffix='r')
    missing.columns = ['max_buy', 'min_sell']
    missing = missing.reset_index()
    missing.columns = ['pitchId', 'time', 'max_buy', 'min_sell']

    # Merge with existing data
    df = pd.concat([df,missing],axis=0)
    df.drop('predict',axis=1, inplace=True)
    return df, True # Return flag to continue processing


def run_regression(df):
    lr = OutlierLinearRegression(0.5, SplineRegressor(smooth_factor=None, order=1))
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

    # Combine preds append to df
    print('Combining daily predictions...')
    preds = pd.concat(preds, axis=0).reset_index()
    df = df.reset_index().merge(preds, on=['pitchId','time']).reset_index().set_index('time')
    df.drop('index',axis=1, inplace=True) # Remove the rangindex artifact created in merge
    return df, linreg

def regroup_to_daily(df):
    # Regroup to daily
    print('Regrouping daily pricing...')
    df.groupby('pitchId').resample('D')[['min_sell', 'max_buy', 'predict']].mean()
    return df

def calculate_returns(df, linreg):
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
    return pitches


if __name__ == '__main__':
    print('Reticulating splines...')
    hourly, continueProcessing = get_hourly()
    if continueProcessing:
        # Process hourly predictions and upload to S3
        hourly_pred, linreg = run_regression(hourly)
        hourly_pred.to_csv('spreads.csv')
        s3.put('spreads.csv','s3://whisky-pricing/spreads.csv')
        
        # Process daily data and upload to S3
        daily = regroup_to_daily(hourly_pred)
        daily.to_csv('mean_daily_spread.csv')
        s3.put('mean_daily_spread.csv', 's3://whisky-pricing/mean_daily_spread.csv')

        # Caluclate returns and upload to S3
        pitches = calculate_returns(daily, linreg)
        pitches.to_csv('pitch_models.csv')
        s3.put('pitch_models.csv', 's3://whisky-pricing/pitch_models.csv')
        print(pitches.head(3))
