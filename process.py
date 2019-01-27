import dask.dataframe as dd
from dateutil import parser
import pandas as pd

PRICING = 'D:\data\whisky\pricing.csv'

def process_to_hourly():
    """
    Processes flat csv file containing order book history into an hourly dataframe
    """
    # Define timezones
    tzinfos = {'BST': 3600,
               'GMT': 0}

    # Read file and fix time
    df = dd.read_csv(PRICING, parse_dates=False)
    unique_time = df.time.unique()
    fix_time = unique_time.map(lambda x: pd.Timestamp(parser.parse(x, tzinfos=tzinfos)) \
                               .tz_convert('UTC'), meta=pd.Series([], dtype='datetime64[ns, UTC]', name='fix_time'))
    fix_time = fix_time.dt.floor('h')  # Get hour only
    unique_time = dd.concat([unique_time, fix_time], axis=1) # Error here, but this is safe
    unique_time.columns = ['time', 'fixed_time']
    unique_time = unique_time.groupby('fixed_time').first().reset_index()
    df = df.merge(unique_time, left_on='time', right_on='time', how='inner')

    # Compute the dask graph
    df = df.compute()

    # Perform grouping (min/max) on computed frame
    max_buy = df[df['buy'] == 1].groupby(['pitchId', 'fixed_time'])[['limit']].max()
    min_sell = df[df['buy'] == 0].groupby(['pitchId', 'fixed_time']).limit.min()
    spreads = max_buy.join(min_sell, on=['pitchId', 'fixed_time'], how='outer', rsuffix='r')
    spreads.columns = ['max_buy', 'min_sell']
    spreads = spreads.reset_index()
    spreads.columns = ['pitchId', 'time', 'max_buy', 'min_sell']
    spreads.set_index('time', inplace=True)

    return spreads

if __name__ == '__main__':
    df = process_to_hourly()
    print(df.head())
