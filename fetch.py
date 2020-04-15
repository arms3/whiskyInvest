import numpy as np
import pandas as pd
import requests
from requests_futures.sessions import FuturesSession
import re
import ast
import os
from scipy import stats
from s3fs import S3FileSystem
# from flask.ext.cache import Cache


# Setup flask cache
# cache = Cache(config={'CACHE_TYPE': 'simple'})


def checkpoint(fname, df_read_args={}):
    """Function decorator to checkpoint functions that return dataframes."""
    def checkpoint_decorator(func):
        def func_wrapper(*args, **kwargs):
            cache = 'cache\\' + fname
            if os.path.isfile(cache):
                # TODO: Add timeout even if file exists - check file update time or insert into database
                print('Cache exists, returning cached data')
                return pd.read_csv(cache, **df_read_args)
            else:
                print('No cache exists')
                df = func(*args, **kwargs)
                df.to_csv(cache, index=False)
                return df
        return func_wrapper
    return checkpoint_decorator


@checkpoint('pitches.csv')
def get_pitches():
    cookies = {'considerationCurrency': 'GBP'}
    liveboardlink = 'https://www.whiskyinvestdirect.com/view_market_json.do'
    res = requests.get(liveboardlink, cookies=cookies)
    if res.status_code != requests.codes.ok:
        res.raise_for_status()
        return

    res = res.json()
    pitches = res["market"]["pitches"]
    pitches = pd.DataFrame(pitches)
    pitches.set_index('pitchId', inplace=True)
    pitches = pitches[pitches.considerationCurrency == 'GBP']

    # Generate and update pitches table
    pitch_table_cols = ['barrelTypeCode', 'bondQuarter', 'bondYear',
       'categoryName', 'considerationCurrency', 'distillery',
       'securityId', 'size', 'soldOut']
    pitch_table = pitches[pitch_table_cols].copy()

    # Clean up and create whisky_type
    pitch_table['formattedDistillery'] = pitch_table['distillery'].apply(lambda x: x.lower().replace("_", '-'))
    cols = ['formattedDistillery', 'bondYear', 'bondQuarter', 'barrelTypeCode']
    pitch_table['whisky_type'] = pitch_table[cols].astype(str).apply(lambda x: '_'.join(x), axis=1)

    # Analyse pricing spread
    # This is a bit hacky, and results in an error from pandas, but works
    pricing = pitches['prices'].apply(pd.Series)
    pricing.reset_index(inplace=True)
    pricing = pd.melt(pricing, id_vars='pitchId').set_index(['pitchId'])
    pricing = pricing['value'].apply(pd.Series).drop([0, 'actionIndicator', 'sell'], axis=1)
    pricing.dropna(axis=0, how='all', inplace=True)
    pricing = pricing.groupby(['pitchId', 'buy'])['limit'].agg([np.min, np.max]).unstack()
    pricing.columns = ['best_sell', 'worst_sell', 'worst_buy', 'best_buy']
    pricing = pricing[['best_buy', 'best_sell']]
    pricing['spread'] = pricing.best_sell - pricing.best_buy

    # Merge pricing info back into pitches
    pitch_table = pitch_table.join(pricing).sort_index().reset_index()

    return pitch_table


def parse_chart(resp, *args, **kwargs):
    """Evaluates chart data from request object, runs as a hook within requests-futures so that data is processed in
    the background. Takes a response object and attached a dataframe to the response object."""
    # match chart container data
    c_dat = re.search(r"Chart\.drawChart\( \$\('#chartContainer'\), (.*?)\)", resp.text).group(1)
    # interpret as list
    c_dat = ast.literal_eval(c_dat)
    df = pd.DataFrame(c_dat[0])
    df['dealDate'] = df['dealDate'].astype("datetime64[ms]")
    df['Currency'] = c_dat[1]
    resp.df = df


@checkpoint('all_whisky.csv', df_read_args={'parse_dates': ['dealDate']})
def scrape_all_charts(pitches):
    """Fetches all chart data from a list of pitches."""
    links = []
    whisky_types = []
    for index, whisky in pitches.iterrows():
        chart_link = 'https://www.whiskyinvestdirect.com/{}/{}/{}/{}/chart.do'.format(whisky.formattedDistillery,
                                                                                      whisky.bondYear,
                                                                                      whisky.bondQuarter,
                                                                                      whisky.barrelTypeCode)
        links.append(chart_link)
        whisky_types.append(whisky.whisky_type)

    # Code to handle requests in parallel
    cookies = {'considerationCurrency': 'GBP'}
    session = FuturesSession(max_workers=10)
    requests = [(session.get(l, cookies=cookies, hooks={'response': parse_chart}),l,5) for l in links]

    # Retry 5 times
    responses = []
    while requests:
        request, arg, tries = requests.pop(0)
        resp = request.result()
        # print('arg={}, tries={}, resp={}'.format(arg, tries, resp))
        if resp.status_code > 299 and tries > 1:
            requests.append((session.get(arg, cookies=cookies, hooks={'response': parse_chart}),arg, tries - 1, ))
        else:
            responses.append(request)

    dfs = []
    for whisky, response in zip(whisky_types, responses):
        df = response.result().df
        df['whisky_type'] = whisky
        dfs.append(df)

    dfs = pd.concat(dfs, axis=0)
    dfs = pd.merge(dfs, pitches, left_on='whisky_type', right_on='whisky_type', how='inner')

    # Remove rows where currency doesn't match
    dfs = dfs[dfs.Currency == dfs.considerationCurrency]
    dfs.drop('Currency', axis=1, inplace=True)

    return dfs

@checkpoint('analysed_pitches.csv')
def analyse_prices(pitches, df):
    """Perform linear regression analysis on all pitches individually."""
    grouped = df.groupby('pitchId')
    linreg = {}
    for key, grp in grouped:
        slope, intercept, r_value, p_value, std_err = stats.linregress(grp.day, grp.priceAvg)
        linreg[int(key)] = {'slope': slope, 'intercept': intercept, 'r_value': r_value}

    # Convert to df
    linreg = pd.DataFrame.from_dict(linreg, orient='index')

    # Calculate returns
    pitches.set_index('pitchId', inplace=True)
    pitches = pitches.join(linreg)
    pitches = calc_returns(pitches)

    # Clean up and return
    pitches.reset_index(inplace=True)
    return pitches


def calc_returns(pitches):
    # Fees
    market_fee = 1.75 / 100.0  # % per transaction, rounded to nearest pence
    storage_fee = 0.15  # pence per annum (charged monthly)
    dayseconds = 24 * 3600 * 1e9  # nanoseconds in a day

    #TODO: Update this based on timeline slider, e.g. what return if i sell in 2 years?

    pitches['adjusted_slope'] = (dayseconds * pitches.slope) - (storage_fee / 365.25)
    pitches['spread_fee_adj'] = pitches.best_sell * (1 + market_fee) - pitches.best_buy
    pitches['days_to_close_spread'] = pitches.spread_fee_adj / pitches.adjusted_slope
    pitches['fee_adjusted_purchase_cost'] = pitches.best_sell * (1 + market_fee)
    pitches['model_price_now'] = model_prices(pitches, pd.Timestamp.now('UTC'))
    pitches['predicted1y'] = pitches.model_price_now + (pitches.adjusted_slope * 365.25)
    pitches['fee_adjusted_sell_cost'] = pitches.predicted1y * (1 - market_fee)
    pitches['annual_return'] = -100 + 100 * pitches.fee_adjusted_sell_cost / pitches.fee_adjusted_purchase_cost

    # Update return for owned whiskies
    if 'owned' in pitches.columns:
        # What is the increase in capital of expected sell in 1 year v. sell price now (after fees)
        annual_return_if_owned = -100 + 100 * pitches.predicted1y * (1-market_fee) / (pitches.best_buy * (1-market_fee))
        pitches['annual_return'] = (pitches.owned * annual_return_if_owned) + (~pitches.owned * pitches.annual_return)

    # Calculate advanced strategy return
    pitches['strike_price_5%'] = 1.05 * pitches.fee_adjusted_purchase_cost / (1 - market_fee)  # Sell price to achive 5% return
    pitches['days_to_5%'] = (((1.05 * pitches.fee_adjusted_purchase_cost) / (1 - 0.0175)) - pitches.model_price_now) \
                            / pitches.adjusted_slope
    pitches['APR_5%'] = 100 * (-1 + 1.05 ** (365.24 / pitches['days_to_5%']))

    return pitches

def model_prices(pitches, time):
    return pitches.intercept + (np.int64(time.value) * pitches.slope)

@checkpoint('mean_daily_spread.csv')
def alls3cache(s3):
    return pd.read_csv(s3.open('whisky-pricing/mean_daily_spread.csv', mode='rb'))

@checkpoint('pitch_models.csv')
def pitchcache(s3):
    return pd.read_csv(s3.open('whisky-pricing/pitch_models.csv', mode='rb'))

def get_from_s3():
    # Uses default config from environment variables
    s3 = S3FileSystem(anon=False)
    all_whisky = alls3cache(s3)
    analysed_pitches = pitchcache(s3)
    pitches = get_pitches()

    pitches = pitches.set_index('pitchId')
    analysed_pitches = analysed_pitches.set_index('pitchId')
    pitches = pitches.join(analysed_pitches, how='inner') # Ignores missing pitches as get_pitches filters out GBP
    # Clear old returns calculations and recalculate
    pitches.drop(['max_buy', 'min_sell', 'spread_fee_adj','days_to_close_spread', 'fee_adjusted_purchase_cost',
                  'annual_return', 'time', 'predict'], axis=1, inplace=True)
    # pitches.slope = pitches.slope*24 * 3600 * 1e9 # This must be commented
    pitches = calc_returns(pitches)
    return pitches, all_whisky


def load_all_data():
    """Runs all of above routines to load all whisky data."""
    pitches = get_pitches()
    all_whisky = scrape_all_charts(pitches)
    pitches2 = analyse_prices(pitches, all_whisky)
    return pitches2, all_whisky


if __name__ == '__main__':
    # ptch = get_pitches()
    # df = scrape_all_charts(ptch)
    # ptch2 = analyse_prices(ptch, df)
    # print(ptch2.head())
    get_from_s3()
