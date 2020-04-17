import datetime
import json
from pathlib import Path
import boto3
import pandas as pd
import requests


def get_utc_days(format='%Y-%m-%d'):
    utc = datetime.datetime.utcnow()
    yesterday = utc.date() - datetime.timedelta(1)
    return utc.strftime(format), yesterday.strftime(format)

def getOrders(to_csv=True):
    liveBoardLink = 'https://www.whiskyinvestdirect.com/view_market_json.do'
    res = requests.get(liveBoardLink)
    if res.status_code != requests.codes.ok:
        res.raise_for_status()
        return
    res = res.json()

    pitches = res["market"]["pitches"]
    pitches = pd.DataFrame(pitches)
    pitches.set_index(['pitchId'], inplace=True)

    # Generate and update pitches table
    pitch_table_cols = ['barrelTypeCode', 'bondQuarter', 'bondYear',
                        'categoryName', 'considerationCurrency', 'distillery',
                        'securityId', 'size', 'soldOut']

    pitch_table = pitches[pitch_table_cols]
    if to_csv:
        PITCHFILE = Path('/tmp/pitches.csv')
        pitch_table.to_csv(PITCHFILE, mode='w')

    # Generate and append pricing table
    pricing = pitches['prices'].apply(pd.Series)
    pricing.reset_index(inplace=True)
    pricing = pd.melt(pricing, id_vars='pitchId').set_index(['pitchId'])
    pricing = pricing['value'].apply(pd.Series).drop([0, 'actionIndicator', 'sell'], axis=1)
    pricing.dropna(axis=0, how='all',inplace=True)
    td, _ = get_utc_days('%Y-%m-%d_%H%M%S')
    updateTime = pd.Timestamp('now', tz='UTC')
    pricing['time'] = updateTime
    pricing = pricing.reset_index().set_index('time')
    
    if to_csv:
        PRICINGFILE = Path('/tmp/'+td+'.csv')
        if PRICINGFILE.exists():
            raise Exception(f'File arlready exists {PRICINGFILE}')
        else:
            pricing.to_csv(PRICINGFILE)

    # Finally merge pitches and pricing
    return pd.merge(pitch_table,pricing, how='inner', left_index=True, right_on='pitchId'), PRICINGFILE, PITCHFILE

def upload_to_s3(price_file, pitch_file):
    """Upload files to s3"""
    td, _ = get_utc_days('%Y-%m-%d')
    print(datetime.datetime.utcnow(), '>>> uploading file: ', price_file)
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('whisky-pricing')
    bucket.upload_file(str(price_file.absolute()), td+'/'+price_file.name) # Create a sub folder for today with timestamped data
    # bucket.upload_file(pitch_file, pitch_file.name) # only do this once a day


def lambda_handler(event, context):
    _, prices, pitches = getOrders(to_csv=True)
    upload_to_s3(prices, pitches)

    return {
        'statusCode': 200,
        'body': json.dumps('Lambda completed successfully.')
    }
