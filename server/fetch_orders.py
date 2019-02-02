import pandas as pd
import requests
from pathlib import Path
from upload_s3 import get_utc_days


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
        pitch_table.to_csv(Path('days/pitches.csv'), mode='w')

    # Generate and append pricing table
    pricing = pitches['prices'].apply(pd.Series)
    pricing.reset_index(inplace=True)
    pricing = pd.melt(pricing, id_vars='pitchId').set_index(['pitchId'])
    pricing = pricing['value'].apply(pd.Series).drop([0, 'actionIndicator', 'sell'], axis=1)
    pricing.dropna(axis=0, how='all',inplace=True)
    td, _ = get_utc_days()
    updateTime = pd.Timestamp('now', tz='UTC')
    pricing['time'] = updateTime
    pricing = pricing.reset_index().set_index('time')

    if to_csv:
        FILE = Path('days/'+td+'.csv')
        if FILE.exists():
            pricing.to_csv(FILE, mode='a', header=False)
        else:
            pricing.to_csv(FILE)

    # Finally merge pitches and pricing
    return pd.merge(pitch_table,pricing, how='inner', left_index=True, right_on='pitchId')


if __name__ == '__main__':
    getOrders()
