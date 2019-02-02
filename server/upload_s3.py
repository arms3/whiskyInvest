import boto3
import datetime

def get_utc_days():
    utc = datetime.datetime.utcnow()
    yesterday = utc.date() - datetime.timedelta(1)
    return utc.strftime('%Y-%m-%d'), yesterday.strftime('%Y-%m-%d')

def main():
    """Upload yesterday's file to s3"""
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('whisky-pricing')
    td, yd = get_utc_days()
    fname = yd + '.csv'
    print(datetime.datetime.utcnow(), '>>> uploading file: ', fname)
    bucket.upload_file('days/'+fname, fname)

if __name__ == '__main__':
    main()