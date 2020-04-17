import json
from consolidate_reforecast import main

def lambda_handler(event, context):
    main() # Do main steps from consolidate reforecast
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
