#!/usr/bin/env bash
echo $(date) 'Fetching prices' >> whisky.log
python fetch_orders.py >> whisky.log
echo $(date) 'Finished run \n' >> whisky.log
