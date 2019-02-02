#!/usr/bin/env bash
echo $(date) 'Fetching prices' >> whisky.log
cd /home/ubuntu/
/home/ubuntu/miniconda3/bin/python /home/ubuntu/fetch_orders.py >> whisky.log
echo $(date) 'Finished run \n' >> whisky.log