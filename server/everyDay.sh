#!/usr/bin/env bash
echo '>>>' $(date) 'Uploading prices' >> whisky.log
cd /home/ubuntu/
/home/ubuntu/miniconda3/bin/python /home/ubuntu/upload_s3.py >> whisky.log
echo '>>>' $(date) 'Finished run \n' >> whisky.log