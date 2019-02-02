#!/usr/bin/env bash
echo '>>>' $(date) 'Uploading prices' >> whisky.log
python upload_s3.py >> whisky.log
echo '>>>' $(date) 'Finished run \n' >> whisky.log