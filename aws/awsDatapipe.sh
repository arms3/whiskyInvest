#!/usr/bin/env bash
echo "Running bash script"
conda install -y -c anaconda boto3 
conda install -y pandas=0.24 scipy s3fs numpy
conda install dask
echo "Finished installing dependencies, running python script..."
python awsLambda.py
echo "Shell script done!"