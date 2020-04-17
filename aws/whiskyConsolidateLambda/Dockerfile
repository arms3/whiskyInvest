FROM lambci/lambda:build-python3.8

COPY consolidate_reforecast.py consolidate_reforecast.py
COPY lambda_function.py lambda_function.py
COPY outlier_spline.py outlier_spline.py

RUN pip install s3fs sklearn pandas

# Finally run script
# RUN python -c 'import lambda_function; lambda_function.lambda_handler(1,2)'