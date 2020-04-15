FROM ubuntu:latest

COPY awsLambda.py /root/awsLambda.py

# Install Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
RUN conda --version

# Install dependencies
RUN conda install -y boto3 pandas dask scipy s3fs numpy scikit-learn

# Finally run script
# RUN python /root/awsLambda.py