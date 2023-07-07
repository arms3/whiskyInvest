# Docker image build for local testing
# FROM heroku/heroku:22
FROM python:3.11

WORKDIR /app
RUN apt-get update && apt-get -y install python3-pip

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY whisky/ ./whisky
EXPOSE 8001
CMD ["gunicorn", "--timeout", "300", "whisky:server", "-b", "0.0.0.0:8001"]