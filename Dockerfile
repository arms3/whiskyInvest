# Docker image build for local testing
FROM heroku/heroku:20

WORKDIR /app
RUN apt-get update && apt-get -y install python3-pip

COPY . ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]