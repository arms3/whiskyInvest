docker build -t whisky .
docker run -v $env:USERPROFILE\.aws:/root/.aws:ro --rm -p 8001:8001 whisky