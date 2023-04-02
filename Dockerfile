FROM python:3.10.6-buster
# FROM --platform=linux/amd64 tensorflow/tensorflow:2.10.0
WORKDIR /prod
# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt
# RUN pip install -r requirements.txt
# Then only, install taxifare!
COPY stories_generator stories_generator
CMD uvicorn stories_generator.api.fast:app --reload --host 0.0.0.0 --port $PORT