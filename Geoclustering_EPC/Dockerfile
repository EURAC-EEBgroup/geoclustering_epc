FROM python:3.11-slim

WORKDIR /app


COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . .



