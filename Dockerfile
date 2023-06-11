# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /app

# COPY requirements.txt requirements.txt
COPY require.txt requirements.txt
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y default-libmysqlclient-dev
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0"]