# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /app

# Update system packages and install build-essential, default-libmysqlclient-dev, and python3-dev
RUN apt-get update && apt-get install -y \
    build-essential \
    default-libmysqlclient-dev \
    python3-dev

# Upgrade pip
RUN pip install --upgrade pip

# Copy over and install the pip requirements
COPY require.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy the rest of the application
COPY . .

CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0"]
