# syntax=docker/dockerfile:1
FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Update system packages and install build-essential, default-libmysqlclient-dev, and python3-dev
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libmysqlclient-dev \
    python3-dev

# Upgrade pip
RUN pip install --upgrade pip

# Copy over and install the pip requirements
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

CMD ["python","appcopy.py"]
