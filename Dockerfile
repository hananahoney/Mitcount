# syntax=docker/dockerfile:1
FROM python:3.10

WORKDIR /app

# Update system packages and install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libmysqlclient-dev \
    python3-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy over and install the pip requirements
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN yum -y install mesa-libGL

# Copy the rest of the application
COPY . .

CMD ["python","appcopy.py"]
