FROM python:3.11.5-bookworm

RUN apt update && apt upgrade -y

RUN pip3 install --upgrade pip
RUN pip3 install mlflow

# Clean apt 
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /mlflow
CMD mlflow server -h 0.0.0.0 -w 2