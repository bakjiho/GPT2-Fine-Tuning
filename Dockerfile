FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update

COPY requirements.txt .
#RUN apt install git
#RUN apt install git-lfs
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .

RUN python3 try.py
