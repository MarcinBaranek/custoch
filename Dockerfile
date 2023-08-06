FROM nvidia/cuda:12.2.0-base-ubuntu22.04

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip

RUN python3.10 -m pip install --upgrade pip
COPY requirements.txt /homoe/requirements.txt
RUN python3.10 -m pip install -r /home/requirements.txt

WORKDIR /home
COPY custoch /custoch
COPY examples /home/examples
COPY tests /home/tests
# do some stuff
