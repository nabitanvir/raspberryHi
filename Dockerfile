FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
	python3 \
	python3-pip \
	libatlas-base-dev \
	libportaudio2 \
	libasound-dev \
	portaudio19-dev \
	vim \
	iputils-ping \
	libsndfile1 \
	&& apt-get clean
COPY requirements.txt .
RUN pip3 install -r requirements.txt
WORKDIR /usr/src/app
COPY . .
