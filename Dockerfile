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
	&& apt-get clean
RUN pip3 install \
    numpy \
    librosa \
    sounddevice \
    tflite-runtime \
    phue \
    scipy \
    numpy
WORKDIR /usr/src/app
COPY . .
