# RaspberryHi!

A smarthome hub-esque device that uses facial and audio recognition models to emulate modern smart home systems, all deployed on a Raspberry Pi 5 with an integrated camera and microphone. 

Many current smarthome systems require purchasing various products in order to build a fully functional smart home ecosystem. This means (on top of being expensive) many devices may not function without a supplementary device that solves the dependencies of these ecosystems. RaspberryHi!, or Berry for short, looks to act as an open source solution by allowing you to integrate various smart home applicances through public APIs or directly connecting to the devices.

## Hardware used:
- Raspberry Pi 5 with 8GB of RAM and 128 GB of storage
- Ubuntu 24.10 OS

## To Do:
- [X] Write training code using Sequential API for branches between wake word and user commands
- [ ] Develop lightweight audio processing model using quantization with bias using custom datasets
- [ ] Write training code for facial recognition using a One-Shot learning through a Siamese network
- [ ] Develop lightweight facial recognition model using a MobileNet and train to recognize me specifically.
- [ ] Chain models together to correctly recognize me and allow control of my smart appliances.
- [ ] Integrate control of my smart home lighting using Philips Hue REST API.
- [ ] Train to recognize wake word "Berry"

## Secondary Tasks:
- [ ] Build a home for Berry!

Documentation on how to develop/build this on your own will be included in the repository.
