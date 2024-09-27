# RaspberryHi!

A smarthome hub-esque device that uses facial and audio recognition models to emulate modern smart home systems, all deployed on a Raspberry Pi 5 with an integrated camera and microphone. 

Many current smarthome systems require purchasing various products in order to build a fully functional smart home ecosystem. This means (on top of being expensive) many devices may not function without a supplementary device that solves the dependencies of these ecosystems. RaspberryHi!, or Berry for short, looks to act as an open source solution by allowing you to integrate various smart home applicances through public APIs or directly connecting to the devices.

## Hardware used:
- Raspberry Pi 5 with 8GB of RAM and 128 GB of storage
- Ubuntu 24.10 OS

IMPORTANT:
There are two sets of code in this project for training:
1. A training architecture that uses transfer learning on a pretrained model and introduces bias to your voice at later layers
2. My own NN architecture that trains from scratch
There are tradeoffs depending on which one you use:
1. The first is recommended if you do not have the computational resources and lack a large enough dataset to train a model from scratch. It will be easier to train but will give you less control over architecture.
2. The second is better if have the means to train a model from scratch and have a large enough dataset, this will give you more control and let you fine tune to your own specifications.

## To Do:
- [X] Write training code for wake word and user commands using transfer learning
- [X] Write training code for wake word and user commands using custom NN architecture
- [ ] Develop lightweight audio processing model using quantization with bias using custom datasets
- [ ] Write training code for facial recognition using a One-Shot learning through a Siamese network
- [ ] Develop lightweight facial recognition model using a MobileNet and train to recognize me specifically.
- [ ] Chain models together to correctly recognize me and allow control of my smart appliances.
- [ ] Integrate control of my smart home lighting using Philips Hue REST API.
- [ ] Train to recognize wake word "Berry"
- [ ] Train to recognite commands "Lights On", "Lights Off", "Status"

## Secondary Tasks:
- [ ] Build a home for Berry!

Documentation on how to develop/build this on your own will be included in the repository.
