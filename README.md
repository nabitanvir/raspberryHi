# RaspberryHi!

A smarthome hub-esque device that uses facial and audio recognition models to emulate modern smart home systems, all deployed on a Raspberry Pi 5 with an integrated camera and microphone. 

Many current smarthome systems require purchasing various products in order to build a fully functional smart home ecosystem. This means (on top of being expensive) many devices may not function without a supplementary device that solves the dependencies of these ecosystems. RaspberryHi!, or "Berry" for short, looks to act as an open source solution by allowing you to integrate various smart home applicances through public APIs or directly connecting to the devices.

## Hardware used:
- Raspberry Pi 5 with 8GB of RAM and 128 GB of storage
- Ubuntu 24.10 OS

## To Do:
- [X] Implement trigger word recognition using gray-scaled MFCC and a CNN
- [ ] Implement facial recognition using a Siamese network
- [ ] Chain models together to correctly recognize me and allow control of my smart appliances.
- [X] Integrate control of my smart home lighting using Philips Hue REST API.
- [X] Create dataset of positive/negative examples of trigger word "Berry"
- [X] Train to recognize wake word "Berry"
- [ ] Create dataset of positive/negative examples for voice commands
- [ ] Train to recognize commands "Lights On", "Lights Off", "Status"
- [X] Write data augmentation code for voice recognition dataset(s)
- [ ] Train to recognize my face when in frame

## Secondary Tasks:
- [ ] Write code for transfer learning using Google Speech Embeddings for command recognition
- [ ] Build a home for Berry!

Notes:

There are two sets of code in this project for training the wake word and commands, both with tradeoffs:

1. A training architecture that uses transfer learning on a pretrained model and introduces bias to your voice at later layers
  - This is recommended if you do not have the computational resources and lack a large enough dataset to train a model from scratch.
  - It will be easier to train but will give you less control over architecture.
2. My own NN architecture that trains from scratch
  - This approach is better if you have the means to train a model from scratch and have a large enough dataset.
  - This will also give you more control and let you fine tune to your own specifications.

Documentation on how to develop/build this on your own will be included in the GUIDE.txt file.
