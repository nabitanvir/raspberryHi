# Marvin

A lean smarthome system that uses facial/audio recognition.

Many current smarthome systems require purchasing various products in order to build a fully functional smart home ecosystem. This means (on top of being expensive) many devices may not function without a supplementary device that solves the dependencies of these ecosystems. Marvin aims to act as an open source solution by allowing you to integrate various smart home applicances through public APIs or directly connecting to the devices.

## Current Specifications:
- Raspberry Pi 5
- Ubuntu 24.10

## To Do:
- [X] Implement trigger word recognition using gray-scaled MFCC and a CNN
- [X] Implement facial recognition using a Siamese neural network
- [ ] Create dataset for training SNN
- [ ] Write face detection code using pretrained models and face alignment (MTCNN)
- [X] Integrate control of my smart home lighting using Philips Hue REST API.
- [X] Create dataset of positive/negative examples of trigger word "Marvin"
- [X] Train to recognize wake word "Marvin"
- [ ] Create dataset of positive/negative examples for voice commands
- [ ] Write code for transfer learning using Google Speech Embeddings for command recognition
- [ ] Train to recognize commands "Lights On", "Lights Off", "Status"
- [X] Write data augmentation code for voice recognition dataset(s)
- [ ] Chain all models together

## Secondary Tasks:
- [ ] Build a home for Marvin

Documentation on how to build this on your own will be included soon.
