# FutabatedLearning

This is a federated learning framework for research (not practical).

This implementation was developed by forking from [Federated-Learning (PyTorch)](https://github.com/AshwinRJ/Federated-Learning-PyTorch), a vanilla implementation of the paper "[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)".

## Setup

### Dev Containers

1. Open Command Palette
2. Select `Dev Containers: Reopen in Container`

### Docker Image Build

```
docker image build -t futabated-learning .
```

### Docker Container Run

```
docker container run --gpus all --rm -it -p 5000:5000 -e PYTHONPATH=/workspace/src/ -v ${PWD}:/workspace futabated-learning /bin/bash
```

## Run an experiment

The baseline experiment with MNIST on CNN model using GPU (if `gpu:0` is available)

```
python src/federatedlearning/main.py
```
