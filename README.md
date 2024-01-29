# FutabatedLearning

This is a federated learning framework

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

```
python3 src/federatedlearning/train.py
```

### Override configuration from the command line

Example

```
python3 src/federatedlearning/train.py \
    mlflow.run_name=exp001 \
    federatedlearning.num_byzantines=8 federatedlearning.byzantine_type=bitflip \
    federatedlearning.aggregation=zeno
```

### Run with multiple different configurations

Example

```
python3 src/federatedlearning/train.py \
    --multirun 'federatedlearning.num_byzantines=range(8,13)'
```

## Visualize, Search, Compare experiments

```
mlflow ui
```