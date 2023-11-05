# Yet-Another Zeno

This is yet another PyTorch implementation of the paper "[Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance](https://proceedings.mlr.press/v97/xie19b.html)"

Official Implementation (mxnet): <https://github.com/xcgoner/icml2019_zeno>

## Setup

### Dev Containers

1. Open Command Palette
2. Select `Dev Containers: Reopen in Container`

### Docker Image Build

```
docker image build -t yet-another-zeno .
```

### Docker Container Run

```
docker container run --gpus all --rm -it -p 5000:5000 -e PYTHONPATH=/workspace/src/ -v ${PWD}:/workspace yet-another-zeno /bin/bash
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