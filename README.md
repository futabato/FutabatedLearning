# Yet-Another Zeno

This is yet another PyTorch implementation of the paper "[Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance](https://proceedings.mlr.press/v97/xie19b.html)"

Official Implementation (mxnet): <https://github.com/xcgoner/icml2019_zeno>

## Setup

### Docker Image Build

```
docker image build -t yet-another-zeno .
```

### Docker Container Run

```
docker container run --gpus all --rm -it -v ${PWD}:/workspace yet-another-zeno bash
```
