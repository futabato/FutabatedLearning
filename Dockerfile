FROM nvcr.io/nvidia/cuda:11.6.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ='Asia/Tokyo'
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y && \
    apt-get install -y wget bzip2 build-essential git git-lfs curl wget  \
    ca-certificates libsndfile1-dev libgl1 software-properties-common vim tmux && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils python3.11-venv \
    build-essential cmake && \
    apt-get -y autoremove && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

WORKDIR /workspace
COPY pyproject.toml poetry.lock /workspace/
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.11
RUN pip install --upgrade pip && pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry lock --no-update && \
    poetry install --no-root

RUN git config --global --add safe.directory /workspace
RUN ln -s /usr/bin/pip3 /usr/local/lib/python3.11/dist-packages/pip && \
    ln -s /usr/bin/pip /usr/local/lib/python3.11/dist-packages/pip && \
    ln -s /usr/bin/python3.11 /usr/bin/python