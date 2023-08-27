#!/bin/bash

python src/federatedlearning/train.py --gpu 0 \
    --num_epochs 200 --lr 0.05 --batch_size 100 --num_workers 20 \
    --num_byzantines 8 --byzantine_type bitflip \
    --rho_ratio 200 --num_trimmed_values 12 --zeno_size 4