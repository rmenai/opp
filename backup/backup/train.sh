#!/usr/bin/env bash

python scripts/train.py \
    --data_path "audio/dataset/keys_1749394837_117.npz" \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.0005 \
    --patience 10 \
    --seed 123
