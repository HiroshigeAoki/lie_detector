#!/bin/bash

thalys_venv/bin/python tapt.py \
    --gpus [0,1] \
    --data_dir_name nested \
    --block_size 128 \
    --batch_size 32