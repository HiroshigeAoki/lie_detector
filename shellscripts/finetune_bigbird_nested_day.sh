#!/bin/bash

venv/bin/python main.py \
    model=hf_bigbird \
    tokenizer=hf_bigbird \
    data=exclude_bbs_nested_day \
    trainer.gpus=[4,5] \
    model.data_module.batch_size=32 \
    optim=AdamW