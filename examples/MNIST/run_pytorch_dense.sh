#!/bin/bash

python3 main.py train-and-test-pytorch-model --accelerator auto --epochs 10 --learning_rate 1e-3 --batch_size 32 --early_stopping --patience 3 --monitor_metric val_loss --deterministic --seed 42 --layer_config 100 --layer_config 10
