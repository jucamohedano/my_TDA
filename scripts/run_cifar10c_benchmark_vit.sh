#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tda_runner.py     --config configs \
                                                --wandb-log \
                                                --datasets C \
                                                --backbone ViT-B/16 