#!/bin/bash
# Example command to run the script: ./run_ood_benchmark_vit.sh "I/A/V/R/S"
DATASETS=${1:-"I/A/V/R/S"}
CUDA_VISIBLE_DEVICES=0 python tda_runner.py     --config configs \
                                                --wandb-log \
                                                --datasets $DATASETS \
                                                --backbone ViT-B/16