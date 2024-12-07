#!/bin/bash
# Example command to run the script: ./run_ood_benchmark_vit_waiting.sh "C"
DATASETS=${1:-"I/A/V/R/S"}
CUDA_VISIBLE_DEVICES=0 python tda_runner_with_waiting.py    --config configs \
                                                            --wandb-log \
                                                            --datasets $DATASETS \
                                                            --use-waiting-list \
                                                            --backbone ViT-B/16