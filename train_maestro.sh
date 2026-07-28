#!/bin/bash
# Adjust --devices, --data_dirs, and other hyperparameters for your setup.
# If multi-gpu, can switch --devices to '0,1,2,3'
# If using multiple datasets, can add additional directories to --data_dirs, overlapping markers are used by default
#
# The values below are sized for the small demo dataset in data/h5 (~5,000 cells per
# sample) and reproduce the bundled output/training/ToyModel checkpoint.
#
# The manuscript configuration (Methods 3.3) is larger and needs a CUDA GPU:
#   --accelerator cuda --devices '0,1,2,3' --subset_size 100000 \
#   --number_cells_subset 40000 --num_outputs 40000 --dim_hidden 384 --epochs 500
#
# --accelerator cuda (default) uses DeepSpeed + bf16; use cpu or mps to run without a GPU.

python -u train_maestro.py \
    --project 'MAESTRO_Demo' \
    --accelerator 'mps' \
    --data_dirs ./data/h5/dataA \
    --subset_size 5000 \
    --number_cells_subset 4000 \
    --num_outputs 4000 \
    --num_inds 16 \
    --dim_hidden 128 \
    --dim_latent 256 \
    --num_heads 1 \
    --epochs 20 \
    --sinkhorn_start 15 \
    --mode 'Train' \
    --student_temperature 0.11 \
    --teacher_temperature 0.04 \
    --center_momentum 0.9 \
    --teacher_beta 0.99
