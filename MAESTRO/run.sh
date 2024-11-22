#!/bin/bash
#BSUB -gpu "num=4"
#BSUB -J MAESTRO
#BSUB -e MAESTRO.%J.err
#BSUB -o MAESTRO.%J.out

module load cuda/12.3
CUDA_VISIBLE_DEVICES=0,1,2,3

python -u run.py \
--project 'MAESTRO' \
--devices '0,1,2,3' \
--data_dir '/home/data/' \
--number_cells_subset 10000 \
--dim_input 30 \
--dim_output 30 \
--num_inds 2048 \
--dim_hidden 2048 \
--dim_latent 1024 \
--num_heads 4 \
--num_outputs 10000 \
--epochs 1000 \
--mode 'Train'