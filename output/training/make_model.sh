#!/bin/bash
# Consolidate a sharded DeepSpeed checkpoint into a single fp32 weights file.
# zero_to_fp32.py is written by DeepSpeed *inside* the last.ckpt/ directory, so it must
# be invoked by its full path. Run this from the repository root.

python -u output/training/ToyModel/last.ckpt/zero_to_fp32.py \
    output/training/ToyModel/last.ckpt/ \
    output/training/ToyModel/model.ckpt
