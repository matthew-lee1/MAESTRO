# MAESTRO

Unofficial, anonymous repository for MAESTRO: MAsked Encoding Set TRansformer with Self-DistillaiOn. The purpose of this code is for validation of the method, an official deanonymized repository will be published after decision.

## Overview

MAESTRO is a novel approach to self-supervised set-based representation learning using masked encoding with self-distillation. It is designed to process cytometry data efficiently while maintaining set-invariant properties.

## Directory Structure

```
MAESTRO/
├── run.sh              # Bash script for running job on GPU
├── run.py              # Main execution code
│
├── configs/
│   └── config.py       # Configurations for DeepSpeed
│
├── data/
│   └── CyTOFDataset.py # Data class for loading in cytometry data
│
└── models/
    ├── MAESTRO.py      # Code for MAESTRO model
    ├── DeepSets.py     # DeepSets model for benchmark
    └── SetTransformer.py # Set Transformer model for benchmark
```

## External References

- DeepSets implementation based on: https://github.com/manzilzaheer/DeepSets
- Set Transformer implementation based on: https://github.com/juho-lee/set_transformer

## License

To be added after deanonymization.
