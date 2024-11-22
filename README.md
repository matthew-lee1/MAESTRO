# MAESTRO

Unofficial, anonymous repositor for MAESTRO: MAsked Encoding Set TRansformer with Self-DistillaiOn. The purpose of this code is for validation of the method, an official deanonymized repository will be published after decision. The repository is organized as follows: 

/MAESTRO
run.sh : bash script for running job on GPU 
run.py : run code
  /configs
  config.py : configurations for DeepSpeed 

  /data
  CyTOFDataset.py : data class for loading in cytometry data 
  
  /models
  MAESTRO.py : code for MAESTRO model
  DeepSets.py : code for DeepSets model for benchmark (https://github.com/manzilzaheer/DeepSets)
  SetTransformer.py : code for Set Transformer model for benchmark (https://github.com/juho-lee/set_transformer)

