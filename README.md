# MAESTRO

Unofficial, anonymous repositor for MAESTRO: MAsked Encoding Set TRansformer with Self-DistillaiOn. The purpose of this code is for validation of the method, an official deanonymized repository will be published after decision. The repository is organized as follows: 

/MAESTRO\n
run.sh : bash script for running job on GPU\n
run.py : run code\n
\t/configs\n
\tconfig.py : configurations for DeepSpeed\n
\n
\t/data\n
\tCyTOFDataset.py : data class for loading in cytometry data\n
\n
\t/models\n
\tMAESTRO.py : code for MAESTRO model\n
\tDeepSets.py : code for DeepSets model for benchmark (https://github.com/manzilzaheer/DeepSets)\n
\tSetTransformer.py : code for Set Transformer model for benchmark (https://github.com/juho-lee/set_transformer)\n

