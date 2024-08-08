#!bin/bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds0.yaml

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds2.yaml

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml

llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

llamafactory-cli train examples/train_lora/tulu_llama3_lora_sft.yaml

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/tulu_lora_sft_ds2.yaml

# dpo train
llamafactory-cli train examples/train_lora/tulu_lora_dpo.yaml

torchrun --nproc_per_node=4 src/train.py bash_script/tulu_lora_dpo.yaml

#job dpo
torchrun --nproc_per_node=8 src/train.py bash_script/tulu_lora_dpo_job.yaml

llamafactory-cli train bash_script/tulu_lora_dpo_job.yaml


accelerate launch --config_file bash_script/config/fsdp_config.yaml src/train.py bash_script/tulu_lora_dpo.yaml