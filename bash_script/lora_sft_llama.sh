#!bin/bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds0.yaml

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds2.yaml

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml

llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

llamafactory-cli train examples/train_lora/tulu_llama3_lora_sft.yaml

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/tulu_lora_sft_ds2.yaml

# dpo train
llamafactory-cli train examples/train_lora/tulu_lora_dpo.yaml