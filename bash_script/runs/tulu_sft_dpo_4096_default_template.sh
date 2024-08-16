#!/bin/bash
set -x

NUM_PROC=8
SFT_CONFIG="bash_script/tulu_lora_sft_4096_default_template_job.yaml"
DPO_CONFIG="bash_script/tulu_lora_dpo_4096_default_template_job.yaml"

echo "Runing sft stage"
torchrun --nproc_per_node=$NUM_PROC src/train.py $SFT_CONFIG
# torchrun --nproc_per_node=8 src/train.py "bash_script/tulu_lora_sft_4096_default_template_job.yaml"

echo "sleep 60"
sleep 60

echo "Runing dpo stage"
torchrun --nproc_per_node=$NUM_PROC src/train.py $DPO_CONFIG
torchrun --nproc_per_node=8 src/train.py "bash_script/tulu_lora_dpo_4096_default_template_job.yaml"
