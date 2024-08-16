#!/bin/bash
set -x

NUM_PROC=4
SFT_CONFIG="bash_script/tulu_lora_sft_ds2_local_test.yaml"
DPO_CONFIG="bash_script/tulu_lora_dpo.yaml"


echo "Runing sft stage"
torchrun --nproc_per_node=$NUM_PROC src/train.py $SFT_CONFIG

echo "sleep 60"
sleep 60

echo "Runing dpo stage"
torchrun --nproc_per_node=$NUM_PROC src/train.py $DPO_CONFIG