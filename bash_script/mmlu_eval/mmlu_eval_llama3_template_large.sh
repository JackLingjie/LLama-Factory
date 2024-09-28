#!/bin/bash
set -x

DEFAULT_MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
MODEL_NAME=${1:-$DEFAULT_MODEL_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 llamafactory-cli eval \
--model_name_or_path /mnt/lingjiejiang/textual_aesthetics/model_checkpoint/sft_merge_checkpoints/${MODEL_NAME} \
--template llama3 \
--task mmlu_test \
--lang en \
--n_shot 5 \
--batch_size 16