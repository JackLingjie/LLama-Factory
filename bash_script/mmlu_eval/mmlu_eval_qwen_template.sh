#!/bin/bash
set -x

DEFAULT_MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
MODEL_NAME=${1:-$DEFAULT_MODEL_NAME}

CUDA_VISIBLE_DEVICES=0 llamafactory-cli eval \
--model_name_or_path /mnt/lingjiejiang/textual_aesthetics/model_checkpoint/sft_merge_checkpoints/${MODEL_NAME} \
--template qwen \
--task mmlu_test \
--lang en \
--n_shot 5 \
--batch_size 16