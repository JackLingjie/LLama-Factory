CUDA_VISIBLE_DEVICES=0 llamafactory-cli eval \
--model_name_or_path /mnt/lingjiejiang/textual_aesthetics/model_checkpoint/sft_merge_checkpoints/Meta-Llama-3.1-8B-Instruct \
--template llama3 \
--task mmlu_test \
--lang en \
--n_shot 5 \
--batch_size 16