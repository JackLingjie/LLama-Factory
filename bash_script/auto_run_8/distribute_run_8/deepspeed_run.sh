#!/bin/bash  

export MASTER_ADDR=$MASTER_ADDR  
export MASTER_PORT=$MASTER_PORT 
export WANDB_API_KEY=dcd2671f5f4745b21bc016f367bbb64e7b3820b5

deepspeed --num_gpus 8 --num_nodes 4 --hostfile bash_script/distribute_run/node_config/hostfile \
    src/train.py \
    --model_name_or_path /mnt/lingjiejiang/reason/checkpoints/Qwen2.5-Math-1.5B \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --flash_attn fa2 \
    --dataset open_thought_bespoke_openr1_294k_16klength \
    --max_samples 10000 \
    --template default \
    --cutoff_len 16384 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir /mnt/lingjiejiang/reason/exp/qwen2.5_math1.5b/full_16k_promptv1_4nodes \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5.0e-6 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000 \
    --val_size 0.1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 500  \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --report_to tensorboard \
    # --run_name debug # optional