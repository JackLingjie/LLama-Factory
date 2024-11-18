#!/usr/bin/bash

step=20000
for idx in 0 1 2 3 4 5 6 7; do
    basefolder="bitnet2_2b_13m_e3_bs512lr3e05wd00_fft_nlg_targetonly_fix_first"
    export CUDA_VISIBLE_DEVICES=$idx
    start_idx=$((idx * step))
    end_idx=$((start_idx + step))
    echo "start_idx: $start_idx, end_idx: $end_idx"
    python infer_batch.py \
        --ckpt_folder "$HOME/checkpoints/$basefolder/checkpoint-5043" \
        --data_file "$HOME/Datasets/L3Rel/Validation_test_trunc256_gemma2_prompt.jsonl" \
        --dst_file "$HOME/Datasets/L3Rel/Results/${basefolder}_imps_fbs_e2_preds_part_${idx}.txt" \
        --start_idx $start_idx --end_idx $end_idx &
        # --data_file "$HOME/Datasets/L3Rel/Validation_test_trunc256_gemma2_prompt_shuf10k.jsonl" \
done