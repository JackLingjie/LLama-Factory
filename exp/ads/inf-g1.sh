#!/usr/bin/bash

# azcopy cp "https://msranlpintern.blob.core.windows.net/lingjiejiang/textual_aesthetics/exp/saves/bitnet_ads/bitnet_dv3_train_2.5m_trunc256_ql_default_template_2e5_e5_bsz2k/checkpoint-3000/?sv=2023-01-03&st=2024-11-15T09%3A47%3A56Z&se=2024-11-22T09%3A47%3A00Z&skoid=4ca952ec-968b-41ca-95cb-e9b2a5fe0a02&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2024-11-15T09%3A47%3A56Z&ske=2024-11-22T09%3A47%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=N0JkiKyzWIp9KAgFlZj7orUFw9aebpYrH5kbf8OvceU%3D" "ads-e5" --recursive


step=160000
for idx in 0; do
    basefolder="ads-e5"
    export CUDA_VISIBLE_DEVICES=3
    start_idx=$((idx * step))
    end_idx=$((start_idx + step))
    echo "start_idx: $start_idx, end_idx: $end_idx"
    python inference.py \
        --ckpt_folder "$HOME/checkpoints/$basefolder/checkpoint-3000" \
        --data_file "/data/users/shaohanh/demo/Validation_test_trunc256_gemma2_prompt.jsonl" \
        --dst_file "${basefolder}_imps_fbs_e2_preds_part_${idx}.txt" \
        --batch_size 128 --start_idx $start_idx --end_idx $end_idx &
        # --data_file "$HOME/Datasets/L3Rel/Validation_test_trunc256_gemma2_prompt_shuf10k.jsonl" \
done