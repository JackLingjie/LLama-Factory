#!/usr/bin/bash

# azcopy cp "https://msranlpintern.blob.core.windows.net/lingjiejiang/textual_aesthetics/exp/saves/bitnet_ads/bitnet_dv3_train_2.5m_trunc256_ql_default_template_2e5_e5_bsz2k/checkpoint-3000/?sv=2023-01-03&st=2024-11-15T09%3A47%3A56Z&se=2024-11-22T09%3A47%3A00Z&skoid=4ca952ec-968b-41ca-95cb-e9b2a5fe0a02&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2024-11-15T09%3A47%3A56Z&ske=2024-11-22T09%3A47%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=N0JkiKyzWIp9KAgFlZj7orUFw9aebpYrH5kbf8OvceU%3D" "ads-e5" --recursive

azcopy cp "https://msranlpintern.blob.core.windows.net/lingjiejiang/textual_aesthetics/exp/saves/bitnet_ads/bitnet_dv3_train_2.5m_trunc256_ql_default_template_2e5_e5_bsz2k/checkpoint-4000/?sv=2023-01-03&st=2024-11-15T09%3A47%3A56Z&se=2024-11-22T09%3A47%3A00Z&skoid=4ca952ec-968b-41ca-95cb-e9b2a5fe0a02&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2024-11-15T09%3A47%3A56Z&ske=2024-11-22T09%3A47%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=N0JkiKyzWIp9KAgFlZj7orUFw9aebpYrH5kbf8OvceU%3D" "ads-e5" --recursive
azcopy cp "https://msranlpintern.blob.core.windows.net/lingjiejiang/textual_aesthetics/exp/saves/bitnet_ads/bitnet_dv3_train_2.5m_trunc256_ql_default_template_2e5_e5_bsz2k/checkpoint-5000/?sv=2023-01-03&st=2024-11-15T09%3A47%3A56Z&se=2024-11-22T09%3A47%3A00Z&skoid=4ca952ec-968b-41ca-95cb-e9b2a5fe0a02&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2024-11-15T09%3A47%3A56Z&ske=2024-11-22T09%3A47%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=N0JkiKyzWIp9KAgFlZj7orUFw9aebpYrH5kbf8OvceU%3D" "ads-e5" --recursive
azcopy cp "https://msranlpintern.blob.core.windows.net/lingjiejiang/textual_aesthetics/exp/saves/bitnet_ads/bitnet_dv3_train_2.5m_trunc256_ql_default_template_2e5_e5_bsz2k/checkpoint-6000/?sv=2023-01-03&st=2024-11-15T09%3A47%3A56Z&se=2024-11-22T09%3A47%3A00Z&skoid=4ca952ec-968b-41ca-95cb-e9b2a5fe0a02&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2024-11-15T09%3A47%3A56Z&ske=2024-11-22T09%3A47%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=N0JkiKyzWIp9KAgFlZj7orUFw9aebpYrH5kbf8OvceU%3D" "ads-e5" --recursive
azcopy cp "https://msranlpintern.blob.core.windows.net/lingjiejiang/textual_aesthetics/exp/saves/bitnet_ads/bitnet_dv3_train_2.5m_trunc256_ql_default_template_2e5_e5_bsz2k/checkpoint-6300/?sv=2023-01-03&st=2024-11-15T09%3A47%3A56Z&se=2024-11-22T09%3A47%3A00Z&skoid=4ca952ec-968b-41ca-95cb-e9b2a5fe0a02&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2024-11-15T09%3A47%3A56Z&ske=2024-11-22T09%3A47%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=N0JkiKyzWIp9KAgFlZj7orUFw9aebpYrH5kbf8OvceU%3D" "ads-e5" --recursive

step=80000
for idx in 0 1; do
    basefolder="ads-e5"
    export CUDA_VISIBLE_DEVICES=$((idx + 2))
    start_idx=$((idx * step))
    end_idx=$((start_idx + step))
    echo "start_idx: $start_idx, end_idx: $end_idx"
    python inference.py \
        --ckpt_folder "$basefolder/checkpoint-4000" \
        --data_file "/data/users/shaohanh/demo/Validation_test_trunc256_gemma2_prompt.jsonl" \
        --dst_file "${basefolder}_4000_imps_fbs_e2_preds_part_${idx}.txt" \
        --batch_size 64 --start_idx $start_idx --end_idx $end_idx &
done
wait
step=80000
for idx in 0 1; do
    basefolder="ads-e5"
    export CUDA_VISIBLE_DEVICES=$((idx + 2))
    start_idx=$((idx * step))
    end_idx=$((start_idx + step))
    echo "start_idx: $start_idx, end_idx: $end_idx"
    python inference.py \
        --ckpt_folder "$basefolder/checkpoint-5000" \
        --data_file "/data/users/shaohanh/demo/Validation_test_trunc256_gemma2_prompt.jsonl" \
        --dst_file "${basefolder}_5000_imps_fbs_e2_preds_part_${idx}.txt" \
        --batch_size 64 --start_idx $start_idx --end_idx $end_idx &
done
wait
step=80000
for idx in 0 1; do
    basefolder="ads-e5"
    export CUDA_VISIBLE_DEVICES=$((idx + 2))
    start_idx=$((idx * step))
    end_idx=$((start_idx + step))
    echo "start_idx: $start_idx, end_idx: $end_idx"
    python inference.py \
        --ckpt_folder "$basefolder/checkpoint-6000" \
        --data_file "/data/users/shaohanh/demo/Validation_test_trunc256_gemma2_prompt.jsonl" \
        --dst_file "${basefolder}_6000_imps_fbs_e2_preds_part_${idx}.txt" \
        --batch_size 64 --start_idx $start_idx --end_idx $end_idx &
done
wait
step=80000
for idx in 0 1; do
    basefolder="ads-e5"
    export CUDA_VISIBLE_DEVICES=$((idx + 2))
    start_idx=$((idx * step))
    end_idx=$((start_idx + step))
    echo "start_idx: $start_idx, end_idx: $end_idx"
    python inference.py \
        --ckpt_folder "$basefolder/checkpoint-6300" \
        --data_file "/data/users/shaohanh/demo/Validation_test_trunc256_gemma2_prompt.jsonl" \
        --dst_file "${basefolder}_6300_imps_fbs_e2_preds_part_${idx}.txt" \
        --batch_size 64 --start_idx $start_idx --end_idx $end_idx &
done
wait
echo "Done"

echo "Calculating metrics"
echo "python calc.py ads-e5_4000_imps_fbs_e2_preds_part_0.txt,ads-e5_4000_imps_fbs_e2_preds_part_1.txt"
python calc.py ads-e5_4000_imps_fbs_e2_preds_part_1.txt,ads-e5_4000_imps_fbs_e2_preds_part_0.txt
echo "python calc.py ads-e5_5000_imps_fbs_e2_preds_part_0.txt,ads-e5_5000_imps_fbs_e2_preds_part_1.txt"
python calc.py ads-e5_5000_imps_fbs_e2_preds_part_1.txt,ads-e5_5000_imps_fbs_e2_preds_part_0.txt
echo "python calc.py ads-e5_6000_imps_fbs_e2_preds_part_0.txt,ads-e5_6000_imps_fbs_e2_preds_part_1.txt"
python calc.py ads-e5_6000_imps_fbs_e2_preds_part_1.txt,ads-e5_6000_imps_fbs_e2_preds_part_0.txt
echo "python calc.py ads-e5_6300_imps_fbs_e2_preds_part_0.txt,ads-e5_6300_imps_fbs_e2_preds_part_1.txt"
python calc.py ads-e5_6300_imps_fbs_e2_preds_part_1.txt,ads-e5_6300_imps_fbs_e2_preds_part_0.txt
