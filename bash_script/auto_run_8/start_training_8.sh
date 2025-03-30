#!/bin/bash
set -x  # 开启调试模式，显示命令执行细节

# 定义所有节点
NODES=("node-0" "node-1" "node-2" "node-3" "node-4" "node-5" "node-6" "node-7")

# 训练 YAML 配置文件（假设 YAML 已经在 `/tmp/Qwen-sft/`）
yaml_files=(
    "bash_script/deepseek-llm-7b-chat_uf_2048_v2.yaml"
    # "bash_script/DeepSeek-V2-Lite-Chat_uf_2048_v2.yaml"
    # "bash_script/think_hybrid_qwen_7b_merged_reasoning_1074k_generall_nothink_oasst2_1749k.yaml"
    # bash_script/am_980k_openr1_synthetic_openthought_aime_kodcode_aops_taco_cf_dedup_1074k_2054k_resume.yaml
    # "bash_script/model_1074k_gpqa_v3_correct_10k_env_fix_new.yaml"
    # "bash_script/model_1074k_gpqa_v3_correct_10k.yaml"
    # "bash_script/model_1074k_redstone-mcq-gpqaStyle_v3_ge0.7_10k.yaml"
    # "bash_script/am_980k_openr1_synthetic_openthought_aime_kodcode_aops_taco_cf_dedup_1074k_2054k.yaml"
    # "bash_script/am_1.4M_dedup1376k_1_5B.yaml"
    # "bash_script/model_7B_1074k_glan2_math_key_31k_rej.yaml"
    # "bash_script/model_1074k_glan2_math_key_31k_rej.yaml"
    # "bash_script/model_7b_1074k_redstone_mcq_gpqaStyle_10k_correct_train_processed_dedup_5k.yaml"
    # "bash_script/model_7B_1074k_glan2_math_key_36k_dedup.yaml"
    # "bash_script/bitnet_add_synthetic_openr1_openthought_aime_rej_kodcode_taco_aops_cf_1075k_epoch5_ctr.yaml"
    # "bash_script/think_hybrid_qwen_7b_merged_reasoning_1074k_generall_nothink_oasst2_1749k.yaml"
    # "bash_script/think_hybrid_llama_8b_merged_reasoning_1074k_generall_nothink_oasst2_1749k.yaml"
    # "bash_script/think_hybrid_qwen_7b_merged_reasoning_1074k_generall_nothink_oasst2_1749k.yaml"
    # "bash_script/model_1074k_redstone_mcq_gpqaStyle_10k_correct_train_processed_dedup_5k_epoch6.yaml"
    # "bash_script/model_1074k_gpqa_40k.yaml"
    # "bash_script/model_1074k_gpqa_30k.yaml"
    # "bash_script/model_1074k_gpqa_20k.yaml"
    # "bash_script/model_1074k_gpqa_10k.yaml"
    # "bash_script/model_1074k_gpqa_34k.yaml"
    # "bash_script/model_1074k_gpqa_44k.yaml"
    # "bash_script/model_1074k_gpqa_24k.yaml"
    # "bash_script/model_1074k_gpqa_14k.yaml"
    # "bash_script/model_7B_1074k_redstone-mcq-gpqaStyle_v1_v2_91k.yaml"
    # "bash_script/model_1074k_redstone-mcq-gpqaStyle_v1_v2_91k.yaml"
    # "bash_script/qwen_1.5B_openr1_synthetic_openthought_aime_kodcode_aops_taco_cf_dedup_1074k.yaml"
    # "bash_script/think_gen_base_deepscaler_rej_tinkgen_v2_all_sorted_0_7.yaml"
    # "bash_script/think_gen_base_deepscaler_rej_tinkgen_v2_all_sorted_0_5.yaml"
    # "bash_script/think_gen_base_deepscaler_rej_tinkgen_v2_all_sorted_0_9_epoch10.yaml"
    # "bash_script/think_gen_base_deepscaler_rej_tinkgen_v2_all_sorted_0_9.yaml"
    # "bash_script/model_1074k_glan2_math_key_36k_dedup.yaml"
    # "bash_script/model_1074k_glan2_math_key_39k.yaml"
    # "bash_script/EXAONE_2b_synthetic_openr1_openthought_aime_rej_kodcode_taco_aops_cf_1075k_16k.yaml"
    # "bash_script/EXAONE_2b_synthetic_openr1_openthought_aime_rej_kodcode_taco_aops_cf_1075k.yaml"
    # "bash_script/model_1074k_redstone_mcq_gpqaStyle_10k_correct_train_processed.yaml"
    # "bash_script/model_1074k_redstone_mcq_gpqaStyle_10k_correct_train_processed.yaml"
    # "bash_script/model_1074k_redstone_mcq_gpqaStyle_10k_correct_train_processed_dedup_5k.yaml"
    # "bash_script/aba_openr1_synthetic_openthought_aime_kodcode_aops_taco_cf_filter_output_docs_1018k.yaml"
    # "bash_script/aba_openr1_synthetic_openthought_aime_kodcode_aops_taco_cf_filter_output_1034k.yaml"
    # "bash_script/qwen_math_add_synthetic_openr1_openthought_aime_rej_kodcode_taco_cf_data_937k_lr1e5.yaml"
    # "bash_script/qwen_math_add_synthetic_openr1_openthought_aime_rej_kodcode_taco_cf_data_937k_lr5e5.yaml"
    # "bash_script/qwen_math_synthetic_openr1_openthought_aimerej_kodcode_aosp_instruct_1011k_bsz128_lr1e4_resume.yaml"
    # "bash_script/qwen_math_synthetic_openr1_openthought_aimerej_kodcode_aosp_instruct_1011k_bsz128_lr1e4.yaml"
    # "bash_script/qwen_math_7b_890k_openr1_983k_aime_rej_dedup_kodcode_829k_bsz128_lr1e4.yaml"

)

YAML_FILE="${yaml_files[0]}"
# YAML_FILE="bash_script/qwen_math_7b_890k_openr1_983k_aime_rej_dedup_583k_without_sys_bsz128_lr1e4.yaml"


# 训练任务所在目录
WORK_DIR="/tmp/LLama-Factory"

# 存储所有进程的 PID
PIDS=()

# 启动训练任务
for i in {0..7}; do
    NODE_RANK=$i
    NODE=${NODES[$i]}
    
    CMD="
        cd $WORK_DIR &&  # 进入工作目录
        export PATH=\$PATH:/home/aiscuser/.local/bin;  # 确保 llamafactory-cli 在 PATH 中
        export NCCL_NET=IB;
        export FORCE_TORCHRUN=1;
        export NNODES=8;  # 修改为 8 个节点
        export NODE_RANK=$NODE_RANK;
        export MASTER_ADDR=node-0;
        export MASTER_PORT=12335;
        llamafactory-cli train \"$YAML_FILE\";
        echo \"Training completed on $NODE, running run_gpu.py...\";
        python run_gpu.py  # 训练结束后执行 run_gpu.py
    "

    if [ "$NODE" == "node-0" ]; then
        echo "Starting training on $NODE (local)..."
        eval "$CMD" &  # 在本机后台运行
        PIDS+=($!)  # 记录进程 ID
    else
        echo "Starting training on $NODE (remote via SSH)..."
        ssh "$NODE" "$CMD" &  # 远程 SSH 运行
        PIDS+=($!)  # 记录进程 ID
    fi
done

# 等待所有训练任务和 run_gpu.py 任务完成
for PID in "${PIDS[@]}"; do
    wait "$PID"
done

echo "All training processes and run_gpu.py executions completed!"
