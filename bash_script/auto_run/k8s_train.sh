#!/bin/bash
set -x  # 开启调试模式，显示执行细节

# 读取传入参数
NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4
CUSTOM_YAML=$5

# 默认 YAML 配置文件（数组格式，方便后续替换）
yaml_files=(
    "bash_script/DeepSeek-V2-Lite-Chat_uf_2048_v2.yaml"
    # "bash_script/think_hybrid_qwen_math_7b_merged_reasoning_1074k_generall_nothink_oasst2_1749k.yaml"
    # "bash_script/bitnet_add_synthetic_openr1_openthought_aime_rej_kodcode_taco_aops_cf_1075k_epoch5_ctr.yaml"
    # "bash_script/am_1.4M_dedup1376k_1_5B_test.yaml"
    # "bash_script/am_qwen_math_7B_1.4M_dedup1376k.yaml"
    # "bash_script/bitnet_add_synthetic_openr1_openthought_aime_rej_kodcode_taco_aops_cf_1075k_epoch5_ctr.yaml"
    # "bash_script/think_hybrid_llama_8b_merged_reasoning_1074k_generall_nothink_oasst2_1749k.yaml"
    # "bash_script/bitnet_add_synthetic_openr1_openthought_aime_rej_kodcode_taco_aops_cf_1075k_epoch5_ctr.yaml"
    # "bash_script/think_hybrid_math_merged_reasoning_1074k_generall_nothink_oasst2_1749k.yaml"
    # "bash_script/think_hybrid_qwen_15b_merged_reasoning_1074k_generall_nothink_oasst2_1749k.yaml"
    # "bash_script/EXAONE_2b_synthetic_openr1_openthought_aime_rej_kodcode_taco_aops_cf_1075k.yaml"
    # "bash_script/bitnet_add_synthetic_openr1_openthought_aime_rej_kodcode_taco_aops_cf_1075k.yaml"
    # "bash_script/qwen_math_7b_openr1_synthetic_openthought_aime_kodcode_aops_taco_cf_dedup_1074k.yaml"
    # "bash_script/ds_think_gen_filter_892k_thinkgen_24k_890k_bsz128_lr1e4.yaml"
    # "bash_script/qwen_math_add_synthetic_openr1_openthought_aime_rej_kodcode_taco_cf_data_937k.yaml"
    # "bash_script/qwen_math_7b_debug.yaml"
    # "bash_script/qwen_math_add_synthetic_openr1_openthought_aime_rej_kodcode_taco_cf_data_937k.yaml"
    # "bash_script/multinode_test.yaml"
    # "bash_script/think_890k_openr1_983k_aime_rej_dedup_583k_bsz128_lr1e4_with_sys.yaml"
)
YAML_FILE="${yaml_files[0]}"  # 默认 YAML 文件

# 如果传入了 YAML 文件，则使用自定义 YAML
if [[ -n "$CUSTOM_YAML" ]]; then
    YAML_FILE="$CUSTOM_YAML"
fi

# 确保所有必须参数都已提供
if [[ -z "$NNODES" || -z "$NODE_RANK" || -z "$MASTER_ADDR" || -z "$MASTER_PORT" ]]; then
    echo "参数缺失: 需要提供 NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT"
    exit 1
fi

# 训练任务所在目录
WORK_DIR="/workspace/LLama-Factory"

echo "启动训练: NODE_RANK=$NODE_RANK, MASTER=$MASTER_ADDR, PORT=$MASTER_PORT, NNODES=$NNODES, YAML_FILE=$YAML_FILE"

# 进入工作目录
cd $WORK_DIR || exit 1

# 配置环境变量
export PATH=$PATH:/root/.local/bin
export NCCL_NET=IB
export FORCE_TORCHRUN=1
export NNODES=$NNODES
export NODE_RANK=$NODE_RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# 执行训练
llamafactory-cli train "$YAML_FILE"

echo "训练完成 (NODE_RANK=$NODE_RANK)，开始执行 run_gpu.py"
python run_gpu.py
