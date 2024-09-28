#!/bin/bash  
set -x


# 参数列表  
PARAMS=( 
    "ta_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8"
    "ta_v2_rejected_noneed_length_tuluv2_dpo_2048_default_template_bsz1_acc8_v2"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v15"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v11"
    # "ta_v2_chosen_tuluv2_dpo_2048_default_template_bsz1_acc8_v2"
    # "ta_chosen_tuluv2_dpo_2048_default_template_bsz1_acc8_v2"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v14"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v14-1500"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v11"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v11_1500"
    # "tulu_v2_8b_2048_default_template_sft"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v5_1500"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v10"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v10_1500"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v5"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v9"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v4-1500"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_1500"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v4"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v2"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v7"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug"
#   "ta_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8_v2"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v7_1500"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v6_1500"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v6"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v3"
#   "tulu_v2_8b_2048_default_template_dpo"
#   "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v8_1500"
#   "tulu_2048_default_template_trible_uf_dpo"
#   "tulu_2048_default_template_trible_rejected_ta_dpo"
#   "tulu_2048_default_template_trible_chosen_ta_dpo_1500"
#   "tulu_2048_default_template_trible_chosen_ta_dpo"
#   "tulu_2048_default_template_trible_rejected_ta_dpo_1500"
  # 添加更多参数  
)  

# 原始日志文件名  
LOG_FILE="evaluation_log"  
counter=0

# 遍历模型列表，生成新的文件名  
for model in "${PARAMS[@]}"; do 
    if [ $counter -ge 3 ]; then  
        break  
    fi
    # 拼接模型名到log_file后  
    LOG_FILE="${LOG_FILE}_${model}"  

    counter=$((counter + 1)) 
done  
  
# 最终的文件名 
LOG_FILE_NAME=${LOG_FILE}
LOG_FILE="./eval_logs/default_${LOG_FILE_NAME}.txt" 
LOG_FILE2="/mnt/lingjiejiang/textual_aesthetics/logs/mmlu/default_${LOG_FILE_NAME}.txt"
# 清空日志文件  
> "$LOG_FILE"  
> "$LOG_FILE2"  
  
echo $LOG_FILE  
echo $LOG_FILE2 
# 遍历参数列表  
for PARAM in "${PARAMS[@]}"; do  
    echo "执行参数: $PARAM" | tee -a "$LOG_FILE" "$LOG_FILE2"  
    bash bash_script/mmlu_eval/mmlu_eval_default_template.sh $PARAM | tee -a "$LOG_FILE" "$LOG_FILE2"  
    echo "----------------------------------------" | tee -a "$LOG_FILE" "$LOG_FILE2"  
done  
  
echo "所有任务已完成。" | tee -a "$LOG_FILE" "$LOG_FILE2"  