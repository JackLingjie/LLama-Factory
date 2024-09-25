#!/bin/bash  
set -x


# 参数列表  
PARAMS=( 
    # "uf_llama3.1_instruct_dpo_2048_trible"
    # "uf_llama3.1_instruct_dpo_2048_trible_ta_chosen"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v6"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v9"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v10"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v11"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v15"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v15_1500"
    # "uf_llama3.1_instruct_dpo_2048_trible_ta_rejected"
    "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v3"
    "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v3_1500"    
    # "ta_rejected_llama3.1_instruct_2048_default_template_v2"
    # "ta_rejected_llama3.1_instruct_2048_default_template_v2-500"
    # "ta_chosen_llama3.1_instruct_dpo_2048_v2"
    # "Meta-Llama-3.1-8B-Instruct"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v4"
    # "uf_llama3.1_instruct_dpo_2048_job"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v5"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v1"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v5_1500"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v2-1500"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v1-1500"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v4_1500"
    # "ta_rejected_llama3.1_instruct_2048_default_template_v2-1000"
    # "ta_chosen_llama3.1_instruct_dpo_2048"
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
LOG_FILE="./eval_logs/llama3_${LOG_FILE_NAME}.txt" 
LOG_FILE2="/mnt/lingjiejiang/textual_aesthetics/logs/mmlu/llama3_${LOG_FILE_NAME}.txt"
# 清空日志文件  
> "$LOG_FILE"  
> "$LOG_FILE2"  
  
echo $LOG_FILE  

echo $LOG_FILE2 
# 遍历参数列表  
for PARAM in "${PARAMS[@]}"; do  
    echo "执行参数: $PARAM" | tee -a "$LOG_FILE" "$LOG_FILE2"  
    bash bash_script/mmlu_eval/mmlu_eval_llama3_template.sh $PARAM | tee -a "$LOG_FILE" "$LOG_FILE2"  
    echo "----------------------------------------" | tee -a "$LOG_FILE" "$LOG_FILE2"  
done  
  
echo "所有任务已完成。" | tee -a "$LOG_FILE" "$LOG_FILE2"  