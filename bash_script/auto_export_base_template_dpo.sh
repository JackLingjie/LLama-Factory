#!/bin/bash  

set -x

# 定义模型名称的数组  
model_names=(
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v9"
    "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v10"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v7"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v4"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v5"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v6"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v8"
    # "ta_llama3_instruct_dpo_list_bsz1_trible_debug_v1"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v2"
    # "tulu_v2_8b_default_template_dpo_list_bsz1_trible_debug_v3"
    # "ta_v2_rejected_noneed_length_tuluv2_dpo_2048_default_template_bsz1_acc8_v2"
    # "ta_v2_rejected_noneed_tuluv2_dpo_2048_default_template_bsz1_acc8_v2"
    # "ta_v2_chosen_tuluv2_dpo_2048_default_template_bsz1_acc8_v2"
    # "ta_v2_chosen_tuluv2_dpo_2048_default_template_bsz1_acc8_v5"
    # "ta_v2_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8_v5"
    # "ta_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8_v4" 
    # "ta_v2_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8" 
    # "ta_v2_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8_v2"
    # "ta_v2_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8_v3"
    # "ta_v2_rejected_tuluv2_dpo_2048_default_template_bsz1_acc8_v4"
)  # 替换成你的模型名称列表  
  
# 定义YAML文件路径  
yaml_file="bash_script/export_model/merge_default_template_dpo.yaml"  
  
# 临时文件路径，用于保存修改后的 YAML 文件  
# temp_yaml_file="bash_script/export_model/temp_auto_merge_dpo.yaml"  
  
# 循环处理每个模型名称  
for model_name in "${model_names[@]}"; do  
    temp_yaml_file="bash_script/export_model/temp_auto_merge_dpo_${model_name}.yaml"  
    # 替换 YAML 文件中的占位符，并写入临时 YAML 文件  
    echo $model_name
    sed "s|{{model_name}}|$model_name|g" "$yaml_file" > "$temp_yaml_file"  
  
    # 执行命令  
    llamafactory-cli export "$temp_yaml_file"  
done  

echo "Finish all merge jobs"
# 删除临时文件  
# rm "$temp_yaml_file"  
