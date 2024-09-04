import json  
  
def convert_json_to_jsonl(input_file, output_file):  
    # 读取JSON文件  
    with open(input_file, 'r', encoding='utf-8') as f:  
        data = json.load(f)  
      
    # 写入JSONL文件  
    with open(output_file, 'w', encoding='utf-8') as f:  
        for item in data:  
            json_line = json.dumps(item)  
            f.write(json_line + '\n')  
  
# 使用示例  
input_file = "/mnt/lingjiejiang/textual_aesthetics/data/glanv2_glanchatv2.json"  # 输入的JSON文件名  
output_file = '/mnt/lingjiejiang/textual_aesthetics/data/glanv2_glanchatv2.jsonl'  # 输出的JSONL文件名  
  
convert_json_to_jsonl(input_file, output_file)  
