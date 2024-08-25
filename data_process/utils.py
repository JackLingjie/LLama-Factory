import json  
  
def read_jsonl(file_path):  
    """  
    读取 JSONL 文件中的数据并返回一个包含所有记录的字典列表。  
  
    参数:  
    file_path (str): JSONL 文件的路径。  
  
    返回:  
    list: 包含所有记录的字典列表。  
    """  
    data = []  
  
    with open(file_path, 'r', encoding='utf-8') as file:  
        for line in file:  
            # 解析每一行的 JSON 对象并添加到列表中  
            data.append(json.loads(line.strip()))  
  
    return data  
  
# 示例用法  
if __name__ == "__main__":  
    file_path = "your_file.jsonl"  
    records = read_jsonl(file_path)  
    print(records)  
