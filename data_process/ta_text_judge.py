import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed  
import os
import pandas as pd
from gpt4o import Openai, API_INFOS
from datasets import load_dataset, Dataset
from tqdm import tqdm
from utils import read_jsonl
def extract_final_verdict(llm_output):  
    """  
    Extracts the final verdict from the LLM output.  
  
    Parameters:  
    llm_output (str): The output string from the LLM.  
  
    Returns:  
    str: The final verdict in the format [[A>>B]], [[A>B]], [[A=B]], [[B>A]], or [[B>>A]].  
    """  
    # Define the regex pattern to match the final verdict  
    pattern = r'\[\[A>>B\]\]|\[\[A>B\]\]|\[\[A=B\]\]|\[\[B>A\]\]|\[\[B>>A\]\]'  
  
    # Search for the pattern in the LLM output  
    match = re.search(pattern, llm_output)  
  
    if match:  
        return match.group(0)  
    else:  
        return None 
    
def get_judged_answer(client, instruction, answer_1, answer_2, user_template, system_template, max_tokens=2048):  
    # 格式化用户模板，插入指令和完成的文本  
    content = user_template.format(instruction=instruction, answer_1=answer_1, answer_2=answer_2)  
      
    # 从客户端获取响应  
    gpt_answer = client.get_response(content=content, system=system_template, max_tokens=max_tokens)  
      
    if gpt_answer is None:  
        gpt_answer = ""  
    gpt_answer = gpt_answer.strip()  
    
    score = extract_final_verdict(gpt_answer)

    return score, gpt_answer  

def get_judge(client, row, user_template, system_template, max_tokens=2048, output_file="judges.jsonl"):  
    # prompt = row['prompt']  
    # response = row['response']  
    # need_modification, revised_text, gpt_answer = get_revised_text(client, prompt, response, user_template, system_template, max_tokens=max_tokens)  
    # print(f"index {index}")
    prompt = row['prompt'] 
    answer_1 = row['response']
    answer_2 = row['revised_text']
    score, judgment = get_judged_answer(client, prompt, answer_1, answer_2, user_template, system_template, max_tokens=2048) 
    result = row
    result['judge'] = judgment
    result['score'] = score    
    with open(output_file, 'a') as f:  
        f.write(json.dumps(result) + "\n")  
    return result 

def main():  
    clients = [Openai(apis=[API_INFOS[i]]) for i in range(len(API_INFOS))]  
    print(f"clients number: {len(clients)}")
    # Initialize multiple clients  
    revised_data = read_jsonl("revised_data/output_sorted.jsonl")
    sample_data = revised_data.select(range(100))
    # sample_data = export_data # all
    # user_template = "User: {instruction}\nCompletion: {completion}"  
    # system_template = "You are a helpful assistant."  
    max_tokens = 2048  
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # data_path = os.path.join(cur_dir, "revised_data/output_sorted.jsonl")
    output_file = "revised_data/output_judge.jsonl"  
    output_file = os.path.join(cur_dir, output_file)
  
    # Clear the output file before starting  
    if os.path.exists(output_file):  
        os.remove(output_file)  
  
    revised_data = []  
  
    with ThreadPoolExecutor(max_workers=len(clients)) as executor:  
        # Create a future for each row in the dataset  
        futures = [executor.submit(get_judge, i, clients[i % len(clients)], row, user_template, system_template, max_tokens, output_file) for i, row in enumerate(sample_data)]  
  
        # Collect the results as they complete  
        for future in tqdm(as_completed(futures), total=len(futures)):  
            revised_data.append(future.result())  

  
    # Load results from JSONL file and ensure the order is preserved  
    with open(output_file, 'r') as f:  
        revised_data = [json.loads(line) for line in f]  
  
    # Sort by the original index  
    revised_dataset = revised_data.sort(key=lambda x: x['index'])  
  
    # Create a new Dataset  
    revised_dataset = Dataset.from_pandas(pd.DataFrame(revised_data))  
    sorted_output_path = os.path.join(cur_dir, "revised_data/output_sorted_v2.jsonl")
    revised_dataset.to_json(sorted_output_path) 