import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed  
import os
import pandas as pd
from gpt4o import Openai, API_INFOS
from datasets import load_dataset, Dataset
from tqdm import tqdm
from utils import read_jsonl

system_template = \
"""
You are an impartial judge tasked with evaluating the textual aesthetics of responses provided by two AI assistants to the user prompt displayed below. Your goal is to determine which response is more aesthetically pleasing and easier to read and understand.  
  
Begin your evaluation by considering the following aspects for each response:  
  
1. **Readability**: Is the text easy to read and understand? Are the sentences of appropriate length and complexity?  
2. **Visual Organization**: Is the text visually organized in a logical manner? Are there appropriate headings, subheadings, lists, and other formatting elements?  
3. **Consistency**: Does the text maintain a consistent style and format throughout?  
4. **Overall Structure**: Are the paragraphs well-structured and logically connected? Is there appropriate spacing between paragraphs?  
  
Follow these steps for your evaluation:  
1. **Analyze each response**: Carefully read and analyze both responses based on the criteria provided.  
2. **Compare both responses**: Determine which response excels in textual aesthetics considering all aspects.  
3. **Make a final decision**: Choose the response that is better in terms of textual aesthetics and justify your choice.  
  
You must output only one of the following choices as your final verdict with a label:  
1. Assistant A is significantly better: [[A>>B]]  
2. Assistant A is slightly better: [[A>B]]  
3. Tie, relatively the same: [[A=B]]  
4. Assistant B is slightly better: [[B>A]]  
5. Assistant B is significantly better: [[B>>A]]  
  
Example output: "My final verdict is Assistant A is slightly better: [[A>B]]."  
"""

## user prompt
user_template  = \
"""
<|User Prompt|>{instruction}
<|The Start of Assistant A's Answer|>
{answer_1}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_2}
<|The End of Assistant B's Answer|>"  
"""

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
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    clients = [Openai(apis=[API_INFOS[i]]) for i in range(len(API_INFOS))]  
    print(f"clients number: {len(clients)}")
    # Initialize multiple clients  
    revised_data = read_jsonl(os.path.join(cur_dir, "revised_data/output_sorted.jsonl"))
    sample_data = revised_data[:10]
    # sample_data = export_data # all
    # user_template = "User: {instruction}\nCompletion: {completion}"  
    # system_template = "You are a helpful assistant."  
    max_tokens = 2048  

    # data_path = os.path.join(cur_dir, "revised_data/output_sorted.jsonl")
    output_file = "revised_data/output_judge.jsonl"  
    output_file = os.path.join(cur_dir, output_file)
  
    # Clear the output file before starting  
    if os.path.exists(output_file):  
        os.remove(output_file)  
  
    revised_data = []  
  
    with ThreadPoolExecutor(max_workers=len(clients)) as executor:  
        # Create a future for each row in the dataset  
        futures = [executor.submit(get_judge, clients[i % len(clients)], row, user_template, system_template, max_tokens, output_file) for i, row in enumerate(sample_data)]  
  
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
    sorted_output_path = os.path.join(cur_dir, "revised_data/output_judge_sorted.jsonl")
    revised_dataset.to_json(sorted_output_path) 

if __name__ == '__main__':
    main()