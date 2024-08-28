import json  
import os  
from concurrent.futures import ThreadPoolExecutor, as_completed  
from datasets import load_dataset, Dataset  
import pandas as pd  
from tqdm import tqdm  
from revise_text import process_row, system_template, user_template
from utils import read_jsonl
from gpt4o import Openai, API_INFOS
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
def retry_missing_values(not_revised_indices, client, export_data, user_template, system_template, max_tokens=2048, output_file="output_retry.jsonl"):  
    revised_data = []  
      
    with ThreadPoolExecutor(max_workers=len(client)) as executor:  
        futures = [  
            executor.submit(  
                process_row,  
                i,  
                client[i % len(client)],  
                export_data[i],  
                user_template,  
                system_template,  
                max_tokens,  
                output_file  
            )   
            for i in not_revised_indices  
        ]  
          
        for future in tqdm(as_completed(futures), total=len(futures)):  
            revised_data.append(future.result())  
      
    return revised_data  
def retry_merge_missing_data():
    revised_data_path = os.path.join(CUR_DIR, "revised_data/output_sorted.jsonl")
    revised_data = read_jsonl(revised_data_path)
    # not_revised_indices = [i for i, item in enumerate(revised_data) if item['revised_text'] == '']
    not_revised_indices = [item['index'] for item in revised_data if item['gpt_answer'] == '']
    # Retry processing for missing values  
    clients = [Openai(apis=[API_INFOS[i]]) for i in range(len(API_INFOS))] 
    max_tokens = 2048
    output_file = "revised_data/output_retry.jsonl"  
    output_file = os.path.join(CUR_DIR, output_file)
    retry_data = retry_missing_values(not_revised_indices, clients, revised_data, user_template, system_template, max_tokens, output_file=output_file)  
if __name__ == "__main__":
    retry_merge_missing_data()