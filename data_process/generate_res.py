from text2img import text_to_image  
from utils import read_jsonl  
import os
import concurrent.futures  
  
def process_item(item):  
    index = item["index"]  
    output_img_original = f"output_original_{index}.png"  
    text_to_image(item["response"], output_img_original, save_dir="original_response", temp_dir="original_temp")  
    output_img = f"output_{index}.png"  
    text_to_image(item["gpt_answer"], output_img, save_dir="revised_response", temp_dir="revised_temp")  

def generate_res():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cur_dir, "revised_data/output_sorted.jsonl")
    revised_data = read_jsonl(data_path)  
    
    # 指定线程数，例如 4  
    max_workers = 20  
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  
        futures = [executor.submit(process_item, item) for item in revised_data]  
        for future in concurrent.futures.as_completed(futures):  
            try:  
                future.result()  
            except Exception as exc:  
                print(f'Generated an exception: {exc}')  

if __name__ == "__main__":
    generate_res()