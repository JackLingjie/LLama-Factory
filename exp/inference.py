import transformers
import torch
import json
import fire
import tqdm

def generate_response(prompt, tokenizer, model, has_system):
    if has_system == 1:
        prompt = f"System: You are an AI assistant that provides helpful responses to user queries, developed by MSRA GenAI group. For politically sensitive questions, security and privacy issues, you will refuse to answer\nHuman: {prompt}\nAssistant:"
    else:
        prompt = f'Human: {prompt}\nAssistant:'
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(prompt)
    tokens = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )
    return tokenizer.decode(tokens[0], skip_special_tokens=True).replace(prompt, '')

def inference(model_path, output_name, gpu_id, has_system):

    # Load the tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained("./sft_12k_5k/checkpoint-5000/", trust_remote_code=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation = "flash_attention_2")
    model.to(f"cuda:{gpu_id}")
    model.eval()

    # read alpaca data
    with open("alpaca_eval.json", "r", encoding='utf-8') as f:
        data = json.load(f)
        for item in tqdm.tqdm(data):
            prompt = item["instruction"]
            response = generate_response(prompt, tokenizer, model, has_system)
            item["output"] = response    
            item['generator'] = output_name
    with open(f"{output_name}.json", "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    fire.Fire(inference)