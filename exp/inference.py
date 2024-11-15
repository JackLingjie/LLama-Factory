import transformers
import torch
import json
import fire
import tqdm

def generate_response(prompts, tokenizer, model, has_system):
    new_prompts = []
    for prompt in prompts:
        if has_system == 1:
            prompt = f"System: You are an AI assistant that provides helpful responses to user queries, developed by MSRA GenAI group. For politically sensitive questions, security and privacy issues, you will refuse to answer\n\nHuman: {prompt}\nAssistant:"
        else:
            prompt = f'Human: {prompt}\nAssistant:'
        new_prompts.append(prompt)
    print(new_prompts)
    inputs = tokenizer(new_prompts, return_tensors="pt", padding=True, padding_side='left').to(model.device)
    #print(inputs)
    tokens = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )
    outputs = []
    for i in range(len(new_prompts)):
        outputs.append(tokenizer.decode(tokens[i], skip_special_tokens=True).replace(new_prompts[i], ''))
    return outputs


def inference(model_path, output_name, gpu_id, has_system):
    bsz = 64
    # Load the tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation = "flash_attention_2")
    #model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation = "eager")
    model.to("cuda:3")
    model.eval()

    # read alpaca data
    with open("alpaca_eval.json", "r", encoding='utf-8') as f:
        data = json.load(f)
        for i in tqdm.tqdm(range(0, len(data), bsz)):
            batch = data[i:i+bsz]
            prompts = [item["instruction"] for item in batch]
            
            responses = generate_response(prompts, tokenizer, model, has_system)
            for j, item in enumerate(batch):
                item["output"] = responses[j]
                item['generator'] = output_name

    with open(f"{output_name}.json", "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    fire.Fire(inference)
