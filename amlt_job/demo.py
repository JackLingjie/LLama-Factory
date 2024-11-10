import gradio as gr
import random
import transformers
import torch

# Load the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained("./sft_12k_5k/checkpoint-5000/", trust_remote_code=True)
#model = transformers.AutoModelForCausalLM.from_pretrained("./glan2_1108/checkpoint-2500/", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation = "flash_attention_2")
model1 = transformers.AutoModelForCausalLM.from_pretrained("./glan1.5_dpo/checkpoint-400/", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation = "flash_attention_2")
model2 = transformers.AutoModelForCausalLM.from_pretrained("./sft_12k_5k/checkpoint-3500/", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation = "flash_attention_2")
model1.to("cuda:0")
model1.to("cuda:1")

def convert_history_to_string(history):
    # [{'role': 'user', 'metadata': {'title': None}, 'content': 'hi'}, {'role': 'assistant', 'metadata': {'title': None}, 'content': "I'm here to help with any questions you have, but I'll need to know more about the specific context and type of question you're asking. Can you please provide more details?"}]
    # f'Human: {prompt}\nAssistant:{assistant_response}\nHuman: {prompt}\nAssistant:{assistant_response}\n'
    history_string = ""
    for i in range(len(history)):
        if history[i]['role'] == 'user':
            history_string += f"Human: {history[i]['content']}\n"
        else:
            history_string += f"Assistant:{history[i]['content']}\n"
    return history_string

def generate_response(prompt, history, model, has_system=True):    
    history_string = ""
    if len(history) > 0:
        history_string = convert_history_to_string(history)
    if has_system:        
        prompt = f"System: You are an AI assistant that provides helpful responses to user queries, developed by MSRA GenAI group.For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n{history_string}Human: {prompt}\nAssistant:"  
    else:        
        prompt = f'Human: {prompt}\nAssistant:'    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(prompt)
    print(inputs)    
    tokens = model.generate(        
        **inputs,
        max_new_tokens=512,        
        temperature=0.7,        
        top_p=0.95,        
        do_sample=True,    
    )    
    return tokenizer.decode(tokens[0], skip_special_tokens=True).replace(prompt, '')


# Define different response models
def random_response_model_1(message, history):
    print(message)
    print(history)
    return generate_response(message, model1)

def random_response_model_2(message, history):
    print(message)
    print(history)
    return generate_response(message, model2)

# Main function that selects the appropriate model based on user choice
def model_selector(message, history, model_choice):
    if model_choice == "glan1.5_dpo_400":
        return random_response_model_1(message, history)
    elif model_choice == "glan1.5":
        return random_response_model_2(message, history)

# Gradio interface setup
with gr.Blocks() as demo:
    # Title and dropdown to select model
    model_choice = gr.Dropdown(choices=["glan1.5_dpo_400", "glan1.5"], label="Choose a Model")

    # Chat interface setup
    chatbot = gr.ChatInterface(
        lambda message, history: model_selector(message, history, model_choice.value),
        type="messages"
    )

    # Display dropdown above chatbot
    model_choice.change(fn=lambda choice: choice, inputs=[model_choice], outputs=[])

# Launch the app
demo.launch()
