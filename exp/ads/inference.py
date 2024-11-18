# -*- coding: utf-8 -*-

import fire
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
import numpy as np

def main(ckpt_folder, data_file, dst_file, batch_size=1, start_idx=0, end_idx=None):
    data = open(data_file).readlines()
    data = [json.loads(x) for x in data]
    if end_idx is not None:
        data = data[start_idx:end_idx]

    tokenizer = AutoTokenizer.from_pretrained(ckpt_folder, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(ckpt_folder, trust_remote_code=True)
    model = model.bfloat16().cuda().eval()

    # wrap the prompt with system info
    for item in data:
        # prompt = f'Human: {prompt}\nAssistant:'
        prompt = item['prompt']
        item['prompt'] = f'Human: {prompt}\nAssistant:'

    res = []
    for batch_idx in tqdm.tqdm(range(0, len(data), batch_size)):
        batch = data[batch_idx:batch_idx+batch_size]
        # inputs = tokenizer([x['prompt'] + 'Answer:' for x in batch], return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = tokenizer([x['prompt'] for x in batch], return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        # yes: 7566, no: 2360
        outputs = model.generate(**inputs, max_new_tokens=4, do_sample=False, return_dict_in_generate=True, output_logits=True)
        sequences = outputs.sequences[:, inputs['input_ids'].shape[1]:]
        preds = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        scores = outputs.logits
        for idx, elem in enumerate(batch):
            # yes_score = scores[0][idx, 7566].item()
            # no_score = scores[0][idx, 2360].item()
            yes_score = scores[2][idx, 7566].item()
            no_score = scores[2][idx, 2360].item()
            elem['completion'] = preds[idx]
            elem['yes_score'] = yes_score
            elem['no_score'] = no_score
            if batch_idx % 100 == 0:
                print('---- yes_score: ', yes_score)
                print('no_score: ', no_score)
                print('pred: ', preds[idx])
        res.append(elem)
    
    res = [json.dumps(x) for x in res]
    with open(dst_file, 'w') as f:
        f.write('\n'.join(res) + '\n')


def cal_softmax(yes_score, no_score):
    scores = [yes_score, no_score]
    scores = np.array(scores)
    scores = np.exp(scores)
    scores = scores / np.sum(scores)
    return scores[0]


def cal_auc(filename):
    data = open(filename).readlines()
    data = [json.loads(x) for x in data]
    labels = []
    preds = []
    for elem in data:
        label = 1.0 if elem['label'].strip() == 'Answer: Yes' else 0.0
        labels.append(label)
        score = cal_softmax(elem['yes_score'], elem['no_score'])
        preds.append(score)
    auc = roc_auc_score(labels, preds)
    print('AUC: ', auc)


if __name__ == '__main__':
    fire.Fire(main)
    # fire.Fire(cal_auc)