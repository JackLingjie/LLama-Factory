# -*- coding: utf-8 -*-

import fire
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
import numpy as np

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
    fire.Fire(cal_auc)