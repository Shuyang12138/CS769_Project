import logging
import os
import sys
import pdb
import subprocess

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import BertModel
from utils_ner import NerDataset, Split, get_labels

unlabel_ratio = 4/9
label_ratio=1-unlabel_ratio
train_ratio = 0.8*label_ratio
dev_ratio = 0.1*label_ratio
test_ratio = 0.1*label_ratio

unlabel_sentence = round(total_sentence*unlabel_ratio)
train_sentence = round(total_sentence*train_ratio)
dev_sentence = round(total_sentence*dev_ratio)
test_sentence = round(total_sentence*test_ratio)

sentences = []
current_sentence = ""
unlabel_train = []
unlabel_test = []
label_train = []
label_dev = []
label_test = []


all_dataset = []
total_sentence = 0
with open("../datasets/NER/NCBI-disease/train_dev.txt","r") as file:
    for line in file:
        if line=="\n":
            total_sentence+=1
        all_dataset.append(line)
file.close()
with open("../datasets/NER/NCBI-disease/test.txt","r") as file:
    for line in file:
        if line=="\n":
            total_sentence+=1
        all_dataset.append(line)
file.close()

count = 0
for line in all_dataset:
    if count<=unlabel_sentence:
        unlabel_test.append(line)
        if line=="\n":
            count+=1
            sentences.append(current_sentence[:-1])
            unlabel_train.append(line)
            current_sentence=""
        else:
            current_sentence+=line.split(" ")[0]+" "
            unlabel_train.append(line.split(" ")[0]+"\n")
    elif count <= unlabel_sentence+train_sentence:
        if line=="\n":
            count+=1
        label_train.append(line)
    elif count <= unlabel_sentence+train_sentence+dev_sentence:
        if line=="\n":
            count+=1
        label_dev.append(line)
    else:
        if line=="\n":
            count+=1
        label_test.append(line)
        
        
with open('../datasets/NER/NCBI-disease/CP_sentences.txt', 'w') as file:
    for item in sentences:
        file.write(item+"\n")
file.close()
with open('../datasets/NER/NCBI-disease/labeled_part/unlabel.txt', 'w') as file:
    for item in unlabel_test:
        file.write(item)
file.close()
with open('../datasets/NER/NCBI-disease/labeled_part/train.txt', 'w') as file:
    for item in label_train:
        file.write(item)
file.close()
with open('../datasets/NER/NCBI-disease/labeled_part/devel.txt', 'w') as file:
    for item in label_dev:
        file.write(item)
file.close()
with open('../datasets/NER/NCBI-disease/labeled_part/test.txt', 'w') as file:
    for item in label_test:
        file.write(item)
file.close()