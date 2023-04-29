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

set_seed(1)
config = AutoConfig.from_pretrained(
        "dmis-lab/biobert-base-cased-v1.1",
    )
tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
)
model = AutoModelForMaskedLM.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    from_tf=bool(".ckpt" in "dmis-lab/biobert-base-cased-v1.1"),
    config=config,
)
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling,LineByLineTextDataset
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
dataset= LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = '../datasets/NER/NCBI-disease/CP_sentences.txt',
    block_size = 128  # maximum sequence length
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='pretrain/',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    
)
trainer.train()
trainer.save_model('pretrain/CP_model/')
tokenizer.save_pretrained("pretrain/CP_model/")
