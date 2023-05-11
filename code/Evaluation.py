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
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    modeling_outputs,
    integrations
)
from transformers import BertModel
from utils_ner import NerDataset, Split, get_labels
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
set_seed(1)
logger = logging.getLogger(__name__)
def collate_fn(batch):
    input_ids = torch.tensor([item.input_ids for item in batch])
    attention_mask = torch.tensor([item.attention_mask for item in batch])
    token_type_ids = torch.tensor([item.token_type_ids for item in batch])
    label_ids = torch.tensor([item.label_ids for item in batch])
    return input_ids, attention_mask, token_type_ids,label_ids
def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
    #preds = np.argmax(predictions, axis=2)
    preds=predictions
    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
        
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list

def compute_metrics(y_pred,y_true) -> Dict:
    preds_list, out_label_list = align_predictions(y_pred, y_true)
        
    return {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
def evaluation(model_path,data_path,split_mode,data_dir,file_name):
    
    labels = get_labels(data_path+"labels.txt")
    num_labels = len(labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
      # download model & vocab.  # download model & vocab.

    config = AutoConfig.from_pretrained(
            data_dir,
            num_labels=num_labels,
            id2label=label_map,
            label2id={label: i for i, label in enumerate(labels)},
            output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
    
    test_dataset = NerDataset(
            data_dir=data_path,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=192,
            overwrite_cache=True,
            mode=split_mode,
        )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    device=torch.device("cuda")
    
    model=AutoModelForTokenClassification.from_pretrained(data_dir)
    model.to(device)
    model.eval()
    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
    #preds = np.argmax(predictions, axis=2)
        preds=predictions
        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(y_pred,y_true) -> Dict:
        preds_list, out_label_list = align_predictions(y_pred, y_true)
        
        return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
# Step 4: Iterate over test data and get predicted tags
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(test_loader,desc="Training: ",dynamic_ncols=True):
            input_ids, attention_mask, token_type_ids,label_ids = batch
            input_ids=input_ids.to(device)
            attention_mask=attention_mask.to(device)
            label_ids=label_ids.to(device)
            token_type_ids=token_type_ids.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,labels=label_ids,token_type_ids=token_type_ids).logits
            _, predicted = torch.max(outputs, 2)
            y_true.extend(label_ids.tolist())
            y_pred.extend(predicted.tolist())

# Step 5: Calculate evaluation metrics
#print(classification_report(y_true, y_pred))
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)

#predictions, label_ids, metrics = trainer.predict(test_dataset)
    preds_list, out_label = align_predictions(y_pred, y_true)
    metrics = compute_metrics(y_pred,y_true)

    with open(data_dir+file_name, "w") as writer:
        logger.info("***** Test results *****")
        for key, value in metrics.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))
#evaluation("dmis-lab/biobert-base-cased-v1.1","../datasets/NER/NCBI-disease/labeled_part/",Split.unlabel,"output_FT/NCBI-disease/labeled_part","/unlabel_result.txt")
#evaluation("pretrain/CP_model","../datasets/NER/NCBI-disease/labeled_part/",Split.unlabel,"output_CP/NCBI-disease/labeled_part","/unlabel_result.txt")
#evaluation("dmis-lab/biobert-base-cased-v1.1","../datasets/NER/NCBI-disease/labeled_part/",Split.unlabel,"output_SL/","unlabel_result.txt")
#evaluation("pretrain/CP_model","../datasets/NER/NCBI-disease/labeled_part/",Split.unlabel,"output_CP_SL/","unlabel_result.txt")

#evaluation("dmis-lab/biobert-base-cased-v1.1","../datasets/NER/NCBI-disease/labeled_part/",Split.charswap,"output_FT/NCBI-disease/labeled_part","/charswap_result.txt")
#evaluation("pretrain/CP_model","../datasets/NER/NCBI-disease/labeled_part/",Split.charswap,"output_CP/NCBI-disease/labeled_part","/charswap_result.txt")
#evaluation("dmis-lab/biobert-base-cased-v1.1","../datasets/NER/NCBI-disease/labeled_part/",Split.charswap,"output_SL/","charswap_result.txt")
#evaluation("pretrain/CP_model","../datasets/NER/NCBI-disease/labeled_part/",Split.charswap,"output_CP_SL/","charswap_result.txt")

#evaluation("dmis-lab/biobert-base-cased-v1.1","../datasets/NER/NCBI-disease/labeled_part/",Split.charkey,"output_FT/NCBI-disease/labeled_part","/charkey_result.txt")
#evaluation("pretrain/CP_model","../datasets/NER/NCBI-disease/labeled_part/",Split.charkey,"output_CP/NCBI-disease/labeled_part","/charkey_result.txt")
#evaluation("dmis-lab/biobert-base-cased-v1.1","../datasets/NER/NCBI-disease/labeled_part/",Split.charkey,"output_SL/","charkey_result.txt")
#evaluation("pretrain/CP_model","../datasets/NER/NCBI-disease/labeled_part/",Split.charkey,"output_CP_SL/","charkey_result.txt")

evaluation("dmis-lab/biobert-base-cased-v1.1","../datasets/NER/BC5CDR-chem/",Split.test,"output_SL_chem/BC5CDR-chem","/chem_test_result.txt")
evaluation("pretrain/CP_model","../datasets/NER/BC5CDR-chem/",Split.test,"output_CP_SL_chem/BC5CDR-chem","/chem_test_result.txt")
#evaluation("dmis-lab/biobert-base-cased-v1.1","../datasets/NER/BC5CDR-chem/",Split.test,"output_SL/","chem_test_unlabel_result.txt")
#evaluation("pretrain/CP_model","../datasets/NER/BC5CDR-chem/",Split.test,"output_CP_SL/","chem_test_unlabel_result.txt")
