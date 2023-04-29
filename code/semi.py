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
from torch import optim
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

data_dir = "output_CP_SL/"
model_path='pretrain/CP_model'
#model_path='dmis-lab/biobert-base-cased-v1.1'

set_seed(1)
logger = logging.getLogger(__name__)
@contextlib.contextmanager


def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d



class NERModel(nn.Module):
    def __init__(self, num_labels,config):
        super().__init__()
        #self.bert = AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1',config=config)
        self.bert = AutoModelForTokenClassification.from_pretrained(model_path,config=config)
        #self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(num_labels, num_labels)
        
    def forward(self, input_ids, attention_mask,labels,token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,labels=labels,token_type_ids=token_type_ids)
        pooled_output = outputs.logits
        #pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

class VATLoss(nn.Module):

    def __init__(self,num_labels, xi=10.0, eps=1.0, ip=1,alpha=0.01):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.num_labels = num_labels
        self.alpha=alpha
        
    def forward(self, model, logits_l, y_l,attention_mask, x_u,attention_mask_u):
        loss = None
        if y_l is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, y_l.view(-1), torch.tensor(loss_fct.ignore_index).type_as(y_l)
                )
                loss_l = loss_fct(active_logits, active_labels)
            else:
                loss_l = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
        
        
        # Generate virtual adversarial perturbations
        with torch.no_grad():
            output_m = model(x_u,attention_mask_u)
            pred = F.softmax(output_m.logits, dim=1)
        x_u_embed = output_m.hidden_states[0]
        x_u_embed = x_u_embed.to(x_u.device)

        # prepare random unit tensor
        d = torch.rand(x_u_embed.shape).sub(0.5).to(x_u.device)
        d = _l2_normalize(d)
        

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(inputs_embeds=x_u_embed + self.xi * d,attention_mask = attention_mask_u).logits
                logp_hat = F.log_softmax(pred_hat, dim=1)
                #print(logp_hat.size())
                #print(attention_mask_u.size())
                adv_distance = F.kl_div(logp_hat, pred, reduction='none')
                adv_distance = adv_distance*attention_mask_u.unsqueeze(-1)
                adv_distance = adv_distance.sum()/logp_hat.size()[0]
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(inputs_embeds=x_u_embed + r_adv,attention_mask=attention_mask_u).logits
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='none')
            lds = lds*attention_mask_u.unsqueeze(-1)
            lds = lds.sum()/logp_hat.size()[0]
            
        # Combine losses and return
        loss = loss_l + self.alpha*lds
        return loss
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
        
from utils_ner import NerDataset, Split, get_labels
labels = get_labels("../datasets/NER/NCBI-disease/labels.txt")
num_labels = len(labels)
label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
      # download model & vocab.  # download model & vocab.

config = AutoConfig.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        output_hidden_states=True
    )
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
)



train_dataset = (
        NerDataset(
            data_dir="../datasets/NER/NCBI-disease/labeled_part/",
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=192,
            overwrite_cache=True,
            mode=Split.train,
        )
    )
unlabel_dataset = (
        NerDataset(
            data_dir="../datasets/NER/NCBI-disease/labeled_part/",
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=192,
            overwrite_cache=True,
            mode=Split.unlabel,
        )
    )
def collate_fn(batch):
    input_ids = torch.tensor([item.input_ids for item in batch])
    attention_mask = torch.tensor([item.attention_mask for item in batch])
    token_type_ids = torch.tensor([item.token_type_ids for item in batch])
    label_ids = torch.tensor([item.label_ids for item in batch])
    
    
    return input_ids, attention_mask, token_type_ids,label_ids
test_dataset = NerDataset(
            data_dir="../datasets/NER/NCBI-disease/labeled_part/",
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=192,
            overwrite_cache=True,
            mode=Split.test,
        )
dev_dataset = (
        NerDataset(
            data_dir="../datasets/NER/NCBI-disease/labeled_part/",
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=192,
            overwrite_cache=True,
            mode=Split.dev,
        )
    )

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
#train_data = [{'text': 'The patient was prescribed aspirin for pain relief', 'targets': [1, 0, 0, 0, 0, 0, 0, 0, 0]}]
#unlabeled_data = [{'text': 'The researchers conducted a study to investigate the effectiveness of the drug', 'targets': [0, 0, 0, 0, 0, 0, 0, 0, 0]}]
#train_dataset = NERDataset(train_data, tokenizer)
#unlabeled_dataset = NERDataset(unlabeled_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
unlabel_loader = DataLoader(unlabel_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
unlabel_iter = []
for item in enumerate(unlabel_loader):
    unlabel_iter.append(item)
#train_data = [{'text': 'The patient was prescribed aspirin for pain relief', 'targets': [1, 0, 0, 0, 0, 0, 0, 0]}]
#unlabeled_data = [{'text': 'The researchers conducted a study to investigate the effectiveness of the drug', 'targets': [0, 0, 0, 0, 0, 0, 0, 0, 0]}]
#train_dataset = NERDataset(train_data, tokenizer)
#unlabeled_dataset = NERDataset(unlabeled_data, tokenizer)
#train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
#unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,
        
    )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"
model.to(device)

criterion = VATLoss(num_labels)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
best_for_now = 0
from tqdm import tqdm
for epoch in tqdm(range(10),desc="Training: ",dynamic_ncols=True):
    model.train()
    total_loss = 0
    for batch_idx, (input_ids, attention_mask, token_type_ids,label_ids) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        input_unlabel = unlabel_iter[batch_idx][1][0]
        attention_mask_unlabel = unlabel_iter[batch_idx][1][1]
        attention_mask = attention_mask.to(device)
        attention_mask_unlabel = attention_mask_unlabel.to(device)
        token_type_ids = token_type_ids.to(device)
        label_ids=torch.tensor(label_ids)
        label_ids = label_ids.to(device)
        input_unlabel = input_unlabel.to(device)
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask,labels=label_ids,token_type_ids=token_type_ids)
        logits = output.logits
        #print(logits)
        loss = criterion(model, logits, label_ids,attention_mask, input_unlabel,attention_mask_unlabel)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Avg. Loss: {avg_loss:.4f}')
    
    model.eval()
# Step 4: Iterate over test data and get predicted tags
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in dev_loader:
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
    if metrics["f1"]>best_for_now:
        best_metrics=metrics
        model.save_pretrained(data_dir)

with open(data_dir+"dev_result.txt", "w") as writer:
    for key, value in best_metrics.items():
        logger.info("***** Dev results *****")
        logger.info("  %s = %s", key, value)
        writer.write("%s = %s\n" % (key, value))        
        
model=AutoModelForTokenClassification.from_pretrained(data_dir)
model.to(device)
model.eval()

# Step 4: Iterate over test data and get predicted tags
y_true = []
y_pred = []
with torch.no_grad():
    for batch in test_loader:
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

with open(data_dir+"test_result.txt", "w") as writer:
    logger.info("***** Test results *****")
    for key, value in metrics.items():
        logger.info("  %s = %s", key, value)
        writer.write("%s = %s\n" % (key, value))