# CS769_Project
## Team ID: 12
## Team Member: Grace Yi Chen, Shuyang Chen, Dandi Chen

This is the python code for **Incorporating Unlabeled Data For Biomedical Named Entity Recognition**

Details of the folder:

1. **code folder**

 *Preprocessing*
 * download.sh
    + This file contains the command to download BioNER data.
 * preprocess.sh and preprocess.py
    + This file contains the command to preprocess BioNER data sets.
 * data_reprocess_for_unlabel.py
    + This code is designed for automatically randomly selecting sentences to mask their labels and generate unlabeled data used in continue pretraining and semi-supervised learning.
 * noisy_data_gen.py
    + Functions for generating noisy data to test model robustness. 
 
 *Training*
 * test_ner.sh
    + This file contains the command to fine-tune BioBERT+Linear model on target data set.
    + Running parameters can be changed in code/test_ner.sh file. 
 * run_ner.py
    + This code contains all the functions for load, pretrain and fine-tune.
 * utils_ner.py
    + This code contains some useful functions for different purpose.
 * ContinuePretraining.py
    + This code contains functions for continuous pretraining on the unlabeled data set.
 * semi_supervise.py
    + This code contains functions for fine-tuning with VAT loss on training data.
 
 *Evaluation*
 * Evaluation.py
    + This script loads models and do evaluation on various test data sets including noisy data.

2. **output folder**
 * This folder contains the evaluation outputs for all models in all settings.

3. **How to run the files**
 * Basic BioBERT training
    + change the model_name_or_path to dmis-lab/biobert-base-cased-v1.1
    + bash download.sh; bash preprocess.sh; bash test_ner.sh
 * BioBERT+TAPT
    + python ContinuePretraining.py
    + change the model_name_or_path to pretrain/CP-model
    + python data_reprocess_for_unlabel.py; bash test_ner.sh
 * BioBERT+VAT
    + change model_path to 'dmis-lab/biobert-base-cased-v1.1'
    + python semi_supervised.py
 * BioBERT+VAT+TAPT
    + change model_path to 'pretrain/CP_model'
    + python semi_supervised.py

Paper we are following: BioBERT paper [1]. 

### Reference

[1] Lee, Jinhyuk, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. "BioBERT: a pre-trained biomedical language representation model for biomedical text mining." Bioinformatics 36, no. 4 (2020): 1234-1240.
