# CS769_Project
## Team ID: 12
## Team Member: Grace Yi Chen, Shuyang Chen, Dandi Chen

This is the python code for **Named Entity Recognition in Biomedical Domain**

Details of the folder:

1. **code folder**

 * preprocess.sh
    + This file contains the command to preprocess BioNER data sets.
    
 * test_ner.sh
    + This file contains the command to fine-tune BioBERT+Linear model on target data set.
    + Running parameters can be changed in code/test_ner.sh file. 
 * run_ner.py
    + This code contains all the functions for load, pretrain and fine-tune.
 * utils_ner.py
    + This code contains some useful functions for different purpose.
 * scripts folder
    + This folder contains the function for preprocess BioNER data sets.

2. **output folder**
 * This folder contains the evaluation outputs for the paper.

3. **download.sh**
 * This file contains the command to download BioNER data.

4. **How to run the files**
 * bash download.sh; bash preprocess.sh; bash test_ner.sh

Paper we are following: BioBERT paper [1]. 

### Reference

[1] Lee, Jinhyuk, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. "BioBERT: a pre-trained biomedical language representation model for biomedical text mining." Bioinformatics 36, no. 4 (2020): 1234-1240.
