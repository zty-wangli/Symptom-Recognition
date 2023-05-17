#!/usr/bin/env python
# coding: utf-8




import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset


import re
import pandas as pd
from bert4keras.snippets import sequence_padding,DataGenerator
from bert4keras.tokenizers import Tokenizer


entity_labels = ['Symptom-0','Symptom-1','Symptom-2'] #data
#entity_labels = ['Symptom'] #data_ner

id2label = {i:j for i,j in enumerate(sorted(entity_labels))}
label2id = {j:i for i,j in id2label.items()}

num_labels = len(entity_labels)+ 1

vocab_path = './bert_weight_file/Chinese-BERT-wwm/vocab.txt'
tokenizer = Tokenizer(vocab_path,do_lower_case=True)



def load_data(data_path,max_len):
    
    sentence = []
    labels = []
    X = []
    y = []
    datasets = []
    samples_len = []
    split_pattern = re.compile(r'[;；。，、？\.\?!]')
    with open(data_path,'r',encoding= 'utf8') as f:
        for line in f.readlines():
            line = line.strip().split()
            if(not line or len(line) < 2):
                X.append(sentence.copy())
                y.append(labels.copy())
                sentence.clear()
                labels.clear()
                continue
            word, tag = line[0], line[1]
            if split_pattern.match(word) and len(sentence) >= max_len:
                sentence.append(word)
                labels.append(tag)
                sentence.clear()
                labels.clear()
            else:
                sentence.append(word)
                labels.append(tag)
    if len(sentence):
        X.append(sentence.copy())
        sentence.clear()
        y.append(labels.copy())
        labels.clear()

    for token_seq,label_seq in zip(X,y):
        if len(token_seq) < 2:
            continue
        sample_seq, last_flag = [], ''
        for token, this_flag in zip(token_seq,label_seq):
            
            if this_flag == 'O' and last_flag == 'O':
                sample_seq[-1][0] += token
            elif this_flag == 'O' and last_flag != 'O':
                sample_seq.append([token, 'O'])
            elif this_flag[:1] == 'B':
                sample_seq.append([token, this_flag[2:]]) 
                save = token
            elif this_flag[:1] == 'I' and last_flag[:1] == 'B':
                del sample_seq[-1][-1] 
                del sample_seq[-1][-1] 
                sample_seq.append([save+token, this_flag[2:]])
                save = save+token
            elif this_flag[:1] == 'I' and last_flag[:1] == 'I':
                del sample_seq[-1][-1] 
                del sample_seq[-1][-1]
                sample_seq.append([save+token, this_flag[2:]])
                save = save+token
            last_flag = this_flag
        datasets.append([x for x in sample_seq if x != []])       
        samples_len.append(len(token_seq))
        
    return datasets,y

class data_generator(DataGenerator):  # 数据生成器
    def __iter__(self, random=True):
        batch_token_ids,batch_segment_ids,batch_labels = [], [], []
        for is_end,item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0] #[CLS]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < 100:
                    token_ids += w_token_ids
                    if l =='O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 +1
                        I = label2id[l] * 2 +1
                        labels += ([B] + [I] * (len(w_token_ids) - 1)) 
                else:
                    break
            token_ids += [tokenizer._token_end_id] # [seq]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    