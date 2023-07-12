# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:14:26 2019
@author: HSU, CHIH-CHAO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sup_contrastive_loss import SupConLoss
from focal_loss import FocalLoss

class TextCNN(nn.Module):
    
    def __init__(self, roberta, tokenizer, dim_channel, kernel_wins, dropout_rate, num_class, args):
        super(TextCNN, self).__init__()
        self.roberta = roberta
        self.tokenizer = tokenizer
        self.args = args
        emb_dim = roberta.config.hidden_size
        # Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        # FC layer
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)
        
    def forward(self, 
               input_ids, 
               labels=None, 
               return_hidden_state=False):
        emb_x = self.roberta.embeddings(input_ids)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).unsqueeze(-1).expand(input_ids.shape[0], input_ids.shape[1], 768)
        emb_x = emb_x * attention_mask                 
        emb_x = emb_x.unsqueeze(1)
        con_x = [conv(emb_x) for conv in self.convs]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logit, labels)
            return loss
        prob = torch.softmax(logit, dim=-1)
        if return_hidden_state:
            return fc_x
        else:
            return prob