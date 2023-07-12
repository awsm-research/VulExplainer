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
    
    def __init__(self, roberta, dim_channel, kernel_wins, dropout_rate, num_class, args):
        super(TextCNN, self).__init__()
        self.roberta = roberta
        self.args = args
        emb_dim = roberta.config.hidden_size
        # Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        # FC layer
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)
        
    def forward(self, 
               source_ids, 
               position_idx,
               attn_mask,
               labels=None, 
               return_hidden_state=False):
        # GraphCodeBERT Embeddings
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)        
        inputs_embeddings = self.roberta.embeddings.word_embeddings(source_ids)
        nodes_to_token_mask = nodes_mask[:,:,None] & token_mask[:,None,:] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        emb_x = inputs_embeddings * (~nodes_mask)[:,:,None] + avg_embeddings*nodes_mask[:,:,None]                  
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