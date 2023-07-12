import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2ForSequenceClassification
from focal_loss import FocalLoss 


class Model(nn.Module):   
    def __init__(self, decoder, config, tokenizer, args, num_labels):
        super(Model, self).__init__()
        self.decoder = decoder
        self.score = nn.Linear(config.n_embd, num_labels, bias=False)
        self.tokenizer = tokenizer
        self.args = args
    
    def forward(self, input_ids, labels=None, logit_adjustment=None, focal_loss=False):
        last_token_loc = torch.ne(input_ids, self.tokenizer.pad_token_id).sum(-1) - 1
        last_hidden_state = self.decoder(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state
        logits = self.score(last_hidden_state)
        logits = logits[torch.arange(input_ids.shape[0], device=self.args.device), last_token_loc]
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            if logit_adjustment is not None:
                logits = logits + logit_adjustment
            if focal_loss:
                criterion_fl = FocalLoss()
                loss = criterion_fl(logits, labels)
                loss = loss.mean()
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob