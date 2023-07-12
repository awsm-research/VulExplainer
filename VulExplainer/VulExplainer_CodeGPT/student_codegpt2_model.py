import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

        
class StudentGPT2(nn.Module):   
    def __init__(self, decoder, config, tokenizer, args, num_labels):
        super().__init__()
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.score = nn.Linear(config.n_embd, num_labels, bias=False)
        self.args = args
    
    def forward(self, 
                input_ids=None, 
                labels=None, 
                soft_label=None, 
                hard_label=None,
                best_beta=None):
        last_token_loc = torch.ne(input_ids, self.tokenizer.pad_token_id).sum(-1) - 1

        last_hidden_state = self.decoder(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state
        logits = self.score(last_hidden_state)
        # take last token as <cls> token
        logits_cls = logits[torch.arange(input_ids.shape[0], device=self.args.device), last_token_loc]
        # take first token as <dis> token
        logits_dis = logits[torch.arange(input_ids.shape[0], device=self.args.device), 0]

        prob_cls = torch.softmax(logits_cls, dim=-1)
        prob_dis = nn.functional.log_softmax(logits_dis, dim=-1)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            kl_loss_fct = nn.KLDivLoss(reduction="batchmean", log_target=True)
            loss_cls = loss_fct(logits_cls, labels)
            # KL Loss with log softmax input and target
            if soft_label is not None:
                loss_dis = kl_loss_fct(prob_dis, soft_label)
                return loss_cls, loss_dis
            elif hard_label is not None:
                loss_dis = loss_fct(logits_dis, hard_label)
                return loss_cls, loss_dis
        else:
            #beta = self.args.beta
            #prob = beta * prob_cls + (1-beta) * prob_dis
            #return prob

            if best_beta is not None:
                prob = best_beta * prob_cls + (1-best_beta) * prob_dis
                return prob
            betas = [0.5, 0.6, 0.7, 0.8, 0.9]
            probs = []
            for beta in betas:
                prob = beta * prob_cls + (1-beta) * prob_dis
                probs.append(prob)
            return probs, betas