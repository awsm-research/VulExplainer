import torch.nn as nn
import torch


class CNNTeacherModel(nn.Module):   
    def __init__(self, shared_model, tokenizer, num_labels, args, hidden_size):
        super().__init__()
        self.shared_model = shared_model
        self.head_head = nn.Linear(hidden_size, num_labels)
        self.mid_head = nn.Linear(hidden_size, num_labels)
        self.tail_head = nn.Linear(hidden_size, num_labels)
        self.tokenizer = tokenizer 
        self.args = args

    def forward(self, source_ids, position_idx, attn_mask, group, labels, return_prob=False, return_logit=False):
        # size: batch_size, num_labels
        hidden_state = self.shared_model(source_ids=source_ids, position_idx=position_idx, attn_mask=attn_mask, return_hidden_state=True)        
        head_logits = self.head_head(hidden_state)
        mid_logits = self.mid_head(hidden_state)
        tail_logits = self.tail_head(hidden_state)
        # iter batch
        logits = torch.empty(head_logits.shape[0], head_logits.shape[1]).float().to(self.args.device)
        for i in range(len(group)):
            if group[i].item() == 0:
                logits[i, :] = head_logits[i]
            elif group[i].item() == 1:
                logits[i, :] = mid_logits[i]
            elif group[i].item() == 2:
                logits[i, :] = tail_logits[i]
            else:
                print("ERROR")
                exit() 
        if return_logit:
            return logits
        elif return_prob:
            prob = torch.softmax(logits, dim=-1)
            return prob
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
