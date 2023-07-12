import torch.nn as nn
import torch


class CNNTeacherModel(nn.Module):   
    def __init__(self, shared_model, tokenizer, num_labels, args, hidden_size):
        super().__init__()
        self.shared_model = shared_model
        self.category_head = nn.Linear(hidden_size, num_labels)
        self.class_head = nn.Linear(hidden_size, num_labels)
        self.variant_head = nn.Linear(hidden_size, num_labels)
        self.base_head = nn.Linear(hidden_size, num_labels)
        self.deprecated_head = nn.Linear(hidden_size, num_labels)
        # Note. pillar has only one label, so no need to train a head
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, source_ids, position_idx, attn_mask, group, labels, return_prob=False, return_logit=False):
        # size: batch_size, num_labels
        hidden_state = self.shared_model(source_ids=source_ids, position_idx=position_idx, attn_mask=attn_mask, return_hidden_state=True)        
        category_logits = self.category_head(hidden_state)
        class_logits = self.class_head(hidden_state)
        variant_logits = self.variant_head(hidden_state)
        base_logits = self.base_head(hidden_state)
        deprecated_logits = self.deprecated_head(hidden_state)
        # iter batch
        logits = torch.empty(category_logits.shape[0], category_logits.shape[1]).float().to(self.args.device)
        for i in range(len(group)):
            if group[i].item() == 0:
                logits[i, :] = category_logits[i]
            elif group[i].item() == 1:
                logits[i, :] = class_logits[i]
            elif group[i].item() == 2:
                logits[i, :] = variant_logits[i]
            elif group[i].item() == 3:
                logits[i, :] = base_logits[i]
            elif group[i].item() == 4:
                logits[i, :] = deprecated_logits[i]
            elif group[i].item() == 5:
                logits[i, :] = labels[i]   
        if return_logit:
            return logits
        elif return_prob:
            prob = torch.softmax(logits, dim=-1)
            return prob
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
