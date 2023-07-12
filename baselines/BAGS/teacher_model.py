import torch.nn as nn
import torch


class CNNTeacherModel(nn.Module):  
    def __init__(self, shared_model, tokenizer, num_labels, args, hidden_size):
        super().__init__()
        self.shared_model = shared_model
        self.g1_head = nn.Linear(hidden_size, num_labels)
        self.g2_head = nn.Linear(hidden_size, num_labels)
        self.g3_head = nn.Linear(hidden_size, num_labels)
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids, groups, labels, return_prob=False, return_logit=False):
        # size: batch_size, num_labels
        hidden_state = self.shared_model(input_ids=input_ids, return_hidden_state=True)
        g1_logits = self.g1_head(hidden_state)
        g2_logits = self.g2_head(hidden_state)
        g3_logits = self.g3_head(hidden_state)
        # iter batch
        logits = torch.empty(g1_logits.shape[0], g1_logits.shape[1]).float().to(self.args.device)
        for i in range(len(groups)):
            if groups[i].item() == 0:
                logits[i, :] = g1_logits[i]
            elif groups[i].item() == 1:
                logits[i, :] = g2_logits[i]
            elif groups[i].item() == 2:
                logits[i, :] = g3_logits[i]  
        if return_prob:
            logits = g1_logits + g2_logits + g3_logits
            prob = torch.softmax(logits, dim=-1)
            return prob
        elif return_logit:
            return logits
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
