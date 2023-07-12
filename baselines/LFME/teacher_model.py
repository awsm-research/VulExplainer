import torch.nn as nn
import torch 


class CNNTeacherModel(nn.Module):  
    def __init__(self, shared_model, tokenizer, num_labels, args, hidden_size):
        super().__init__()
        self.shared_model = shared_model
        self.cls_head = nn.Linear(hidden_size, num_labels)
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids, labels=None, soft_label=None, return_prob=False, return_logit=False, return_hidden_state=False):
        # size: batch_size, num_labels
        hidden_state = self.shared_model(input_ids=input_ids, return_hidden_state=True)
        logits = self.cls_head(hidden_state)  
        prob_dis = nn.functional.log_softmax(logits, dim=-1)
        if return_prob:
            prob = torch.softmax(logits, dim=-1)
            return prob
        elif return_logit:
            return logits
        elif soft_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            kl_loss_fct = nn.KLDivLoss(reduction="batchmean", log_target=True)
            loss_dis = kl_loss_fct(prob_dis, soft_label)
            return loss, loss_dis
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
