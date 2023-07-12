import torch.nn as nn
import torch


class CombinedTeacher(nn.Module):   
    def __init__(self, g1_model, g2_model, g3_model, config, tokenizer, args):
        super().__init__()
        self.g1_model = g1_model
        self.g2_model = g2_model
        self.g3_model = g3_model
        self.tokenizer = tokenizer
        self.config = config
        self.args = args

    def forward(self, input_ids, group):
        # size: batch_size, num_labels
        logit_g1 = self.g1_model(input_ids, return_logit=True)
        logit_g2 = self.g2_model(input_ids, return_logit=True)
        logit_g3 = self.g3_model(input_ids, return_logit=True)
        # iter batch
        logits = torch.empty(logit_g1.shape[0], logit_g1.shape[1]).float().to(self.args.device)
        for i in range(len(group)):
            # if group 1
            if group[i].item() == 0:
                logits[i, :] = logit_g1[i]
            elif group[i].item() == 1:
                logits[i, :] = logit_g2[i]
            elif group[i].item() == 2:
                logits[i, :] = logit_g3[i]
        return logits
