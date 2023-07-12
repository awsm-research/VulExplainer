import torch.nn as nn
import torch


class CombinedTeacher(nn.Module):   
    def __init__(self, g1_model, g2_model, g3_model, g4_model, g5_model, config, tokenizer, args):
        super().__init__()
        self.g1_model = g1_model
        self.g2_model = g2_model
        self.g3_model = g3_model
        self.g4_model = g4_model
        self.g5_model = g5_model
        self.tokenizer = tokenizer
        self.config = config
        self.args = args

    def forward(self, source_ids, position_idx, attn_mask, group):
        # size: batch_size, num_labels
        logit_g1 = self.g1_model(source_ids=source_ids, position_idx=position_idx, attn_mask=attn_mask, return_logits=True)
        logit_g2 = self.g2_model(source_ids=source_ids, position_idx=position_idx, attn_mask=attn_mask, return_logits=True)
        logit_g3 = self.g3_model(source_ids=source_ids, position_idx=position_idx, attn_mask=attn_mask, return_logits=True)
        logit_g4 = self.g4_model(source_ids=source_ids, position_idx=position_idx, attn_mask=attn_mask, return_logits=True)
        logit_g5 = self.g5_model(source_ids=source_ids, position_idx=position_idx, attn_mask=attn_mask, return_logits=True)
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
            elif group[i].item() == 3:
                logits[i, :] = logit_g4[i]
            elif group[i].item() == 4 or group[i].item() == 5:
                logits[i, :] = logit_g5[i]
            else:
                print("ERROR")
                exit()
        return logits
