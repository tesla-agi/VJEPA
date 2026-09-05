import torch
import torch.nn.functional as F

def vjepa_loss(s_hat,s_y,tgt_idx,d_model):
    idx_exp=tgt_idx.unsqueeze(-1).expand(-1,-1,d_model)
    t=torch.gather(s_y,1,idx_exp)
    t=t.detach()
    return F.l1_loss(s_hat,t)


