import torch
import math
import torch.nn as nn
from models.layers import ViTBlock,_make_norm


class Predictor(nn.Module):
    def __init__(self,d_model=192,d_pred=96,n_heads=3,n_patches=256,depth=2,norm_type="layer_norm",ffn_type="gelu_mlp"):
        super(Predictor,self).__init__()

        resid_std=0.02/math.sqrt(2*depth)
        self.proj_in=nn.Linear(d_model,d_pred)
        self.mask_token=nn.Parameter(torch.zeros(1,1,d_pred))
        self.pos_pred=nn.Parameter(torch.zeros(1,n_patches,d_pred))
        self.blocks=nn.ModuleList([
            ViTBlock(d_model=d_pred,n_heads=n_heads,resid_std=resid_std,
                     norm_type=norm_type,ffn_type=ffn_type)
            for _ in range(depth)
        ])
        self.norm=_make_norm(norm_type,d_pred)
        self.proj_out=nn.Linear(d_pred,d_model)

        nn.init.trunc_normal_(self.proj_in.weight,std=0.02)
        nn.init.trunc_normal_(self.mask_token,std=0.02)
        nn.init.trunc_normal_(self.pos_pred,std=0.02)

        nn.init.trunc_normal_(self.proj_out.weight,std=resid_std)

        nn.init.zeros_(self.proj_in.bias)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self,x,ctx_idx,tgt_idx):
        ctx=self.proj_in(x)+self.pos_pred[:,ctx_idx,:]
        M=self.mask_token+self.pos_pred[:,tgt_idx,:]
        M=M.expand(ctx.shape[0],-1,-1)
        H=torch.cat([ctx,M],dim=1)
        for block in self.blocks:
            H=block(H)

        H=self.norm(H)
        H=self.proj_out(H)
        return H[:,ctx.shape[1]:,:]