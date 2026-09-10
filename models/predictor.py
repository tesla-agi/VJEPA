import math

import torch
import torch.nn as nn

from models.layers import ViTBlock, _make_norm


class Predictor(nn.Module):
    def __init__(self, d_model=192, d_pred=96, n_heads=3, n_patches=256, depth=4,
                 norm_type="layer_norm", ffn_type="gelu_mlp"):
        super().__init__()

        resid_std = 0.02 / math.sqrt(2 * depth)
        self.proj_in = nn.Linear(d_model, d_pred)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_pred))
        self.pos_pred = nn.Parameter(torch.zeros(1, n_patches, d_pred))
        self.blocks = nn.ModuleList([
            ViTBlock(d_model=d_pred, n_heads=n_heads, resid_std=resid_std,
                     norm_type=norm_type, ffn_type=ffn_type)
            for _ in range(depth)
        ])
        self.norm = _make_norm(norm_type, d_pred)
        self.proj_out = nn.Linear(d_pred, d_model)
        nn.init.trunc_normal_(self.proj_in.weight, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_pred, std=0.02)
        nn.init.trunc_normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.zeros_(self.proj_out.bias)

    def _pos(self, idx):
        d = self.pos_pred.shape[-1]
        pos = self.pos_pred.expand(idx.shape[0], -1, -1)
        return torch.gather(pos, 1, idx.unsqueeze(-1).expand(-1, -1, d))

    def forward(self, x, ctx_idx, tgt_idx):
        ctx = self.proj_in(x) + self._pos(ctx_idx)
        M = self.mask_token + self._pos(tgt_idx)
        H = torch.cat([ctx, M], dim=1)
        for block in self.blocks:
            H = block(H)
        H = self.proj_out(self.norm(H))
        # .contiguous(): the slice is non-contiguous and MPS's smooth_l1 rejects it
        return H[:, ctx.shape[1]:, :].contiguous()
