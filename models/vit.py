import math

import torch
import torch.nn as nn

from models.layers import ViTBlock, _make_norm


class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, d=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.n_spatial = self.grid_size ** 2
        self.proj = nn.Conv3d(in_chans, d,
                              kernel_size=(1, patch_size, patch_size),
                              stride=(1, patch_size, patch_size))

        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, T, H, W = x.size()
        assert H == W == self.img_size
        h = self.proj(x)                                   # (B, d, T, g, g)
        h = h.flatten(2).permute(0, 2, 1).contiguous()     # (B, T*g*g, d)
        return h


class ViT(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, d_model=192, n_heads=3,
                 n_frames=4, num_layers=6, norm_type="layer_norm", ffn_type="gelu_mlp"):
        super().__init__()
        assert img_size % patch_size == 0
        resid_std = 0.02 / math.sqrt(2 * num_layers)
        self.n_frames = n_frames
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.n_spatial = self.grid_size ** 2
        self.n_tokens = self.n_frames * self.n_spatial
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, d=d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        self.blocks = nn.ModuleList([
            ViTBlock(d_model=d_model, n_heads=n_heads, resid_std=resid_std,
                     norm_type=norm_type, ffn_type=ffn_type)
            for _ in range(num_layers)
        ])
        self.norm = _make_norm(norm_type, d_model, affine=False)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, idx=None):
        B, C, T, H, W = x.size()
        assert T == self.n_frames, f"expected {self.n_frames} frames, got {T}"
        tokens = self.patch_embed(x) + self.pos_embed
        if idx is not None:
            tokens = torch.gather(tokens, 1,
                                  idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward_pooled(self, x, idx=None):
        return self.forward(x, idx=idx).mean(dim=1)

    @torch.no_grad()
    def attention_maps(self, x):
        tokens = self.patch_embed(x) + self.pos_embed
        maps = []
        for block in self.blocks:
            tokens, attn = block(tokens, return_attn=True)
            maps.append(attn)
        return self.norm(tokens), maps
