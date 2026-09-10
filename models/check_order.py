import torch
from models.vit import ViT

m = ViT()
x = torch.zeros(1, 1, 4, 64, 64)
x[:, :, 2] = 1.0                      # only frame 2 is non-zero
tok = m.patch_embed(x)                # (1, 256, 192)
nz = (tok.abs().sum(-1) > 1e-6).nonzero()[:, 1]
print("non-zero token range:", nz.min().item(), nz.max().item())