import torch


@torch.no_grad()
def collapse_stats(z, max_tokens=4096, seed=0):
    z = z.detach().float()
    if z.dim() == 3:
        z = z.flatten(0, 1)
    z = z.cpu()
    if z.shape[0] > max_tokens:
        g = torch.Generator().manual_seed(seed)
        z = z[torch.randperm(z.shape[0], generator=g)[:max_tokens]]

    norm = z.norm(dim=-1).mean()
    std = z.std(dim=0, unbiased=False).mean()

    def _eff_rank(a):
        sigma = torch.linalg.svdvals(a)
        p = sigma / (sigma.sum() + 1e-12)
        return torch.exp(-(p * torch.log(p + 1e-12)).sum())
    eff_rank = _eff_rank(z)
    eff_rank_c = _eff_rank(z - z.mean(dim=0, keepdim=True))

    u = torch.nn.functional.normalize(z, dim=-1)
    h = u.shape[0] // 2
    cos = (u[:h] * u[h:2 * h]).sum(-1).mean()

    return {
        "norm": norm.item(),
        "std": std.item(),
        "rel_std": (std / (norm + 1e-12)).item(),
        "eff_rank": eff_rank.item(),
        "eff_rank_c": eff_rank_c.item(),
        "cos": cos.item(),
    }

def is_collapsing(stats, min_eff_rank=3.0, max_cos=0.98, min_rel_std=0.01):
    reasons = []
    if stats["eff_rank"] < min_eff_rank:
        reasons.append(f"eff_rank {stats['eff_rank']:.2f} < {min_eff_rank}")
    if stats["cos"] > max_cos:
        reasons.append(f"token cos {stats['cos']:.4f} > {max_cos}")
    if stats["rel_std"] < min_rel_std:
        reasons.append(f"rel_std {stats['rel_std']:.5f} < {min_rel_std}")
    return reasons

def monitor(z):
    s = collapse_stats(z)
    return s["std"] ** 2, s["eff_rank"]
