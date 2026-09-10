"""V-JEPA objective.

The original loss in this repo was

    F.l1_loss(s_hat, s_y.gather(...).detach())

which has a *trivial global minimum at zero*: both encoders end in a norm layer
with a learnable gain, so driving that gain to 0 makes the target identically 0
and the predictor only has to output 0.  Gradient descent finds it, and AdamW's
weight decay actively pushes the gain there.  That is the representation
collapse.

Two changes fix it:

1. `normalize_target` -- per-token LayerNorm of the EMA target before the loss
   (this is what I-JEPA/V-JEPA actually do:
   `h = F.layer_norm(h, (h.size(-1),))`).  Every target token now has fixed
   norm sqrt(d), so "shrink everything" is no longer a minimiser.

2. A VICReg-style variance + covariance term on the online embedding.  (1)
   removes the *scale* collapse but not the *directional* one -- mapping every
   patch to the same unit vector is still a global minimum.  The hinge variance
   term forces each feature dimension to keep std >= gamma across the token
   batch, and the correlation term decorrelates dimensions so the embedding
   cannot fold onto a low-dimensional subspace.
"""

import torch
import torch.nn.functional as F


def normalize_target(y):
    """I-JEPA target normalisation: LayerNorm over the feature dim, no affine."""
    return F.layer_norm(y, (y.shape[-1],))


def gather_tokens(z, idx):
    """z: (B, N, d), idx: (B, K) -> (B, K, d)"""
    return torch.gather(z, 1, idx.unsqueeze(-1).expand(-1, -1, z.shape[-1]))


def variance_loss(z, gamma=1.0, eps=1e-4):
    """Hinge on the per-dimension std of the pooled token batch.

    Zero once every dimension carries at least `gamma` of spread, so it does
    nothing to a healthy model and only bites when the encoder starts to
    collapse.
    """
    z = z.flatten(0, 1)                                  # (B*N, d)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(z, eps=1e-4):
    """Off-diagonal mass of the *correlation* matrix.

    VICReg uses the raw covariance; we use the correlation so the term is
    scale-invariant and therefore does not fight `variance_loss` (which wants
    scale up) by simply shrinking the features.
    """
    z = z.flatten(0, 1)
    z = z - z.mean(dim=0, keepdim=True)
    n, d = z.shape
    std = z.std(dim=0, unbiased=False) + eps
    z = z / std
    corr = (z.T @ z) / max(n - 1, 1)
    off = corr.pow(2).sum() - corr.pow(2).diagonal().sum()
    return off / d


def vjepa_loss(s_hat, s_y_all, tgt_idx, s_ctx=None,
               lambda_var=1.0, lambda_cov=0.04, gamma=1.0, beta=1.0):
    """Returns (total_loss, parts_dict).

    s_hat     : (B, K, d) predictor output for the masked tokens
    s_y_all   : (B, N, d) EMA-target output over the *full* clip
    tgt_idx   : (B, K)    indices of the masked tokens
    s_ctx     : (B, C, d) online encoder output on the context (regularised)
    """
    with torch.no_grad():
        t = normalize_target(gather_tokens(s_y_all, tgt_idx))

    pred = F.smooth_l1_loss(s_hat.contiguous(), t.contiguous(), beta=beta)

    parts = {"pred": pred.detach()}
    total = pred

    if s_ctx is not None and (lambda_var > 0 or lambda_cov > 0):
        var = variance_loss(s_ctx, gamma=gamma)
        cov = covariance_loss(s_ctx)
        total = total + lambda_var * var + lambda_cov * cov
        parts["var"] = var.detach()
        parts["cov"] = cov.detach()

    parts["total"] = total.detach()
    return total, parts
