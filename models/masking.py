import math
import torch


def sample_block(grid_size, scale_range, aspect_range, rng):
    s = rng.uniform(*scale_range)
    r = rng.uniform(*aspect_range)
    A = s * (grid_size ** 2)
    h = int(round(math.sqrt(A / r)))
    w = int(round(math.sqrt(A * r)))
    h = max(1, min(h, grid_size))
    w = max(1, min(w, grid_size))
    top = int(rng.integers(0, grid_size - h + 1))
    left = int(rng.integers(0, grid_size - w + 1))
    return [row * grid_size + col
            for row in range(top, top + h)
            for col in range(left, left + w)]


def spatial_mask(grid_size, scale_range, aspect_range, n_blocks, rng):
    blocks = [sample_block(grid_size, scale_range, aspect_range, rng)
              for _ in range(n_blocks)]
    union = set().union(*blocks)
    complement = sorted(set(range(grid_size ** 2)) - union)
    return sorted(union), complement


def expand_tube(spatial_idx, n_spatial, n_frames):
    return [t * n_spatial + s for t in range(n_frames) for s in spatial_idx]


def sample_mask(batch_size, grid_size, n_frames, n_blocks, k, m,
                scale_range, aspect_range, rng, max_tries=100):
    n_spatial = grid_size ** 2
    assert k + m <= n_spatial, f"k+m={k + m} exceeds {n_spatial} cells"
    ctx_batch, tgt_batch = [], []
    for _ in range(batch_size):
        for _try in range(max_tries):
            masked, ctx_cells = spatial_mask(grid_size, scale_range,
                                             aspect_range, n_blocks, rng)
            if len(ctx_cells) >= k and len(masked) >= m:
                break
        else:
            raise RuntimeError(
                f"no valid mask in {max_tries} tries "
                f"(k={k}, m={m}, n_blocks={n_blocks}, scale={scale_range})")
        rng.shuffle(ctx_cells)
        rng.shuffle(masked)
        ctx_batch.append(expand_tube(sorted(ctx_cells[:k]), n_spatial, n_frames))
        tgt_batch.append(expand_tube(sorted(masked[:m]), n_spatial, n_frames))
    return (torch.tensor(ctx_batch, dtype=torch.long),
            torch.tensor(tgt_batch, dtype=torch.long))


def sample_causal_mask(batch_size, grid_size, n_frames, n_ctx_frames=3):
    n_spatial = grid_size ** 2
    ctx = list(range(n_ctx_frames * n_spatial))
    tgt = list(range(n_ctx_frames * n_spatial, n_frames * n_spatial))
    c = torch.tensor(ctx, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    t = torch.tensor(tgt, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    return c.contiguous(), t.contiguous()


GROUPS = {
    "SHORT_RANGE": dict(n_blocks=6, scale_range=(0.10, 0.18), k=16, m=24),
    "LONG_RANGE":  dict(n_blocks=2, scale_range=(0.30, 0.45), k=16, m=24),
}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    for name, g in GROUPS.items():
        c, t = sample_mask(64, 8, 4, aspect_range=(0.75, 1.5), rng=rng, **g)
        assert c.shape == (64, g["k"] * 4) and t.shape == (64, g["m"] * 4)
        # context and target must be disjoint
        for i in range(64):
            assert not (set(c[i].tolist()) & set(t[i].tolist()))
        print(f"{name:12s} ctx {tuple(c.shape)}  tgt {tuple(t.shape)}  disjoint ok")
    c, t = sample_causal_mask(4, 8, 4)
    print("CAUSAL      ctx", tuple(c.shape), " tgt", tuple(t.shape))
