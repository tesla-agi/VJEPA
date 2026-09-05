import os
import numpy as np
import torch
from torch.utils.data import Dataset


def segments(lives, episode):
    """Split at life losses AND episode changes. Returns [(start, end)] half-open."""
    cuts = np.flatnonzero((np.diff(lives) < 0) | (np.diff(episode) != 0)) + 1
    bounds = [0, *cuts.tolist(), len(lives)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


class ClipDataset(Dataset):
    def __init__(self, proc_dir="proc", T=4, stride=1,
                 static_thresh=20, filter_static=True,
                 sub=41.81, div=58.91):
        self.T = T
        self.stride = stride
        self.sub = sub
        self.div = div

        self.frames = np.load(os.path.join(proc_dir, "frames.npy"), mmap_mode="r")
        lives = np.load(os.path.join(proc_dir, "lives.npy"))
        episode = np.load(os.path.join(proc_dir, "episode.npy"))
        motion = np.load(os.path.join(proc_dir, "motion.npy"))

        self.index_map = []
        self.n_static = 0
        self.n_total = 0

        for a, b in segments(lives, episode):
            for s in range(a, b - T + 1, stride):
                self.n_total += 1
                if filter_static and motion[s:s + T - 1].max() < static_thresh:
                    self.n_static += 1
                    continue
                self.index_map.append(s)

        self.index_map = np.array(self.index_map, dtype=np.int64)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        s = int(self.index_map[i])
        clip = np.array(self.frames[s:s + self.T])      # (T, 64, 64) uint8
        x = torch.from_numpy(clip).float().unsqueeze(0)       # (1, T, 64, 64)
        return (x - self.sub) / self.div


def dataset_stats(ds, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    old = ds.sub, ds.div
    ds.sub, ds.div = 0.0, 1.0
    xs = torch.cat([ds[int(i)].flatten() for i in idx])
    ds.sub, ds.div = old
    return xs.mean().item(), xs.std().item()


if __name__ == "__main__":
    ds = ClipDataset(filter_static=True)
    print(f"clips {len(ds)}  static dropped {ds.n_static}/{ds.n_total} "
          f"({ds.n_static / ds.n_total:.1%})")

    x = ds[0]
    assert x.shape == (1, ds.T, 64, 64), x.shape
    assert x.dtype == torch.float32

    sub, div = dataset_stats(ds)
    print(f"pixel mean {sub:.2f}  std {div:.2f}")

    # confirm no clip spans a life loss or an episode seam
    lives = np.load("data/proc/lives.npy")
    episode = np.load("data/proc/episode.npy")
    for s in ds.index_map[:2000]:
        w = slice(int(s), int(s) + ds.T)
        assert len(set(lives[w].tolist())) == 1, f"clip at {s} spans a life loss"
        assert len(set(episode[w].tolist())) == 1, f"clip at {s} spans an episode seam"

    print("dataset ok")