import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def segments(lives, episode):
    cuts = np.flatnonzero((np.diff(lives) < 0) | (np.diff(episode) != 0)) + 1
    bounds = [0, *cuts.tolist(), len(lives)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def load_meta(proc_dir="proc"):
    path = os.path.join(proc_dir, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


class ClipDataset(Dataset):
    def __init__(self, proc_dir="proc", T=4, stride=1,
                 static_thresh=20, filter_static=True,
                 sub=None, div=None, with_index=False):
        self.T = T
        self.stride = stride
        self.with_index = with_index
        meta = load_meta(proc_dir)
        self.sub = float(meta.get("pixel_mean", 41.81)) if sub is None else float(sub)
        self.div = float(meta.get("pixel_std", 58.91)) if div is None else float(div)

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

    def clip(self, i):
        s = int(self.index_map[i])
        return np.array(self.frames[s:s + self.T]), s

    def __getitem__(self, i):
        clip, s = self.clip(i)
        x = torch.from_numpy(clip).float().unsqueeze(0)       # (1, T, 64, 64)
        x = (x - self.sub) / self.div
        return (x, s) if self.with_index else x


def dataset_stats(ds, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    old_flag, ds.with_index = ds.with_index, False
    old = ds.sub, ds.div
    ds.sub, ds.div = 0.0, 1.0
    xs = torch.cat([ds[int(i)].flatten() for i in idx])
    ds.sub, ds.div = old
    ds.with_index = old_flag
    return xs.mean().item(), xs.std().item()


if __name__ == "__main__":
    ds = ClipDataset(filter_static=True)
    print(f"clips {len(ds)}  static dropped {ds.n_static}/{ds.n_total} "
          f"({ds.n_static / ds.n_total:.1%})  norm sub {ds.sub:.2f} div {ds.div:.2f}")

    x = ds[0]
    assert x.shape == (1, ds.T, 64, 64), x.shape
    assert x.dtype == torch.float32

    sub, div = dataset_stats(ds)
    print(f"pixel mean {sub:.2f}  std {div:.2f}")

    lives = np.load("proc/lives.npy")
    episode = np.load("proc/episode.npy")
    for s in ds.index_map[:2000]:
        w = slice(int(s), int(s) + ds.T)
        assert len(set(lives[w].tolist())) == 1, f"clip at {s} spans a life loss"
        assert len(set(episode[w].tolist())) == 1, f"clip at {s} spans an episode seam"

    print("dataset ok")
