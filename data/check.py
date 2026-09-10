import json
import os
import numpy as np
from data.dataset import ClipDataset, dataset_stats

for n in ["frames", "lives", "episode", "motion", "ball"]:
    p = f"proc/{n}.npy"
    if not os.path.exists(p):
        print(f"{n:8s} MISSING  (run python -m data.{'ball_labels' if n == 'ball' else 'preprocess'})")
        continue
    print(f"{n:8s}", np.load(p, mmap_mode="r").shape, f"{os.path.getsize(p) / 1e6:.1f} MB")

with open("proc/meta.json") as f:
    print("meta    ", json.load(f))

ds = ClipDataset()
print(f"clips   {len(ds)}  static dropped {ds.n_static}/{ds.n_total} "
      f"({ds.n_static / ds.n_total:.1%})  sub {ds.sub:.2f} div {ds.div:.2f}")
print("seed1:", dataset_stats(ds, n=4000, seed=1))
print("seed2:", dataset_stats(ds, n=4000, seed=2))

ball = np.load("proc/ball.npy")
frames = np.load("proc/frames.npy", mmap_mode="r")
assert len(ball) == len(frames), (len(ball), len(frames))
vis = ball[:, 0] > 0
print(f"ball    visible {vis.mean():.1%}  y {ball[vis, 1].min():.1f}-{ball[vis, 1].max():.1f}"
      f"  x {ball[vis, 2].min():.1f}-{ball[vis, 2].max():.1f}")

# the labelled pixel should actually be bright in the 64x64 frame
idx = np.flatnonzero(vis)[::997][:200]
hit = 0
for i in idx:
    y, x = int(round(ball[i, 1])), int(round(ball[i, 2]))
    patch = frames[i, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
    hit += patch.max() > 20
print(f"label check: {hit}/{len(idx)} labelled positions land on a bright pixel")
print("proc ok")
