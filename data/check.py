import numpy as np, os
from data.dataset import ClipDataset, dataset_stats

for n in ["frames", "lives", "episode", "motion"]:
    p = f"proc/{n}.npy""
    print(n, np.load(p, mmap_mode="r").shape, f"{os.path.getsize(p)/1e6:.1f} MB")

ds = ClipDataset()
print("seed1:", dataset_stats(ds, n=4000, seed=1))
print("seed2:", dataset_stats(ds, n=4000, seed=2))