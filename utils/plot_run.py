import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.logger import read_metrics

PANELS = [
    ("loss_pred",      "prediction loss (smooth L1)",     "log"),
    ("loss_var",       "variance hinge (0 = healthy)",    "linear"),
    ("loss_cov",       "feature correlation penalty",     "log"),
    ("tgt_eff_rank",   "target effective rank  (collapse -> 1)", "linear"),
    ("tgt_cos",        "target token cosine  (collapse -> 1)",   "linear"),
    ("tgt_rel_std",    "target rel. std  (collapse -> 0)",       "linear"),
    ("ctx_eff_rank",   "online ctx effective rank",       "linear"),
    ("tgt_norm",       "target token norm",               "linear"),
    ("lr",             "learning rate",                   "linear"),
]


def main(run_dir):
    m = read_metrics(run_dir)
    steps = m["step"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, (key, title, scale) in zip(axes.ravel(), PANELS):
        if key not in m:
            ax.set_visible(False)
            continue
        ax.plot(steps, m[key], lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.set_yscale(scale)
        ax.grid(alpha=0.25)
        if key == "tgt_cos":
            ax.axhline(0.98, color="crimson", ls="--", lw=0.9)
            ax.set_ylim(-0.1, 1.05)
        if key == "tgt_eff_rank":
            ax.axhline(3.0, color="crimson", ls="--", lw=0.9)
    fig.suptitle(f"V-JEPA run: {os.path.basename(run_dir.rstrip('/'))}", fontsize=13)
    fig.tight_layout()
    out = os.path.join(run_dir, "curves.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/vjepa")
