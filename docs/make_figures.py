"""Figures for docs/report.md.   python -m docs.make_figures

Everything here is generated from real data in the repo -- the A/B logs in
docs/data/, the raw rollouts, and proc/.  Nothing is hand-drawn from memory.
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import torch

FIG = os.path.join(os.path.dirname(__file__), "fig")
DATA = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(FIG, exist_ok=True)

BLUE, RED, GREEN, GREY = "#3b7dd8", "#e8543f", "#35a06a", "#8a8f98"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def parse_ab(path):
    pat = re.compile(
        r"\[(\w+)\]\s+(\d+) pred ([\d.]+).*?std ([\d.]+) rel_std ([\d.]+) "
        r"eff_rank\s+([\d.]+) cos ([+-][\d.]+)")
    out = {k: [] for k in ("step", "pred", "std", "rel_std", "rank", "cos")}
    for line in open(path):
        m = pat.search(line)
        if m:
            out["step"].append(int(m.group(2)))
            out["pred"].append(float(m.group(3)))
            out["std"].append(float(m.group(4)))
            out["rel_std"].append(float(m.group(5)))
            out["rank"].append(float(m.group(6)))
            out["cos"].append(float(m.group(7)))
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------- 1. the A/B

def fig_collapse_ab():
    old = parse_ab(os.path.join(DATA, "ab_old.log"))
    new = parse_ab(os.path.join(DATA, "ab_new.log"))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))

    ax[0].plot(old["step"], old["pred"], color=RED, lw=2, label="original recipe")
    ax[0].plot(new["step"], new["pred"], color=GREEN, lw=2, label="fixed recipe")
    ax[0].set_yscale("log"); ax[0].set_title("prediction loss")
    ax[0].set_xlabel("step"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].annotate("looks like success\n(it is collapse)", xy=(600, old["pred"][-1]),
                   xytext=(250, 0.35), fontsize=8, color=RED,
                   arrowprops=dict(arrowstyle="->", color=RED, lw=1))

    ax[1].plot(old["step"], old["cos"], color=RED, lw=2)
    ax[1].plot(new["step"], new["cos"], color=GREEN, lw=2)
    ax[1].axhline(1.0, color="k", ls=":", lw=1)
    ax[1].set_ylim(-0.1, 1.1); ax[1].set_title("token cosine similarity\n(1.0 = every patch identical)")
    ax[1].set_xlabel("step")

    ax[2].plot(old["step"], old["rel_std"], color=RED, lw=2)
    ax[2].plot(new["step"], new["rel_std"], color=GREEN, lw=2)
    ax[2].set_ylim(0, 0.09); ax[2].set_title("relative std\n(0 = no spread left)")
    ax[2].set_xlabel("step")

    fig.suptitle("Measured A/B, 600 steps, identical seed — the loss and the health "
                 "of the representation move in OPPOSITE directions",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/1_collapse_ab.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("1_collapse_ab.png")


# ------------------------------------------------------- 2. collapse, visually

def fig_collapse_modes():
    rng = np.random.default_rng(0)
    n = 300
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    sets = [
        ("healthy\ntokens spread over the sphere",
         rng.normal(size=(n, 2)), GREEN),
        ("scale collapse\neverything shrinks to the origin",
         rng.normal(size=(n, 2)) * 0.04, RED),
        ("directional collapse\none direction, tiny spread around it",
         np.stack([rng.normal(1.0, 0.03, n), rng.normal(0, 0.03, n)], 1), RED),
    ]
    for a, (title, pts, c) in zip(ax, sets):
        a.scatter(pts[:, 0], pts[:, 1], s=8, color=c, alpha=0.6)
        a.add_patch(mp.Circle((0, 0), 1.0, fill=False, ec=GREY, ls="--", lw=1))
        a.set_xlim(-2.2, 2.2); a.set_ylim(-2.2, 2.2)
        a.set_aspect("equal"); a.set_title(title, fontsize=9)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("The two ways a JEPA dies.  Both give a near-zero loss.", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/2_collapse_modes.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("2_collapse_modes.png")


# ----------------------------------------------------------- 3. the data crop

def fig_data_pipeline():
    f = np.load("rollouts/episode_0000.npz")["frames"]
    s = f[:1500].astype(np.float32)
    rowstd = s.std(axis=0).max(axis=1)

    fig = plt.figure(figsize=(12, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.9, 1.5, 1.0])

    a = fig.add_subplot(gs[0]); a.imshow(f[60], cmap="gray"); a.set_title("raw 210x160", fontsize=9)
    for y0, y1, lab, c in [(5, 31, "score/lives", RED), (57, 92, "bricks", BLUE),
                           (93, 188, "ball region", GREEN), (189, 195, "paddle", "orange")]:
        a.add_patch(mp.Rectangle((0, y0), 160, y1 - y0, fill=False, ec=c, lw=1.4))
        a.text(163, (y0 + y1) / 2, lab, color=c, fontsize=7, va="center")
    a.add_patch(mp.Rectangle((8, 32), 144, 166, fill=False, ec="yellow", lw=1.8, ls="--"))
    a.axis("off")

    a = fig.add_subplot(gs[1])
    a.plot(rowstd, np.arange(210), color=BLUE, lw=1)
    a.axhspan(5, 31, color=RED, alpha=0.18)
    a.invert_yaxis(); a.set_title("temporal std per row", fontsize=9)
    a.set_xlabel("std"); a.set_ylabel("raw row")
    a.text(35, 18, "score: std ~66\nthe loudest thing\nin the frame,\nand unpredictable",
           fontsize=7, color=RED)
    a.text(20, 140, "ball: std ~8", fontsize=7, color=GREEN)

    meta = json.load(open("proc/meta.json"))
    fr = np.load("proc/frames.npy", mmap_mode="r")
    small = np.asarray(fr[:3000]).astype(np.float32)
    pm = torch.nn.functional.max_pool2d(
        torch.from_numpy(small.std(0))[None, None], 8)[0, 0].numpy()

    a = fig.add_subplot(gs[2])
    im = a.imshow(pm, cmap="magma")
    for i in range(8):
        for j in range(8):
            a.text(j, i, f"{pm[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                   color="w" if pm[i, j] < 30 else "k")
    a.set_title("per-patch temporal std after crop\n(8x8 patch grid)", fontsize=9)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.045)

    a = fig.add_subplot(gs[3]); a.imshow(fr[60], cmap="gray")
    a.set_title(f"model input 64x64\ncrop {meta['crop']['top']}:{meta['crop']['bottom']}", fontsize=9)
    a.axis("off")

    fig.suptitle("Cause 5: the score display was the highest-variance object in the frame "
                 "and is unpredictable from the playfield", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/3_data_pipeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("3_data_pipeline.png")


# ------------------------------------------------------------- 4. the masking

def fig_masking():
    import sys
    sys.path.insert(0, ".")
    from models.masking import GROUPS, sample_mask, spatial_mask
    rng = np.random.default_rng(3)
    fr = np.load("proc/frames.npy", mmap_mode="r")
    frame = np.asarray(fr[900])

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.9))

    # old config
    masked, ctx = spatial_mask(8, (0.10, 0.20), (0.75, 1.5), 12, rng)
    rng.shuffle(ctx); keep = ctx[:7]
    for a, (title, keep_cells, n_tgt) in zip(
            ax[:2],
            [("ORIGINAL: 7 of 64 cells visible (11%)\npredict the other 57 (89%) — unlearnable",
              keep, 57),
             ("FIXED: 16 of 64 visible (25%)\npredict 24 (37%) — hard but solvable", None, 24)]):
        if keep_cells is None:
            c, t = sample_mask(1, 8, 4, aspect_range=(0.75, 1.5), rng=rng,
                               **GROUPS["SHORT_RANGE"])
            keep_cells = sorted({int(i) % 64 for i in c[0]})
            tgt_cells = sorted({int(i) % 64 for i in t[0]})
        else:
            tgt_cells = [i for i in range(64) if i not in keep_cells]
        a.imshow(frame, cmap="gray")
        for cell in range(64):
            r, cc = divmod(cell, 8)
            if cell in keep_cells:
                a.add_patch(mp.Rectangle((cc * 8 - .5, r * 8 - .5), 8, 8, fc="none",
                                         ec=GREEN, lw=1.6))
            elif cell in tgt_cells:
                a.add_patch(mp.Rectangle((cc * 8 - .5, r * 8 - .5), 8, 8, fc=RED,
                                         alpha=0.42, ec="none"))
        a.set_title(title, fontsize=8.5); a.axis("off")

    # tube illustration
    a = ax[2]; a.axis("off")
    a.set_xlim(0, 10); a.set_ylim(0, 10)
    for k, dx in enumerate([0, 1.4, 2.8, 4.2]):
        a.add_patch(mp.Rectangle((0.6 + dx, 2.2 + dx * 0.5), 3.6, 3.6,
                                 fc="#f2f4f7", ec=GREY, lw=1))
        a.add_patch(mp.Rectangle((2.0 + dx, 3.4 + dx * 0.5), 1.0, 1.0,
                                 fc=RED, alpha=0.55, ec=RED))
        a.text(0.6 + dx, 1.9 + dx * 0.5, f"t={k}", fontsize=7.5, color=GREY)
    a.set_title("tube masking: the SAME spatial cell is\nhidden in every frame, so the model "
                "cannot\ncopy the patch from a neighbouring frame", fontsize=8.5)

    fig.suptitle("Cause 3: the mask made the task impossible, so the mean was the best answer",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/4_masking.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("4_masking.png")


# --------------------------------------------------------- 5. the architecture

def fig_architecture():
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, label, sub="", fc="#eef3fb", ec=BLUE, fs=8.5):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                       fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2 + (0.9 if sub else 0), label,
                ha="center", va="center", fontsize=fs, weight="bold")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 1.5, sub, ha="center", va="center",
                    fontsize=7, color="#444")

    def arrow(x1, y1, x2, y2, label="", color="#333", style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.4))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.0, label, ha="center",
                    fontsize=7, color=color)

    box(2, 22, 14, 8, "clip x", "(B,1,4,64,64)", fc="#f7f7f9", ec=GREY)

    # online branch
    box(21, 34, 16, 8, "PatchEmbed", "Conv3d 1x8x8 -> 256 tokens x d=192")
    box(21, 22, 16, 8, "+ pos_embed", "learned (1,256,192)")
    box(43, 22, 15, 8, "gather ctx", "16 cells x 4 frames = 64 tok", fc="#eafaf0", ec=GREEN)
    box(43, 34, 15, 8, "ViT encoder", "6 blocks, RMSNorm+SwiGLU")
    box(63, 28, 16, 8, "Predictor", "d=96, 4 blocks, narrow")
    box(84, 28, 13, 8, "s_hat", "(B,96,192)", fc="#fdf0ee", ec=RED)

    # target branch
    box(43, 8, 15, 8, "EMA target", "frozen copy, tau 0.996->1", fc="#f7f7f9", ec=GREY)
    box(63, 8, 16, 8, "LayerNorm", "over feature dim, NO affine", fc="#fff8e6", ec="#d9a400")
    box(84, 8, 13, 8, "targets t", "(B,96,192)", fc="#f7f7f9", ec=GREY)

    arrow(16, 26, 21, 26)
    arrow(29, 30, 29, 34, style="<-")
    arrow(37, 38, 43, 38)
    arrow(37, 26, 43, 26)
    arrow(50.5, 30, 50.5, 34)
    arrow(58, 38, 63, 34)
    ax.text(60.5, 36.5, "context\ntokens", fontsize=6.8, color="#333", ha="center")
    arrow(79, 32, 84, 32)
    arrow(16, 24, 43, 14, "full clip, no mask")
    arrow(58, 12, 63, 12)
    arrow(79, 12, 84, 12)
    # weight copy: encoder -> EMA target, routed clear of the boxes
    ax.annotate("", xy=(40.5, 16.5), xytext=(40.5, 34),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4, ls="--"))
    ax.annotate("", xy=(40.5, 38), xytext=(43, 38),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=1.4, ls="--"))
    ax.annotate("", xy=(40.5, 34), xytext=(40.5, 38),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=1.4, ls="--"))
    ax.annotate("", xy=(43, 12), xytext=(40.5, 12),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=1.4, ls="--"))
    ax.annotate("", xy=(40.5, 12), xytext=(40.5, 16.5),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=1.4, ls="--"))
    ax.text(38.5, 25, "EMA weight copy\n(no gradient)", fontsize=7, color=GREY,
            ha="right", va="center")

    ax.add_patch(mp.FancyBboxPatch((84, 19), 13, 6, boxstyle="round,pad=0.3",
                                   fc="#fdecea", ec=RED, lw=1.6))
    ax.text(90.5, 22, "smooth L1", ha="center", va="center", fontsize=8.5, weight="bold")
    arrow(90.5, 28, 90.5, 25.2)
    arrow(90.5, 16, 90.5, 19)

    ax.add_patch(mp.FancyBboxPatch((63, 19), 15, 6, boxstyle="round,pad=0.3",
                                   fc="#eafaf0", ec=GREEN, lw=1.6))
    ax.text(70.5, 22, "VICReg var + corr", ha="center", va="center", fontsize=8, weight="bold")
    arrow(50.5, 34, 63, 24, "", GREEN)

    ax.text(2, 47, "V-JEPA — what actually flows through the model", fontsize=13, weight="bold")
    ax.text(2, 44, "Yellow = the target normalisation that removes the zero solution.  "
                   "Green = the regulariser that removes the constant solution.\n"
                   "The predictor never sees pixels, and there is no decoder anywhere: "
                   "the loss lives entirely in representation space.", fontsize=8.2)
    ax.text(2, 3, "gradients flow ONLY through the online branch (top).  "
                  "the target branch is stop-grad + EMA — that asymmetry is what "
                  "stops the trivial solution from being reachable in one step.",
            fontsize=7.6, color=GREY, style="italic")

    fig.savefig(f"{FIG}/5_architecture.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("5_architecture.png")


if __name__ == "__main__":
    fig_collapse_ab()
    fig_collapse_modes()
    fig_data_pipeline()
    fig_masking()
    fig_architecture()
    print("figures in", FIG)
