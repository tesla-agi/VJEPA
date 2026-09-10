import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from data.dataset import ClipDataset
from eval.probe import extract, fit_probe, load_encoder
from models.masking import GROUPS, sample_causal_mask, sample_mask
from models.predictor import Predictor


def to_img(frame_u8):
    a = np.asarray(frame_u8, dtype=np.float32) / 255.0
    return np.stack([a, a, a], -1)


def patch_grid_overlay(ax, grid_size, img_size=64, color="w", lw=0.3):
    step = img_size / grid_size
    for i in range(1, grid_size):
        ax.axhline(i * step - 0.5, color=color, lw=lw, alpha=0.25)
        ax.axvline(i * step - 0.5, color=color, lw=lw, alpha=0.25)


def cell_to_extent(cell, grid_size, img_size=64):
    r, c = divmod(cell, grid_size)
    s = img_size / grid_size
    return c * s - 0.5, r * s - 0.5, s, s


def mark_ball(ax, lab, color="lime", label=None):
    if lab[0] < 0.5:
        return
    ax.add_patch(mpatches.Circle((lab[2], lab[1]), 4.0, fill=False,
                                 ec=color, lw=1.8, label=label))


def upsample_patch_map(vals, grid_size, img_size=64):
    return np.kron(vals.reshape(grid_size, grid_size),
                   np.ones((img_size // grid_size, img_size // grid_size)))


def load_predictor(ckpt_path, device, enc):
    state = torch.load(ckpt_path, map_location=device)
    a = state.get("args", {})
    pr = Predictor(d_model=a.get("d_model", 192), d_pred=a.get("d_pred", 96),
                   depth=a.get("pred_depth", 4), n_patches=enc.n_tokens,
                   norm_type=a.get("norm_type", "rms_norm"),
                   ffn_type=a.get("ffn_type", "swiglu")).to(device)
    pr.load_state_dict(state["predictor"])
    pr.eval()
    return pr

def fig_masks(ds, ball, enc, out, clip_i=0, seed=0):
    clip, s = ds.clip(clip_i)
    rng = np.random.default_rng(seed)
    ctx, tgt = sample_mask(1, enc.grid_size, enc.n_frames,
                           aspect_range=(0.75, 1.5), rng=rng,
                           **GROUPS["SHORT_RANGE"])
    g, ns = enc.grid_size, enc.n_spatial
    ctx_cells = sorted({int(i) % ns for i in ctx[0]})
    tgt_cells = sorted({int(i) % ns for i in tgt[0]})

    fig, axes = plt.subplots(2, enc.n_frames, figsize=(3.1 * enc.n_frames, 6.6))
    for t in range(enc.n_frames):
        ax = axes[0, t]
        ax.imshow(to_img(clip[t]))
        mark_ball(ax, ball[s + t])
        patch_grid_overlay(ax, g)
        ax.set_title(f"real frame t={t}", fontsize=9)
        ax.axis("off")

        ax = axes[1, t]
        ax.imshow(to_img(clip[t]))
        for cell in range(ns):
            x, y, w, h = cell_to_extent(cell, g)
            if cell in ctx_cells:
                ax.add_patch(mpatches.Rectangle((x, y), w, h, fc="none",
                                                ec="#35d07f", lw=1.4))
            elif cell in tgt_cells:
                ax.add_patch(mpatches.Rectangle((x, y), w, h, fc="#e8543f",
                                                alpha=0.45, ec="none"))
        mark_ball(ax, ball[s + t])
        ax.set_title("green = context seen · red = must predict", fontsize=8)
        ax.axis("off")
    fig.suptitle("1. tube masking on a real clip (ball circled)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


@torch.no_grad()
def fig_collapse(enc, ds, device, out, n_clips=64):
    step = max(len(ds) // n_clips, 1)
    x = torch.stack([torch.from_numpy(ds.clip(i * step)[0]).float().unsqueeze(0)
                     for i in range(n_clips)]).to(device)
    x = (x - ds.sub) / ds.div
    z = enc(x).flatten(0, 1).float().cpu()

    u = torch.nn.functional.normalize(z, dim=-1)
    sel = torch.randperm(len(u))[:4000]
    u = u[sel]
    cos = (u[:2000] * u[2000:4000]).sum(-1).numpy()
    sv = torch.linalg.svdvals(z - z.mean(0, keepdim=True)).numpy()
    p = sv / sv.sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(cos, bins=80, color="#3b7dd8")
    axes[0].axvline(1.0, color="crimson", ls="--", lw=1)
    axes[0].set_title(f"pairwise token cosine  (mean {cos.mean():+.3f})\n"
                      "collapsed = a single spike at 1.0", fontsize=10)
    axes[0].set_xlim(-1.05, 1.05)
    axes[0].set_xlabel("cosine similarity")

    axes[1].semilogy(sv / sv[0], lw=1.5, color="#3b7dd8")
    axes[1].set_title(f"singular spectrum  (effective rank {eff_rank:.1f} / {z.shape[1]})\n"
                      "collapsed = falls off a cliff after component 1", fontsize=10)
    axes[1].set_xlabel("component"); axes[1].grid(alpha=0.25)

    verdict = ("HEALTHY" if cos.mean() < 0.9 and eff_rank > 5 else "COLLAPSED")
    fig.suptitle(f"2. collapse check — {verdict}", fontsize=13,
                 color="#1a7f37" if verdict == "HEALTHY" else "crimson")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out, f"[{verdict}] cos {cos.mean():+.3f} eff_rank {eff_rank:.1f}")
    return {"cos": float(cos.mean()), "eff_rank": eff_rank, "verdict": verdict}


@torch.no_grad()
def fig_features(enc, ds, ball, device, out, clip_i=0, fit_clips=96):
    """PCA of patch tokens -> RGB, painted back onto the real frame."""
    step = max(len(ds) // fit_clips, 1)
    xs = torch.stack([torch.from_numpy(ds.clip(i * step)[0]).float().unsqueeze(0)
                      for i in range(fit_clips)]).to(device)
    xs = (xs - ds.sub) / ds.div
    z = enc(xs).flatten(0, 1).float().cpu()
    mu = z.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(z - mu, q=3)

    clip, s = ds.clip(clip_i)
    x = torch.from_numpy(clip).float().unsqueeze(0).unsqueeze(0).to(device)
    x = (x - ds.sub) / ds.div
    tok = enc(x)[0].float().cpu()
    proj = (tok - mu) @ V                                  # (T*ns, 3)
    lo, hi = proj.min(0).values, proj.max(0).values
    rgb = ((proj - lo) / (hi - lo + 1e-8)).numpy()

    g, ns = enc.grid_size, enc.n_spatial
    fig, axes = plt.subplots(2, enc.n_frames, figsize=(3.1 * enc.n_frames, 7.0))
    for t in range(enc.n_frames):
        axes[0, t].imshow(to_img(clip[t]))
        mark_ball(axes[0, t], ball[s + t])
        axes[0, t].set_title(f"real frame t={t}", fontsize=9)
        axes[0, t].axis("off")

        img = rgb[t * ns:(t + 1) * ns].reshape(g, g, 3)
        img = np.kron(img, np.ones((8, 8, 1)))
        axes[1, t].imshow(img)
        mark_ball(axes[1, t], ball[s + t], color="k")
        axes[1, t].set_title("encoder features (PCA->RGB)", fontsize=8)
        axes[1, t].axis("off")
    fig.suptitle("3. what the encoder represents — top-3 PCA of the patch tokens, "
                 "painted back onto the frame\n"
                 "a collapsed encoder paints this one flat colour",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.2)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


@torch.no_grad()
def fig_pred_error(enc, pred, ds, ball, device, out, clip_i=0, seed=0):
    import torch.nn.functional as F
    clip, s = ds.clip(clip_i)
    x = torch.from_numpy(clip).float().unsqueeze(0).unsqueeze(0).to(device)
    x = (x - ds.sub) / ds.div
    rng = np.random.default_rng(seed)
    ctx, tgt = sample_mask(1, enc.grid_size, enc.n_frames,
                           aspect_range=(0.75, 1.5), rng=rng,
                           **GROUPS["SHORT_RANGE"])
    ctx, tgt = ctx.to(device), tgt.to(device)
    y = enc(x)
    s_hat = pred(enc(x, ctx), ctx, tgt)
    t_true = torch.gather(y, 1, tgt.unsqueeze(-1).expand(-1, -1, y.shape[-1]))
    t_true = F.layer_norm(t_true, (t_true.shape[-1],))
    cos = F.cosine_similarity(s_hat, t_true, dim=-1)[0].float().cpu().numpy()

    g, ns = enc.grid_size, enc.n_spatial
    fig, axes = plt.subplots(1, enc.n_frames, figsize=(3.3 * enc.n_frames, 3.9))
    for t in range(enc.n_frames):
        m = np.full(ns, np.nan)
        for j, tok_id in enumerate(tgt[0].cpu().numpy()):
            if tok_id // ns == t:
                m[tok_id % ns] = cos[j]
        axes[t].imshow(to_img(clip[t]))
        im = axes[t].imshow(upsample_patch_map(m, g), cmap="RdYlGn",
                            vmin=-1, vmax=1, alpha=0.72)
        mark_ball(axes[t], ball[s + t], color="cyan")
        axes[t].set_title(f"t={t}", fontsize=9)
        axes[t].axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, label="cos(predicted, true)")
    fig.suptitle("4. predictor accuracy per masked patch  (green = predicted well)",
                 fontsize=12)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, f"mean cos {np.nanmean(cos):+.3f}")


@torch.no_grad()
def extract_imagined(enc, pred, ds, ball, device, idx, batch=64):
    """Predictor's imagined tokens for the hidden last frame, + its true ball.

    A probe fitted on *real* encodings does not transfer to *predicted* ones:
    the predictor is 0.98 cosine-accurate, but a 12288-dimensional ridge map
    amplifies that last 2% into tens of pixels.  So the readout for the rollout
    panel is fitted in the space it is actually applied in.
    """
    X, Y = [], []
    for i0 in range(0, len(idx), batch):
        xs, labels = [], []
        for i in idx[i0:i0 + batch]:
            clip, s = ds.clip(int(i))
            last = s + ds.T - 1
            if ball[last, 0] < 0.5:
                continue
            xs.append(torch.from_numpy(clip).float().unsqueeze(0))
            labels.append(ball[last, 1:])
        if not xs:
            continue
        x = ((torch.stack(xs).to(device)) - ds.sub) / ds.div
        ctx, tgt = sample_causal_mask(len(xs), enc.grid_size, enc.n_frames,
                                      n_ctx_frames=enc.n_frames - 1)
        ctx, tgt = ctx.to(device), tgt.to(device)
        X.append(pred(enc(x, ctx), ctx, tgt).flatten(1).cpu())
        Y.append(torch.tensor(np.array(labels), dtype=torch.float32))
    return torch.cat(X), torch.cat(Y)


@torch.no_grad()
def _probe_predict(probe, mu, sd, feats, device):
    f = ((feats.cpu() - mu) / sd).to(device)
    return probe(f).cpu().numpy()


@torch.no_grad()
def fig_ball_track(enc, probe, mu, sd, ds, ball, device, out, gif_out,
                   start_clip=0, n=24):
    """Read the ball position out of the frozen features of real frames."""
    clips, labs, feats = [], [], []
    i = start_clip
    got = 0
    while got < n and i < len(ds):
        clip, s = ds.clip(i)
        last = s + ds.T - 1
        if ball[last, 0] > 0.5:
            clips.append(clip[-1]); labs.append(ball[last, 1:])
            x = torch.from_numpy(clip).float().unsqueeze(0).unsqueeze(0).to(device)
            tok = enc((x - ds.sub) / ds.div)[:, -enc.n_spatial:]
            tok = torch.nn.functional.layer_norm(tok, (tok.shape[-1],))
            feats.append(tok.flatten(1))
            got += 1
        i += 1
    F_ = torch.cat(feats)
    pred = _probe_predict(probe, mu, sd, F_, device)
    labs = np.array(labs)
    err = np.abs(pred - labs).mean()

    cols = 8
    rows = int(np.ceil(len(clips) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.85 * rows))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        if k >= len(clips):
            ax.axis("off"); continue
        ax.imshow(to_img(clips[k]))
        ax.add_patch(mpatches.Circle((labs[k][1], labs[k][0]), 4.0, fill=False,
                                     ec="lime", lw=1.6))
        ax.add_patch(mpatches.Circle((pred[k][1], pred[k][0]), 3.0, fill=False,
                                     ec="red", lw=1.6, ls="--"))
        ax.axis("off")
    fig.suptitle(f"5. the ball, read out of frozen V-JEPA features\n"
                 f"green = true position · red dashed = linear probe on features "
                 f"· mean error {err:.2f} px",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out, f"mean px error {err:.2f}")

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        f2, a2 = plt.subplots(figsize=(4.2, 4.4))
        a2.axis("off")
        im = a2.imshow(to_img(clips[0]))
        c_true = mpatches.Circle((0, 0), 4.0, fill=False, ec="lime", lw=2)
        c_pred = mpatches.Circle((0, 0), 3.0, fill=False, ec="red", lw=2, ls="--")
        a2.add_patch(c_true); a2.add_patch(c_pred)
        ttl = a2.set_title("")

        def upd(k):
            im.set_data(to_img(clips[k]))
            c_true.center = (labs[k][1], labs[k][0])
            c_pred.center = (pred[k][1], pred[k][0])
            ttl.set_text(f"frame {k}   true (green) vs read-out (red)")
            return im, c_true, c_pred, ttl

        FuncAnimation(f2, upd, frames=len(clips), blit=False).save(
            gif_out, writer=PillowWriter(fps=4))
        plt.close(f2)
        print("wrote", gif_out)
    except Exception as e:
        print("gif skipped:", e)
    return err


@torch.no_grad()
def fig_rollout(enc, pred, probe, mu, sd, ds, ball, device, out,
                start_clip=0, n=8):
    """Hide the last frame, imagine its tokens, read the ball out of them."""
    rows = []
    i, got = start_clip, 0
    while got < n and i < len(ds):
        clip, s = ds.clip(i)
        last = s + ds.T - 1
        if ball[last, 0] > 0.5 and ball[s + ds.T - 2, 0] > 0.5:
            rows.append((clip, s)); got += 1
        i += 1

    x = torch.stack([torch.from_numpy(c).float().unsqueeze(0) for c, _ in rows]).to(device)
    x = (x - ds.sub) / ds.div
    ctx, tgt = sample_causal_mask(len(rows), enc.grid_size, enc.n_frames,
                                  n_ctx_frames=enc.n_frames - 1)
    ctx, tgt = ctx.to(device), tgt.to(device)
    imagined = pred(enc(x, ctx), ctx, tgt)          # (B, n_spatial, d) for last frame
    p_imag = _probe_predict(probe, mu, sd, imagined.flatten(1), device)

    # how good is the prediction in the space the model was actually trained in?
    y_full = enc(x)
    true = torch.nn.functional.layer_norm(
        torch.gather(y_full, 1, tgt.unsqueeze(-1).expand(-1, -1, y_full.shape[-1])),
        (y_full.shape[-1],))
    tok_cos = torch.nn.functional.cosine_similarity(imagined, true, dim=-1).mean().item()

    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.8))
    for k, (clip, s) in enumerate(rows):
        axes[0, k].imshow(to_img(clip[-2]))
        mark_ball(axes[0, k], ball[s + ds.T - 2])
        axes[0, k].set_title("last seen frame", fontsize=7)
        axes[0, k].axis("off")

        ax = axes[1, k]
        ax.imshow(to_img(clip[-1]))
        lab = ball[s + ds.T - 1]
        ax.add_patch(mpatches.Circle((lab[2], lab[1]), 4.0, fill=False, ec="lime", lw=1.6))
        ax.add_patch(mpatches.Circle((p_imag[k][1], p_imag[k][0]), 3.0, fill=False,
                                     ec="orange", lw=1.8, ls="--"))
        ax.set_title("hidden frame: true vs imagined", fontsize=7)
        ax.axis("off")
    err = np.abs(p_imag - np.array([ball[s + ds.T - 1, 1:] for _, s in rows])).mean()
    fig.suptitle(f"6. world-model rollout — the last frame is never shown to the encoder.\n"
                 f"green = where the ball really was · orange = read out of the "
                 f"predictor's imagined tokens · mean error {err:.2f} px\n"
                 f"imagined vs true tokens: cosine {tok_cos:+.4f}  "
                 f"(this is what the model was trained to optimise)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out, f"imagined-token ball error {err:.2f} px  "
          f"| imagined-vs-true token cosine {tok_cos:+.4f}")
    return err, tok_cos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/vjepa/checkpoint/final.pt")
    ap.add_argument("--proc-dir", default="proc")
    ap.add_argument("--out", default="")
    ap.add_argument("--which", default="target", choices=["target", "online"])
    ap.add_argument("--clip", type=int, default=120)
    ap.add_argument("--n-probe", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(args.ckpt)), "viz")
    os.makedirs(out_dir, exist_ok=True)

    ds = ClipDataset(proc_dir=args.proc_dir)
    ball = np.load(os.path.join(args.proc_dir, "ball.npy"))
    enc = load_encoder(args.ckpt, device, args.which)
    pred = load_predictor(args.ckpt, device, enc)
    print(f"device {device} · clips {len(ds)} · out {out_dir}")

    fig_masks(ds, ball, enc, os.path.join(out_dir, "1_masks.png"), args.clip, args.seed)
    fig_collapse(enc, ds, device, os.path.join(out_dir, "2_collapse.png"))
    fig_features(enc, ds, ball, device, os.path.join(out_dir, "3_features.png"), args.clip)
    fig_pred_error(enc, pred, ds, ball, device,
                   os.path.join(out_dir, "4_pred_error.png"), args.clip, args.seed)

    # probe for panels 5 and 6
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), min(args.n_probe, len(ds)), replace=False)
    split = int(0.85 * len(idx))
    # ln=True: the probe then lives in the same space the predictor outputs,
    # so panel 6 can apply it to imagined tokens
    Xtr, Ytr = extract(enc, ds, ball, device, idx[:split], ln=True)
    Xte, Yte = extract(enc, ds, ball, device, idx[split:], ln=True)
    probe, mu, sd, m = fit_probe(Xtr, Ytr, Xte, Yte, device=device)
    print(f"probe R^2 {m['r2']:.3f}  MAE {m['mae_y_px']:.2f}/{m['mae_x_px']:.2f} px")

    fig_ball_track(enc, probe, mu, sd, ds, ball, device,
                   os.path.join(out_dir, "5_ball_track.png"),
                   os.path.join(out_dir, "ball_track.gif"), args.clip)

    Itr, Jtr = extract_imagined(enc, pred, ds, ball, device, idx[:split])
    Ite, Jte = extract_imagined(enc, pred, ds, ball, device, idx[split:])
    iprobe, imu, isd, im = fit_probe(Itr, Jtr, Ite, Jte, device=device)
    print(f"rollout probe (fitted on imagined tokens) R^2 {im['r2']:.3f}  "
          f"MAE {im['mae_y_px']:.2f}/{im['mae_x_px']:.2f} px")
    fig_rollout(enc, pred, iprobe, imu, isd, ds, ball, device,
                os.path.join(out_dir, "6_rollout.png"), args.clip)
    print("\nall figures in", out_dir)


if __name__ == "__main__":
    main()
