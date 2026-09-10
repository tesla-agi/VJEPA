import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from data.dataset import ClipDataset
from models.vit import ViT


def load_encoder(ckpt_path, device, which="target"):
    state = torch.load(ckpt_path, map_location=device)
    a = state.get("args", {})
    enc = ViT(d_model=a.get("d_model", 192), num_layers=a.get("depth", 6),
              norm_type=a.get("norm_type", "rms_norm"),
              ffn_type=a.get("ffn_type", "swiglu")).to(device)
    enc.load_state_dict(state[which if which in state else "online"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.no_grad()
def extract(enc, ds, ball, device, idx, batch=64, mode="tokens", ln=False):
    """Features for the *last* frame of each clip, plus its ball label."""
    X, Y = [], []
    for i0 in range(0, len(idx), batch):
        sel = idx[i0:i0 + batch]
        xs, labels = [], []
        for i in sel:
            clip, s = ds.clip(int(i))
            last = s + ds.T - 1
            if ball[last, 0] < 0.5:                 # ball not on screen
                continue
            xs.append(torch.from_numpy(clip).float().unsqueeze(0))
            labels.append(ball[last, 1:])
        if not xs:
            continue
        x = torch.stack(xs).to(device)
        x = (x - ds.sub) / ds.div
        if mode == "pixels":
            f = x[:, 0, -1].flatten(1)              # last frame, raw pixels
        else:
            tok = enc(x)                            # (B, T*n_spatial, d)
            tok = tok[:, -enc.n_spatial:]           # last frame's patch tokens
            if ln:
                # match the space the predictor is trained to output, so the
                # same probe can be applied to imagined tokens (visualize.py)
                tok = torch.nn.functional.layer_norm(tok, (tok.shape[-1],))
            f = tok.flatten(1)
        X.append(f.cpu())
        Y.append(torch.tensor(np.array(labels), dtype=torch.float32))
    return torch.cat(X), torch.cat(Y)


def _ridge_dual(Xtr, Ytr, alpha):
    n = Xtr.shape[0]
    G = Xtr @ Xtr.T
    G.diagonal().add_(alpha)
    a = torch.linalg.solve(G, Ytr)
    return Xtr.T @ a


def _r2(pred, Y, base):
    ss_res = ((pred - Y) ** 2).sum(0)
    ss_tot = ((Y - base) ** 2).sum(0)
    return 1 - ss_res / ss_tot, ss_res, ss_tot


def fit_probe(Xtr, Ytr, Xte, Yte, alphas=(1e1, 1e2, 1e3, 1e4, 1e5, 1e6),
              val_frac=0.2, device="cpu", verbose=False):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).double()
    Xte = ((Xte - mu) / sd).double()
    Ytr, Yte = Ytr.double(), Yte.double()

    ym = Ytr.mean(0, keepdim=True)
    n_val = max(int(val_frac * len(Xtr)), 1)
    Xf, Yf = Xtr[:-n_val], Ytr[:-n_val]
    Xv, Yv = Xtr[-n_val:], Ytr[-n_val:]
    yfm = Yf.mean(0, keepdim=True)

    best, best_alpha = -1e18, alphas[0]
    for alpha in alphas:
        W = _ridge_dual(Xf, Yf - yfm, alpha)
        r2, _, _ = _r2(Xv @ W + yfm, Yv, yfm)
        if verbose:
            print(f"  alpha {alpha:9.0f}  val R^2 {r2.mean().item():+.4f}")
        if r2.mean().item() > best:
            best, best_alpha = r2.mean().item(), alpha

    W = _ridge_dual(Xtr, Ytr - ym, best_alpha)
    pred = Xte @ W + ym
    r2, ss_res, ss_tot = _r2(pred, Yte, ym)
    mae = (pred - Yte).abs().mean(0)

    probe = nn.Linear(Xtr.shape[1], 2)
    with torch.no_grad():
        probe.weight.copy_(W.T.float())
        probe.bias.copy_(ym[0].float())
    probe = probe.to(device).eval()

    return probe, mu, sd, {"alpha": best_alpha,
                           "r2_y": r2[0].item(), "r2_x": r2[1].item(),
                           "r2": (1 - ss_res.sum() / ss_tot.sum()).item(),
                           "mae_y_px": mae[0].item(), "mae_x_px": mae[1].item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/vjepa/checkpoint/final.pt")
    ap.add_argument("--proc-dir", default="proc")
    ap.add_argument("--which", default="target", choices=["target", "online"])
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines", action="store_true", default=True)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    ds = ClipDataset(proc_dir=args.proc_dir)
    ball = np.load(os.path.join(args.proc_dir, "ball.npy"))
    episode = np.load(os.path.join(args.proc_dir, "episode.npy"))

    # split by episode so train and test never share a trajectory
    ep_of_clip = episode[ds.index_map]
    n_ep = ep_of_clip.max() + 1
    test_eps = set(range(int(n_ep * 0.85), int(n_ep)))
    rng = np.random.default_rng(args.seed)
    tr_pool = np.flatnonzero(~np.isin(ep_of_clip, list(test_eps)))
    te_pool = np.flatnonzero(np.isin(ep_of_clip, list(test_eps)))
    tr = rng.choice(tr_pool, min(args.n_train, len(tr_pool)), replace=False)
    te = rng.choice(te_pool, min(args.n_test, len(te_pool)), replace=False)

    enc = load_encoder(args.ckpt, device, args.which)
    results = {}

    Xtr, Ytr = extract(enc, ds, ball, device, tr)
    Xte, Yte = extract(enc, ds, ball, device, te)
    print(f"probe set: train {len(Xtr)}  test {len(Xte)}  feature dim {Xtr.shape[1]}")
    probe, mu, sd, m = fit_probe(Xtr, Ytr, Xte, Yte, device=device)
    results["vjepa"] = m
    print(f"V-JEPA ({args.which})  R^2 {m['r2']:.3f}  "
          f"(y {m['r2_y']:.3f}, x {m['r2_x']:.3f})  "
          f"MAE {m['mae_y_px']:.2f}/{m['mae_x_px']:.2f} px")

    if args.baselines:
        Ptr, _ = extract(enc, ds, ball, device, tr, mode="pixels")
        Pte, _ = extract(enc, ds, ball, device, te, mode="pixels")
        _, _, _, mp = fit_probe(Ptr, Ytr, Pte, Yte, device=device)
        results["pixels"] = mp
        print(f"raw pixels (upper bound)  R^2 {mp['r2']:.3f}  "
              f"MAE {mp['mae_y_px']:.2f}/{mp['mae_x_px']:.2f} px")
        rnd = ViT(d_model=enc.d_model, num_layers=len(enc.blocks),
                  norm_type="rms_norm", ffn_type="swiglu").to(device).eval()
        for q in rnd.parameters():
            q.requires_grad_(False)
        Rtr, _ = extract(rnd, ds, ball, device, tr)
        Rte, _ = extract(rnd, ds, ball, device, te)
        _, _, _, mr = fit_probe(Rtr, Ytr, Rte, Yte, device=device)
        results["random_init"] = mr
        print(f"random-init encoder        R^2 {mr['r2']:.3f}  "
              f"MAE {mr['mae_y_px']:.2f}/{mr['mae_x_px']:.2f} px")

        # constant predictor == what a fully collapsed encoder can achieve
        const = Ytr.mean(0, keepdim=True)
        ss_res = ((const - Yte) ** 2).sum(0)
        ss_tot = ((Yte - Ytr.mean(0)) ** 2).sum(0)
        results["constant"] = {"r2": (1 - ss_res.sum() / ss_tot.sum()).item(),
                               "mae_y_px": (const - Yte).abs().mean(0)[0].item(),
                               "mae_x_px": (const - Yte).abs().mean(0)[1].item()}
        print(f"constant (collapsed floor)  R^2 {results['constant']['r2']:.3f}  "
              f"MAE {results['constant']['mae_y_px']:.2f}/"
              f"{results['constant']['mae_x_px']:.2f} px")

    if args.save:
        torch.save({"probe": probe.state_dict(), "mu": mu, "sd": sd,
                    "results": results}, args.save)
        print("saved probe to", args.save)
    return results


if __name__ == "__main__":
    main()
