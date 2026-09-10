import argparse
import math
import os
import numpy as np
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import ClipDataset
from models.ema import build_target, tau_schedule, update_target
from models.masking import GROUPS, sample_causal_mask, sample_mask
from models.predictor import Predictor
from models.vit import ViT
from train.loss import vjepa_loss
from utils.logger import RunLogger
from utils.monitor import collapse_stats, is_collapsing


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--proc-dir", default="proc")
    p.add_argument("--run", default="runs/vjepa")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--norm-type", default="rms_norm")
    p.add_argument("--ffn-type", default="swiglu")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--pred-depth", type=int, default=4)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--d-pred", type=int, default=96)
    p.add_argument("--tau-start", type=float, default=0.996)
    p.add_argument("--lambda-var", type=float, default=1.0)
    p.add_argument("--lambda-cov", type=float, default=0.04)
    p.add_argument("--groups", default="SHORT_RANGE,LONG_RANGE")
    p.add_argument("--causal-frac", type=float, default=0.25,
                   help="fraction of steps that hide the whole last frame "
                        "and predict it from the past (world-model task)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=5, help="epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--abort-on-collapse", action="store_true", default=True)
    p.add_argument("--max-steps", type=int, default=0, help="0 = full schedule")
    return p.parse_args()


def param_groups(modules, weight_decay):
    decay, no_decay = [], []
    for m in modules:
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(("pos_embed", "pos_pred", "mask_token")):
                no_decay.append(p)
            else:
                decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def main():
    args = parse_args()
    os.makedirs(args.run, exist_ok=True)
    ckpt_dir = os.path.join(args.run, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    dataset = ClipDataset(proc_dir=args.proc_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True,
                            persistent_workers=args.num_workers > 0)

    online = ViT(d_model=args.d_model, num_layers=args.depth,
                 norm_type=args.norm_type, ffn_type=args.ffn_type).to(device)
    predictor = Predictor(d_model=args.d_model, d_pred=args.d_pred,
                          depth=args.pred_depth, n_patches=online.n_tokens,
                          norm_type=args.norm_type, ffn_type=args.ffn_type).to(device)
    target = build_target(online).to(device)

    steps_per_epoch = len(dataloader)
    total_steps = args.max_steps or args.epochs * steps_per_epoch
    warmup = min(args.warmup, max(total_steps // 10, 1))

    optimizer = AdamW(param_groups([online, predictor], args.weight_decay),
                      lr=args.lr, betas=(0.9, 0.95))

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        progress = (s - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    groups = [GROUPS[g] for g in args.groups.split(",")]
    rng = np.random.default_rng(args.seed)

    start_epoch, step = 0, 0
    last = os.path.join(ckpt_dir, "last.pt")
    if args.resume and os.path.exists(last):
        state = torch.load(last, map_location=device)
        online.load_state_dict(state["online"])
        target.load_state_dict(state["target"])
        predictor.load_state_dict(state["predictor"])
        optimizer.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["sched"])
        start_epoch, step = state["epoch"] + 1, state["step"]
        print(f"resumed from {last} at epoch {start_epoch} step {step}")

    logger = RunLogger(args.run, config={**vars(args), "device": device,
                                         "clips": len(dataset),
                                         "steps_per_epoch": steps_per_epoch,
                                         "total_steps": total_steps},
                       resume=args.resume)

    print(f"device {device} · clips {len(dataset)} · {steps_per_epoch} steps/epoch "
          f"· {total_steps} total · groups {args.groups}")

    online.train(); predictor.train(); target.eval()
    stop = False

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for x in pbar:
            x = x.to(device)
            B = x.shape[0]
            causal = rng.random() < args.causal_frac
            if causal:
                ctx_idx, tgt_idx = sample_causal_mask(
                    B, online.grid_size, online.n_frames,
                    n_ctx_frames=online.n_frames - 1)
            else:
                group = groups[step % len(groups)]
                ctx_idx, tgt_idx = sample_mask(B, grid_size=online.grid_size,
                                               n_frames=online.n_frames,
                                               aspect_range=(0.75, 1.5),
                                               rng=rng, **group)
            ctx_idx, tgt_idx = ctx_idx.to(device), tgt_idx.to(device)

            with torch.no_grad():
                s_y_all = target(x)
            s_c = online(x, ctx_idx)
            s_hat = predictor(s_c, ctx_idx, tgt_idx)

            loss, parts = vjepa_loss(s_hat, s_y_all, tgt_idx, s_ctx=s_c,
                                     lambda_var=args.lambda_var,
                                     lambda_cov=args.lambda_cov)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                list(online.parameters()) + list(predictor.parameters()),
                args.clip_grad)
            optimizer.step()
            sched.step()

            tau = tau_schedule(step, total_steps, args.tau_start)
            update_target(target, online, tau)

            if step % args.log_every == 0:
                ts = collapse_stats(s_y_all)
                cs = collapse_stats(s_c)
                row = {"step": step, "epoch": epoch, "causal": int(causal),
                       "lr": optimizer.param_groups[0]["lr"], "tau": tau,
                       "grad_norm": gnorm.item(),
                       "loss_total": parts["total"].item(),
                       "loss_pred": parts["pred"].item(),
                       "loss_var": parts.get("var", torch.zeros(())).item(),
                       "loss_cov": parts.get("cov", torch.zeros(())).item(),
                       **{f"tgt_{k}": v for k, v in ts.items()},
                       **{f"ctx_{k}": v for k, v in cs.items()}}
                logger.log(row)
                pbar.set_postfix(pred=f"{row['loss_pred']:.4f}",
                                 cos=f"{ts['cos']:+.3f}",
                                 rank=f"{ts['eff_rank']:.1f}")

                reasons = is_collapsing(ts)
                if reasons and step > 300:
                    msg = "COLLAPSE: " + "; ".join(reasons)
                    tqdm.write(msg)
                    if args.abort_on_collapse:
                        with open(os.path.join(args.run, "COLLAPSED.txt"), "w") as f:
                            f.write(f"step {step}\n{msg}\n")
                        stop = True
                        break
            step += 1
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

        torch.save({"online": online.state_dict(), "target": target.state_dict(),
                    "predictor": predictor.state_dict(),
                    "optimizer": optimizer.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "step": step, "args": vars(args)}, last)
        if (epoch + 1) % args.save_every == 0:
            torch.save({"online": online.state_dict(), "target": target.state_dict(),
                        "predictor": predictor.state_dict(), "epoch": epoch,
                        "step": step, "args": vars(args)},
                       os.path.join(ckpt_dir, f"epoch{epoch + 1:03d}.pt"))
            tqdm.write(f"checkpoint saved · epoch {epoch + 1} · step {step}")
        if stop:
            break

    torch.save({"online": online.state_dict(), "target": target.state_dict(),
                "predictor": predictor.state_dict(), "step": step,
                "args": vars(args)}, os.path.join(ckpt_dir, "final.pt"))
    logger.close()
    print(f"done · step {step} · {args.run}")

    try:
        from utils.plot_run import main as plot
        plot(args.run)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
