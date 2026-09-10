import argparse
import os
import time

VERDICT_OK = "\033[32m"
VERDICT_BAD = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"

COLS = [
    ("step", "step", "{:>6}"),
    ("epoch", "ep", "{:>3}"),
    ("task", "task", "{:>4}"),
    ("loss_pred", "pred", "{:>7.4f}"),
    ("loss_var", "var", "{:>7.4f}"),
    ("loss_cov", "cov", "{:>7.4f}"),
    ("tgt_cos", "cos", "{:>+7.3f}"),
    ("tgt_eff_rank", "rank", "{:>6.1f}"),
    ("tgt_rel_std", "relstd", "{:>7.4f}"),
    ("lr", "lr", "{:>9.2e}"),
]


def read_last(path):
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def fmt(row):
    row = dict(row, task="cau" if row.get("causal") == "1" else "blk")
    out = []
    for key, label, spec in COLS:
        v = row.get(key, "")
        try:
            v = spec.format(float(v) if "." in spec or "e" in spec else int(float(v)))
        except (ValueError, TypeError):
            v = f"{str(v):>7}"
        out.append(f"{DIM}{label}{RESET} {v}")
    return "  ".join(out)


def health(row):
    try:
        cos = float(row["tgt_cos"])
        rank = float(row["tgt_eff_rank"])
        rel = float(row["tgt_rel_std"])
    except (KeyError, ValueError, TypeError):
        return DIM + "?" + RESET
    if cos > 0.98 or rank < 3 or rel < 0.01:
        return VERDICT_BAD + "COLLAPSED" + RESET
    if cos > 0.7:
        return VERDICT_BAD + "collapsing" + RESET
    return VERDICT_OK + "healthy" + RESET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="runs/vjepa")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=5.0)
    args = ap.parse_args()

    path = os.path.join(args.run, "metrics.csv")
    print(f"watching {path}   (ctrl-C to stop; this does not affect training)\n")
    print(f"{DIM}collapse is 'cos' climbing to 1.0 and 'rank' falling -- "
          f"NOT 'pred' going down{RESET}")
    print(f"{DIM}task blk/cau = block mask vs causal mask; they sit at "
          f"different loss levels by design{RESET}\n")

    last_step = None
    while True:
        if not os.path.exists(path):
            print("waiting for the run to start...", end="\r")
            time.sleep(args.every)
            continue
        row = read_last(path)
        if row and row.get("step") != last_step:
            last_step = row.get("step")
            print(f"{fmt(row)}   {health(row)}")
        if args.once:
            break
        if os.path.exists(os.path.join(args.run, "COLLAPSED.txt")):
            print(VERDICT_BAD + "\nrun aborted: see " +
                  os.path.join(args.run, "COLLAPSED.txt") + RESET)
            break
        # normal completion: final.pt is written once the loop is done.  Without
        # this the watcher spins forever on a finished run and looks like
        # training is still going.
        if os.path.exists(os.path.join(args.run, "checkpoint", "final.pt")):
            print(VERDICT_OK + "\nTRAINING FINISHED" + RESET +
                  f"  ({last_step} steps)\n\nnext:\n"
                  f"  python -m eval.probe --ckpt {args.run}/checkpoint/final.pt\n"
                  f"  python visualize.py  --ckpt {args.run}/checkpoint/final.pt")
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
