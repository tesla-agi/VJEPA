# V-JEPA on Atari Breakout

A small V-JEPA (self-supervised video joint-embedding predictive architecture)
trained on random Breakout rollouts, end to end: collect -> preprocess -> train
-> monitor -> probe -> visualise.

See **[NOTES.md](NOTES.md)** for the collapse diagnosis and what was changed, and
**[docs/report.md](docs/report.md)** (or `docs/VJEPA_report.pdf`) for the write-up
with figures.

## Setup

No pinned lockfile; the dependencies are
`torch`, `numpy`, `gymnasium`, `ale-py`, `matplotlib`, `pillow`, `tqdm`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy gymnasium ale-py matplotlib pillow tqdm
```

Every command below is run from the repo root, so `python -m <module>` resolves.

## Layout

```
data/
  collect.py        300 random episodes -> rollouts/episode_*.npz
  preprocess.py     crop to playfield, resize 64x64
                    -> proc/{frames,lives,episode,motion}.npy + meta.json
  ball_labels.py    exact ball (y, x) per frame -> proc/ball.npy
  dataset.py        ClipDataset: 4-frame clips that never cross a life loss
                    or an episode boundary
  check.py          shapes, clip count, ball-visibility summary
  inspect.py        ball-blob and resize-interpolation probes -> frames/*.png
models/
  vit.py            space-time ViT encoder (6 layers, d=192, 3 heads,
                    8x8 patches -> 4 frames x 64 = 256 tokens)
  predictor.py      narrow predictor (d=96, 4 layers, 3 heads)
  masking.py        multi-block tube masking + a causal (predict-next-frame) mask
  ema.py            EMA target encoder
  layers.py         attention / MLP / SwiGLU / norms
  check_order.py    asserts token order matches the patch grid
train/
  loss.py           target normalisation + smooth-L1 + VICReg anti-collapse
  vjepa.py          training loop
utils/
  monitor.py        collapse diagnostics + tripwire
  logger.py         metrics.csv + config.json per run
  plot_run.py       curves.png
  watch.py          live tail of metrics.csv with a health verdict
eval/
  probe.py          linear probe: frozen features -> ball position
visualize.py        six figures + a gif, on the real frames
docs/               make_figures.py, report.md/html/pdf, A/B collapse logs
```

## Run it

```bash
python -m data.collect                 # 300 random episodes  (only once)
python -m data.preprocess              # -> proc/*.npy, meta.json
python -m data.ball_labels             # -> proc/ball.npy
python -m data.dataset                 # sanity checks

python -m train.vjepa --run runs/vjepa --epochs 40
python -m utils.plot_run runs/vjepa    # runs/vjepa/curves.png

python -m eval.probe --ckpt runs/vjepa/checkpoint/final.pt
python visualize.py  --ckpt runs/vjepa/checkpoint/final.pt
```

`eval.probe` defaults to the EMA target encoder; `--which online` probes the
online one instead.

25% of steps use a **causal mask** (`--causal-frac`): the whole last frame is
hidden and predicted from the previous three. Same objective, but it is the
world-model question, and it is what panel 6 of `visualize.py` exercises.

Checkpoints land in `runs/<name>/checkpoint/`: `last.pt` every epoch,
`epoch<NNN>.pt` every `--save-every` epochs (default 5), and `final.pt` at the
end. Only `last.pt` carries optimiser and scheduler state, so `--resume` picks
up from that one; the others are weights only.

### Scale of the reference run

300 episodes -> 52,566 frames -> 38,887 clips. At `--batch-size 32` with
`drop_last`, that is 1,215 steps/epoch and 48,600 steps over 40 epochs. On an
M-series GPU (`mps`) the run took ~95 min wall clock: roughly 8.5 it/s, so
~2.4 min/epoch.

## Watching a run

```bash
python -m utils.watch                  # live, refreshes every 5s
python -m utils.watch --once           # just the latest row
python -m utils.watch runs/other --every 2
```

Prints one line per logged step with a `healthy` / `collapsing` / `COLLAPSED`
verdict. It only reads `metrics.csv`, so starting and stopping it never touches
the training process.

## Reading the training log

Every `--log-every` steps (default 50) a row lands in `runs/<name>/metrics.csv`,
alongside a one-off `config.json`. Each row carries `step`, `epoch`, `causal`,
`lr`, `tau`, `grad_norm`, the four loss terms (`loss_total`, `loss_pred`,
`loss_var`, `loss_cov`), and the full collapse-stat block for **both** encoders
— `tgt_*` for the EMA target, `ctx_*` for the online context encoder
(`norm`, `std`, `rel_std`, `eff_rank`, `eff_rank_c`, `cos`).

The target-side columns are the ones to watch:

| column | healthy | collapsed |
|---|---|---|
| `tgt_cos` | stays < 0.3 | climbs to ~1.0 |
| `tgt_rel_std` | holds or rises | decays toward 0 |
| `tgt_eff_rank` | tens to hundreds | -> 1 |
| `loss_var` | ~0 (hinge satisfied) | > 0 and rising |
| `loss_pred` | plateaus at a nonzero value | crashes toward 0 |

For reference, the healthy 40-epoch run ends at `tgt_cos` 0.0008,
`tgt_rel_std` 0.072, `tgt_eff_rank` 188, `loss_var` 0.0005, `loss_pred` 0.105.

**A prediction loss going to zero is a bad sign, not a good one.** Past step 300
the tripwire fires if `tgt_cos > 0.98`, `tgt_eff_rank < 3` or
`tgt_rel_std < 0.01`; training then writes `COLLAPSED.txt` and stops.

## What visualize.py produces

Into `runs/<name>/viz/` (derived from the `--ckpt` path):

**1. `1_masks.png`** — the real clip with context patches (green outline) and
target patches (red fill), ball circled

<img width="1736" height="924" alt="1_masks.png" src="https://github.com/user-attachments/assets/8be2f4ba-f14c-4a63-9745-89b0949285a7" />

**2. `2_collapse.png`** — pairwise-cosine histogram and singular spectrum, with
a HEALTHY / COLLAPSED verdict

<img width="1540" height="588" alt="2_collapse.png" src="https://github.com/user-attachments/assets/cc670a2f-903d-4f3e-9587-7716e12d2632" />

**3. `3_features.png`** — encoder tokens PCA'd to RGB and painted back onto the
frame; bricks / playfield / paddle should come out as different colours

<img width="1736" height="980" alt="3_features.png" src="https://github.com/user-attachments/assets/f506e1f4-a6ad-4b4c-bb07-d5ab3c483e18" />

**4. `4_pred_error.png`** — per-patch cosine between predicted and true target
embedding, over the real frame

<img width="1552" height="514" alt="4_pred_error.png" src="https://github.com/user-attachments/assets/588b1f29-0b69-415b-a5f4-ae2f7e42ee04" />

**5. `5_ball_track.png` + `ball_track.gif`** — **the ball**: true position
(green) vs the position a linear probe reads out of the frozen features (red)

<img width="1904" height="777" alt="5_ball_track.png" src="https://github.com/user-attachments/assets/051df484-5b14-4d91-819f-098372229234" />
<img width="420" height="440" alt="ball_track.gif" src="https://github.com/user-attachments/assets/ea90d59f-3057-4a90-96c8-7caa2ad32007" />

**6. `6_rollout.png`** — the last frame is hidden from the encoder entirely; the
predictor imagines its tokens and the probe reads the ball position out of the
*imagined* tokens (orange) against where the ball really was (green)

<img width="2352" height="672" alt="6_rollout.png" src="https://github.com/user-attachments/assets/af118c12-c163-45a4-814b-f228e0299c23" />
