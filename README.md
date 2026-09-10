# V-JEPA on Atari Breakout

A small V-JEPA (self-supervised video joint-embedding predictive architecture)
trained on random Breakout rollouts, end to end: collect -> preprocess -> train
-> monitor -> probe -> visualise.

See **[NOTES.md](NOTES.md)** for the collapse diagnosis and what was changed.

## Layout

```
data/
  collect.py        random rollouts -> rollouts/episode_*.npz
  preprocess.py     crop to playfield, resize 64x64 -> proc/*.npy + meta.json
  ball_labels.py    exact ball (y, x) per frame -> proc/ball.npy
  dataset.py        ClipDataset: 4-frame clips that never cross a life loss
models/
  vit.py            space-time ViT encoder (6 layers, d=192, 8x8 patches)
  predictor.py      narrow predictor (d=96, 4 layers)
  masking.py        multi-block tube masking + a causal (predict-next-frame) mask
  ema.py            EMA target encoder
  layers.py         attention / MLP / SwiGLU / norms
train/
  loss.py           target normalisation + smooth-L1 + VICReg anti-collapse
  vjepa.py          training loop
utils/
  monitor.py        collapse diagnostics + tripwire
  logger.py         metrics.csv per run
  plot_run.py       curves.png
eval/
  probe.py          linear probe: frozen features -> ball position
visualize.py        six figures + a gif, on the real frames
```

## Run it

```bash
python -m data.collect                 # 300 random episodes  (only once)
python -m data.preprocess              # -> proc/frames.npy, meta.json
python -m data.ball_labels             # -> proc/ball.npy
python -m data.dataset                 # sanity checks

python -m train.vjepa --run runs/vjepa --epochs 40
python -m utils.plot_run runs/vjepa    # runs/vjepa/curves.png

python -m eval.probe --ckpt runs/vjepa/checkpoint/final.pt
python visualize.py  --ckpt runs/vjepa/checkpoint/final.pt
```

25% of steps use a **causal mask** (`--causal-frac`): the whole last frame is
hidden and predicted from the previous three. Same objective, but it is the
world-model question, and it is what panel 6 of `visualize.py` exercises.

Resume an interrupted run with `--resume`. Roughly 10 it/s on an M-series GPU
(`mps`), so ~2.5 min/epoch at 1442 steps.

## Watching a run

```bash
PYTHONPATH=. ../.venv/bin/python -m utils.watch          # live, refreshes every 5s
PYTHONPATH=. ../.venv/bin/python -m utils.watch --once    # just the latest row
```

Prints one line per logged step with a `healthy` / `collapsing` / `COLLAPSED`
verdict. It only reads `metrics.csv`, so starting and stopping it never touches
the training process.

## Reading the training log

Every `--log-every` steps a row lands in `runs/<name>/metrics.csv`:

| column | healthy | collapsed |
|---|---|---|
| `tgt_cos` | stays < 0.3 | climbs to ~1.0 |
| `tgt_rel_std` | holds or rises | decays toward 0 |
| `tgt_eff_rank` | tens to hundreds | -> 1 |
| `loss_var` | 0 (hinge satisfied) | > 0 and rising |
| `loss_pred` | plateaus at a nonzero value | crashes toward 0 |

**A prediction loss going to zero is a bad sign, not a good one.** Training
aborts and writes `COLLAPSED.txt` if `tgt_cos > 0.98`, `tgt_eff_rank < 3` or
`tgt_rel_std < 0.01` after step 300 (`--abort-on-collapse`, on by default).

## What visualize.py produces

Into `runs/<name>/viz/`:

1. `1_masks.png` — the real clip with context patches (green outline) and target
   patches (red fill), ball circled
2. `2_collapse.png` — pairwise-cosine histogram and singular spectrum, with a
   HEALTHY / COLLAPSED verdict
3. `3_features.png` — encoder tokens PCA'd to RGB and painted back onto the
   frame; bricks / playfield / paddle should come out as different colours
4. `4_pred_error.png` — per-patch cosine between predicted and true target
   embedding, over the real frame
5. `5_ball_track.png` + `ball_track.gif` — **the ball**: true position (green)
   vs the position a linear probe reads out of the frozen features (red)
6. `6_rollout.png` — the last frame is hidden from the encoder entirely; the
   predictor imagines its tokens and the probe reads the ball position out of
   the *imagined* tokens (orange) against where the ball really was (green)
