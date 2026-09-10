# Why the representation was collapsing, and what changed

## The symptom

The prediction loss went **down** beautifully while the model learned nothing.
That is the signature of representation collapse: the objective
`L1(predictor(context), EMA_target(clip))` is minimised perfectly by making the
encoder output the *same thing everywhere*, and gradient descent finds that
solution long before it finds a useful one.

Measured on the original recipe (600 steps, batch 32, seed 42):

| step | prediction loss | token cosine | rel. std |
|-----:|----------------:|-------------:|---------:|
|   0  | 0.790 | **+0.033** | 0.0655 |
| 120  | 0.112 | +0.734 | 0.0348 |
| 280  | 0.025 | +0.957 | 0.0141 |
| 600  | **0.013** | **+0.979** | **0.0098** |

"token cosine" is the mean cosine similarity between two random disjoint halves
of the token batch. It goes from 0.03 (patches are distinguishable) to 0.98
(every patch of every frame maps to essentially one vector) while the loss
drops 60x. The loss curve alone looked like a successful run.

Note also that **effective rank stayed at ~55 the whole time**, which is why the
original `utils/monitor.py` never flagged anything — it computed the rank of the
*centred* token matrix, and a collapsed representation still has full-rank
residual noise around its single mean direction. Cosine similarity is the
detector that works; the monitor now reports the uncentred rank too, which does
go to 1 under collapse.

## Five causes, in order of how much they mattered

### 1. The loss had a trivial global minimum at zero  (`train/loss.py`)

Both encoders end in a norm layer with a learnable gain. Drive that gain to 0
and the target is identically 0, so the predictor only has to output 0 and the
loss is exactly 0. Nothing in the original objective forbade this.

**Fix:** normalise the target before the loss, which is what I-JEPA/V-JEPA
actually do (`h = F.layer_norm(h, (h.size(-1),))`). Every target token now has
fixed norm `sqrt(d)`, so shrinking is no longer a minimiser. Also switched L1 to
smooth-L1, matching the paper and giving better-behaved gradients near zero.

Belt and braces: `models/vit.py` now builds the encoder's *final* norm with
`elementwise_affine=False`, removing the mechanism as well as the incentive.

### 2. Nothing prevented directional collapse  (`train/loss.py`)

Fixing (1) blocks "shrink to zero" but not "map every patch to the same unit
vector", which is still a global minimum. In fact the A/B table above was
measured *with* fix (1) already in place — the model collapsed anyway.

**Fix:** a VICReg-style pair of terms on the online encoder output:
a hinge that forces every feature dimension to keep `std >= 1` across the token
batch, and a penalty on the off-diagonal mass of the feature *correlation*
matrix (correlation, not covariance, so it is scale-invariant and does not fight
the variance term by simply shrinking the features).

### 3. The masking made the task unlearnable  (`models/masking.py`)

The context was subsampled to `k=7` of 64 spatial cells — **11% visible** — and
*every* remaining cell was a prediction target, 89%. Reconstructing 89% of a
scene from 11% of it is not a solvable problem, so the best available strategy
really is to emit the conditional mean, which is close to a constant. The mask
was pushing the model into the collapse basin.

**Fix:** context is now 16 cells (25%) and targets 24 cells (37%) — hard but
solvable. Groups retuned accordingly (`SHORT_RANGE` 6 blocks @ 0.10-0.18,
`LONG_RANGE` 2 blocks @ 0.30-0.45), and the sampler now rejection-samples until
both the context and the mask are large enough.

### 4. Weight decay was applied to the collapse-critical parameters  (`train/vjepa.py`)

`AdamW(..., lr=3e-4)` uses the default `weight_decay=0.01` on *everything*,
including norm gains, biases, `pos_embed` and `mask_token`. Decaying a norm gain
is a constant pull toward exactly the degenerate solution.

**Fix:** `param_groups()` puts all 1-D parameters and the positional/mask
embeddings in a no-decay group.

### 5. The score display was ~40% of the image variance  (`data/preprocess.py`)

The raw 210x160 frame was squashed whole into 64x64. Rows 5-31 are the
score/lives HUD, whose temporal std is ~66 against the ball's ~8 — the single
highest-variance thing in the frame, and completely unpredictable from the
playfield. A predictive objective fed unpredictable high-variance noise is being
actively pushed toward outputting the mean.

**Fix:** crop to the playfield (`rows 32:198, cols 8:152`) before resizing.
Verified row map: 5-31 HUD, 57-92 bricks, 93-188 ball region, 189-194 paddle.
The crop also buys the ball ~20% more pixels. `preprocess.py` now writes
`proc/meta.json` with the crop and the pixel mean/std, and `ClipDataset` reads
its normalisation from there instead of hard-coded `41.81 / 58.91`.

## After

Same 600 steps, same seed, new recipe:

| step | prediction loss | token cosine | rel. std | eff. rank |
|-----:|----------------:|-------------:|---------:|----------:|
|   0  | 0.433 | +0.032 | 0.0655 |  56 |
| 120  | 0.077 | +0.056 | 0.0665 |  92 |
| 360  | 0.073 | +0.064 | 0.0694 | 142 |
| 600  | 0.060 | **+0.051** | **0.0700** | **155** |

Cosine stays near zero, relative std holds, effective rank *grows*. The
prediction loss settles at 0.060 rather than crashing to 0.013 — higher, because
the model is now solving the real problem instead of the degenerate one.

One knock-on to be careful about: `motion` is computed on the *cropped* frame
now, so `MOTION_ROW_CUT` had to move from 180 (raw coords, "above the paddle")
to 155 (cropped coords, same meaning). Left at 180 it would have covered the
whole cropped frame including the paddle, and the static-clip filter would have
quietly stopped filtering — drop rate fell from 24.9% to 3.4% before this was
caught. With the fix: 38,887 clips, 18.6% dropped as static.

## A note on why `gamma = 1.0` is the right hinge

The encoder ends in a non-affine RMSNorm, so every token has norm exactly
`sqrt(192) = 13.86`. An isotropic embedding therefore has per-dimension std of
exactly 1.0. Setting the variance hinge at `gamma = 1.0` asks for precisely
"as spread out as an isotropic representation would be" — it is satisfied
(loss ~0.008) by a healthy model and only produces gradient when the embedding
starts folding up. It is a guard rail, not a driving force.

## Addition: the causal task

25% of training steps now hide the entire last frame and predict its tokens
from the previous three (`sample_causal_mask`, `--causal-frac`). This is not
part of the V-JEPA recipe — it is the world-model question, and without it the
predictor is out of distribution when `visualize.py` asks it to imagine a whole
unseen frame. The objective is unchanged; only the mask differs.

## Other fixes made along the way

- `models/predictor.py`: the output head `proj_out` was initialised with the
  *residual* std `0.02/sqrt(2*depth)`, which it is not — it is an output
  projection. It now uses 0.02, so the first few hundred steps are not dead.
  Depth raised 2 -> 4. Returns a contiguous tensor (MPS `smooth_l1` rejects the
  non-contiguous slice).
- `models/layers.py`: attention uses `F.scaled_dot_product_attention`;
  `return_attn=True` keeps the explicit path for visualisation.
- `models/ema.py`: EMA now copies buffers as well as parameters, and the
  momentum schedule is an explicit `tau_start -> tau_end` linear ramp.
- `data/dataset.py`: added `clip(i)` returning the raw uint8 clip and its
  absolute frame index, needed by the probe and the visualiser.

## What is *not* claimed

`eval/probe.py` reports that a **random-init encoder of the same architecture
also probes at R^2 ~ 0.88**. That is expected — a random ViT is a nearly
information-preserving random projection, and this is a 64x64 grayscale game.
The probe's job here is to detect collapse (a collapsed encoder drops to R^2 ~ 0,
the constant-predictor floor), not to argue that V-JEPA pretraining beats a
random network. Demonstrating a *gain* over random-init would need a harder
downstream task than reading out a visible ball.
