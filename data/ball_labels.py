import glob
import json
import os

import numpy as np

BALL_VALUE = 110
BALL_ROWS = (93, 188)
BALL_COLS = (8, 152)


def build(data_dir="rollouts", out_dir="proc"):
    with open(os.path.join(out_dir, "meta.json")) as fh:
        meta = json.load(fh)
    crop, size = meta["crop"], meta["size"]
    ch = crop["bottom"] - crop["top"]
    cw = crop["right"] - crop["left"]

    out = []
    for path in sorted(glob.glob(os.path.join(data_dir, "episode_*.npz"))):
        d = np.load(path)
        f = d["frames"]
        d.close()
        reg = f[:, BALL_ROWS[0]:BALL_ROWS[1], BALL_COLS[0]:BALL_COLS[1]] == BALL_VALUE
        lab = np.zeros((len(f), 3), dtype=np.float32)
        for i, m in enumerate(reg):
            ys, xs = np.nonzero(m)
            if len(ys) == 0:
                continue
            y_raw = ys.mean() + BALL_ROWS[0]
            x_raw = xs.mean() + BALL_COLS[0]
            # raw -> cropped -> 64x64
            lab[i] = (1.0,
                      (y_raw - crop["top"]) * size / ch,
                      (x_raw - crop["left"]) * size / cw)
        out.append(lab)

    ball = np.concatenate(out)
    np.save(os.path.join(out_dir, "ball.npy"), ball)
    vis = ball[:, 0] > 0
    print(f"frames {len(ball)}  ball visible {vis.mean():.1%}  "
          f"y {ball[vis, 1].min():.1f}-{ball[vis, 1].max():.1f}  "
          f"x {ball[vis, 2].min():.1f}-{ball[vis, 2].max():.1f}")


if __name__ == "__main__":
    build()
