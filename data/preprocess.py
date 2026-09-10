import glob
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

SIZE = 64
CROP = dict(top=32, bottom=198, left=8, right=152)
MOTION_ROW_CUT = 155


def build(data_dir="rollouts", out_dir="proc", size=SIZE, crop=CROP):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    assert files, f"no episodes in {data_dir}"
    frames_out, lives_out, ep_out, motion_out = [], [], [], []
    for ep, path in enumerate(files):
        d = np.load(path)
        f, lives = d["frames"], d["lives"]
        d.close()
        f = f[:, crop["top"]:crop["bottom"], crop["left"]:crop["right"]]
        mo = np.abs(np.diff(f[:, :MOTION_ROW_CUT, :].astype(np.int16), axis=0))
        mo = mo.reshape(len(mo), -1).max(axis=1).astype(np.int16)
        mo = np.append(mo, -1)
        x = torch.from_numpy(f).float().unsqueeze(1)
        x = F.interpolate(x, size=(size, size), mode="area")
        small = x.squeeze(1).round().clamp(0, 255).numpy().astype(np.uint8)
        frames_out.append(small)
        lives_out.append(lives)
        ep_out.append(np.full(len(f), ep, dtype=np.int32))
        motion_out.append(mo)
    frames = np.concatenate(frames_out)
    np.save(os.path.join(out_dir, "frames.npy"), frames)
    np.save(os.path.join(out_dir, "lives.npy"), np.concatenate(lives_out))
    np.save(os.path.join(out_dir, "episode.npy"), np.concatenate(ep_out))
    np.save(os.path.join(out_dir, "motion.npy"), np.concatenate(motion_out))
    mean, std = float(frames.mean()), float(frames.std())
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump({"size": size, "crop": crop, "n_frames": int(len(frames)),
                   "n_episodes": len(files), "pixel_mean": mean, "pixel_std": std},
                  fh, indent=2)

    print(f"episodes {len(files)}  frames {len(frames)}  {size}x{size}  "
          f"crop {crop}  mean {mean:.2f}  std {std:.2f}  "
          f"{len(frames) * size * size / 1e6:.0f} MB")


if __name__ == "__main__":
    build()
