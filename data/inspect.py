import numpy as np, os, torch
import torch.nn.functional as Fn
from PIL import Image

os.makedirs("frames", exist_ok=True)
frames = np.load("data/episode_0000.npz")["frames"]


def down(f, mode):
    t = torch.from_numpy(f).float()[None, None]
    kw = dict(align_corners=False) if mode == "bilinear" else {}
    return Fn.interpolate(t, size=(64, 64), mode=mode, **kw)[0, 0].clamp(0, 255).numpy().astype(np.uint8)


# find frames with a small moving blob in the mid-field (below bricks, above paddle)
diff = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
mid = (diff[:, 90:180, :] > 20).reshape(len(diff), -1).sum(axis=1)
cands = [int(i) for i in np.argsort(-mid)[:6]]
print("ball-candidate frames:", sorted(cands), "blob sizes:", mid[cands].tolist())

for i in sorted(cands):
    f = frames[i]
    cols = [np.array(Image.fromarray(f).resize((64, 64), Image.NEAREST)),
            down(f, "bilinear"),
            down(f, "area")]

    # confirm the three resizes actually differ
    if i == sorted(cands)[0]:
        print("nearest vs bilinear identical:", np.array_equal(cols[0], cols[1]))
        print("bilinear vs area identical:   ", np.array_equal(cols[1], cols[2]))

    cols = [np.array(Image.fromarray(c).resize((256, 256), Image.NEAREST)) for c in cols]
    raw = np.array(Image.fromarray(f).resize((195, 256), Image.NEAREST))
    Image.fromarray(np.concatenate([raw] + cols, axis=1)).save(f"frames/ball_{i:03d}.png")

# crop around the moving blob for the best candidate, so the ball fills the view
b = sorted(cands)[0]
ys, xs = np.where(diff[b, 90:180, :] > 20)
if len(ys):
    cy, cx = int(ys.mean()) + 90, int(xs.mean())
    y0, x0 = max(0, cy - 20), max(0, cx - 20)
    crop = frames[b][y0:y0 + 40, x0:x0 + 40]
    Image.fromarray(np.array(Image.fromarray(crop).resize((320, 320), Image.NEAREST))).save("frames/ball_crop_raw.png")
    small = down(frames[b], "area")
    sy, sx = int(cy * 64 / 210), int(cx * 64 / 160)
    c2 = small[max(0, sy - 6):sy + 6, max(0, sx - 6):sx + 6]
    Image.fromarray(np.array(Image.fromarray(c2).resize((320, 320), Image.NEAREST))).save("frames/ball_crop_64.png")
    print(f"blob at raw ({cy},{cx}) -> 64x64 ({sy},{sx}); wrote ball_crop_raw.png / ball_crop_64.png")