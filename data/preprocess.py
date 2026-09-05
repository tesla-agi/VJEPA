import os
import glob
import numpy as np
import torch
import torch.nn.functional as F

ROW_CUT=180
SIZE=64


def build(data_dir="rollouts", out_dir="proc", size=SIZE, row_cut=ROW_CUT):
    os.makedirs(out_dir,exist_ok=True)
    files=sorted(glob.glob(os.path.join(data_dir,"episode_*.npz")))
    assert files, f"no episodes in {data_dir}"

    frames_out,lives_out,ep_out,motion_out=[],[],[],[]

    for ep, path in enumerate(files):
        d=np.load(path)
        f,lives=d["frames"],d["lives"]
        d.close()
        mo=np.abs(np.diff(f[:,:row_cut,:].astype(np.int16),axis=0))
        mo=mo.reshape(len(mo),-1).max(axis=1).astype(np.int16)
        mo=np.append(mo,-1)

        x=torch.from_numpy(f).float().unsqueeze(1)
        x=F.interpolate(x,size=(size,size),mode="area")
        small=x.squeeze(1).round().clamp(0,255).numpy().astype(np.uint8)

        frames_out.append(small)
        lives_out.append(lives)
        ep_out.append(np.full(len(f),ep,dtype=np.int32))
        motion_out.append(mo)

    np.save(os.path.join(out_dir,"frames.npy"),np.concatenate(frames_out))
    np.save(os.path.join(out_dir,"lives.npy"),np.concatenate(lives_out))
    np.save(os.path.join(out_dir,"episode.npy"),np.concatenate(ep_out))
    np.save(os.path.join(out_dir,"motion.npy"),np.concatenate(motion_out))

    n=sum(len(a) for a in frames_out)
    mb=n*size*size/1e6
    print(f"episodes {len(files)}  frames {n}  {size}x{size}  {mb:.0f} MB")


if __name__ == "__main__":
    build()