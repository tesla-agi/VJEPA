import math
import torch
import random

def sample_block(grid_size,scale_range,aspect_range,rng):
    s=rng.uniform(*scale_range)
    r=rng.uniform(*aspect_range)
    A=s*(grid_size**2)
    h=int(round(math.sqrt(A/r)))
    w=int(round(math.sqrt(A*r)))
    h=max(1,min(h,grid_size))
    w=max(1,min(w,grid_size))
    top=int(rng.integers(0,grid_size-h+1))
    left=int(rng.integers(0,grid_size-w+1))
    return [row*grid_size+col
            for row in range(top,top+h)
            for col in range(left,left+w)]

def spatial_mask(grid_size,scale_range,aspect_range,n_blocks,rng):
    B=[sample_block(grid_size,scale_range,aspect_range,rng)
       for _ in range(n_blocks)]
    union=set().union(*B)
    C=sorted(set(range(grid_size**2))-union)
    return sorted(union),C

def expand_tube(spatial_idx,n_spatial,n_frames):
    return [t*n_spatial+s
            for t in range(n_frames)
            for s in spatial_idx]

def sample_mask(batch_size,grid_size,n_frames,n_blocks,k,scale_range,aspect_range,rng):
    ctx_batch,tgt_batch=[],[]
    n_spatial=grid_size**2
    for clip in range(batch_size):
        mask_cells,ctx_cell=spatial_mask(grid_size,scale_range,aspect_range,n_blocks,rng)
        rng.shuffle(ctx_cell)
        assert len(ctx_cell)>=k
        keep=ctx_cell[:k]
        overflow=ctx_cell[k:]
        mask_cells=sorted(mask_cells+overflow)
        ctx_batch.append(expand_tube(keep,n_spatial,n_frames))
        tgt_batch.append(expand_tube(mask_cells,n_spatial,n_frames))
    c=torch.tensor(ctx_batch,dtype=torch.long)
    t=torch.tensor(tgt_batch,dtype=torch.long)
    return c,t


