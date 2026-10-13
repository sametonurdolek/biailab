"""Shared deterministic processing. No learned restoration or target-dependent inference."""
import cv2
import numpy as np
import torch
from nlm import nlm
from config import TILE, PATCH, SEARCH

# Frozen candidate set before validation. All use continuous intensities; no binarization.
CANDIDATES = [dict(name='identity',mode='identity',white=.98,black=0.),
    *[dict(name=f'global_{white}',mode='global',white=white,black=.02) for white in [.9,.98]],
    *[dict(name=f'local_{size}_{black}',mode='local',size=size,white=.98,black=black) for size in [31,63] for black in [.0,.1]]]

def preprocess(gray, cfg):
    x=gray.astype(np.float32)
    if cfg['mode']=='identity': return x.copy()
    if cfg['mode']=='global':
        lo,hi=np.quantile(x,[cfg['black'],cfg['white']])
        return np.clip((x-lo)/max(float(hi-lo),10/255),0,1).astype(np.float32)
    # Closing suppresses dark foreground when estimating slowly varying paper brightness.
    s=cfg['size'];bg=cv2.morphologyEx(x,cv2.MORPH_CLOSE,np.ones((s,s),np.uint8),borderType=cv2.BORDER_REFLECT_101)
    bg=cv2.GaussianBlur(bg,(0,0),s/6,borderType=cv2.BORDER_REFLECT_101)
    x=np.clip(x/np.maximum(bg,.1),0,1)
    b=cfg['black']
    return np.clip((x-b)/(1-b),0,1).astype(np.float32)

def postprocess(output):
    # Only clip numerical roundoff; no hard threshold, sharpening or learned pixels.
    return output.clamp(0,1)

def nlm_tiled(x,h):
    """Exact neighborhood coverage with 13px halo; grouping affects floating sum order only."""
    height,width=x.shape[-2:];out=torch.empty_like(x);halo=SEARCH//2+PATCH//2
    for y in range(0,height,TILE):
        for z in range(0,width,TILE):
            y1=max(0,y-halo);x1=max(0,z-halo);y2=min(height,y+TILE+halo);x2=min(width,z+TILE+halo)
            value=nlm(x[...,y1:y2,x1:x2],h[...,y1:y2,x1:x2],patch=PATCH,search=SEARCH)
            endy=min(y+TILE,height);endx=min(z+TILE,width)
            out[...,y:endy,z:endx]=value[...,y-y1:endy-y1,z-x1:endx-x1]
    return postprocess(out)

def restore(x,model,lo,hi):
    # H-Net sees the entire preprocessed page. Scalar pools across the whole page.
    h=model(x,lo,hi)
    return nlm_tiled(x,h),h

