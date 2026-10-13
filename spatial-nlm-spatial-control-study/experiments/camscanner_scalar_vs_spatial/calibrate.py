"""Select common processing on validation only; then sweep document-domain h."""
import numpy as np
import torch
from config import ROOT
from common import seed_all,load_json,save_json,load_pair,environment
from document_pipeline import CANDIDATES,preprocess,nlm_tiled
from metrics import psnr

def main():
    seed_all();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[r for r in load_json('split.json')['prepared'] if r['split']=='val']
    data=[load_pair(r) for r in rows];scores=[]
    for cfg in CANDIDATES:
        per=[]
        for dirty,clean in data:
            output=preprocess(dirty,cfg);per.append(float(-10*np.log10(max(np.mean((output-clean)**2),1e-12))))
        scores.append(dict(config=cfg,mean_psnr=float(np.mean(per)),per_image=dict(zip([r['id'] for r in rows],per))))
    selected=max(scores,key=lambda r:r['mean_psnr'])['config']
    # Noise statistic of TRAIN after selected common preprocessing, not Kodak's range.
    noises=[]
    for row in load_json('split.json')['prepared']:
        if row['split']!='train':continue
        dirty,_=load_pair(row);x=preprocess(dirty,selected)
        lap=cv2_filter(x);noises.append(float(np.mean(np.abs(lap))*np.sqrt(np.pi/2)/6))
    noise=float(np.median(noises));lo=max(1e-4,noise/8);hi=min(1.,max(.1,noise*16))
    tensors=[(torch.from_numpy(preprocess(d,selected))[None,None].to(device),torch.from_numpy(c)[None,None].to(device)) for d,c in data]
    sweep=[]
    def run_grid(grid):
        for h in grid:
            per=[]
            with torch.no_grad():
                for x,c in tensors:
                    y=nlm_tiled(x,torch.full_like(x,float(h)));per.append(float(psnr(y,c)))
            item=dict(h=float(h),mean_psnr=float(np.mean(per)),per_image=dict(zip([r['id'] for r in rows],per)));sweep.append(item)
            save_json('manifests/calibration_progress.json',dict(selected_pipeline=selected,sweep=sweep))
            print(f'h={h:.8f} val_PSNR={item["mean_psnr"]:.6f}',flush=True)
    grid=np.geomspace(lo,hi,12);run_grid(grid)
    best=max(sweep,key=lambda r:r['mean_psnr'])
    # Precommitted one-decade extension if optimum at either endpoint.
    if best['h']==lo:
        lo=lo/10;run_grid(np.geomspace(lo,grid[0],6)[:-1])
    elif best['h']==hi:
        hi=min(2.,hi*10);run_grid(np.geomspace(grid[-1],hi,6)[1:])
    best=max(sweep,key=lambda r:r['mean_psnr']);h0=best['h']
    # A boundary optimum is reported; open sigmoid initialization nudged 0.1% inside bounds.
    init=float(np.clip(h0,lo+(hi-lo)*.001,hi-(hi-lo)*.001))
    calibration=dict(status='frozen',pipeline=selected,pipeline_candidates=scores,training_noise_median=noise,
        h_lo=float(lo),h_hi=float(hi),h_star=h0,h_init=init,fixed_h_val_psnr=best['mean_psnr'],sweep=sweep,
        endpoint_optimum=bool(h0==lo or h0==hi),rule='12 geomspace points from max(1e-4,train noise/8) to min(1,max(.1,train noise*16)); extend winning endpoint once by decade/5 extra points',
        postprocess='clip[0,1]',selected_without_test=True)
    save_json('manifests/calibration.json',calibration);save_json('manifests/environment.json',environment())
    print('FROZEN',selected,'bounds',lo,hi,'h*',h0,flush=True)

def cv2_filter(x):
    import cv2
    return cv2.filter2D(x,-1,np.array([[1,-2,1],[-2,4,-2],[1,-2,1]],np.float32),borderType=cv2.BORDER_REFLECT_101)

if __name__=='__main__':main()

