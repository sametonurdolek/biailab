"""One frozen test evaluation, journaled by image; refuses to overwrite final scores."""
import csv,json,traceback
import numpy as np
import torch
from PIL import Image
from config import ROOT
from common import seed_all,load_json,save_json,load_pair
from audit_dataset import digest
from models import HNet
from metrics import psnr,ssim
from document_pipeline import preprocess,restore

def main():
    seed_all();frozen=load_json('frozen.json');cal=load_json('calibration.json');device='cuda' if torch.cuda.is_available() else 'cpu'
    if (ROOT/'results'/'metrics.json').exists():raise RuntimeError('Test already scored; read existing metrics, do not rerun')
    for key,name in [('crop_manifest_sha256','crops.json'),('split_sha256','split.json'),('calibration_sha256','calibration.json')]:
        assert frozen[key]==digest(ROOT/'manifests'/name),'Frozen manifest changed'
    assert frozen['nlm_sha256']==digest(ROOT/'nlm.py')
    for name,sha in frozen['protocol_source_sha256'].items():assert digest(ROOT/name)==sha,'Protocol source changed: '+name
    models={}
    for name,spatial in [('scalar',False),('spatial',True)]:
        file=ROOT/'checkpoints'/name/'best.pt';assert digest(file)==frozen[name]
        completed=json.loads((file.parent/'completed.json').read_text());assert completed['status']=='completed'
        ckpt=torch.load(file,map_location=device,weights_only=True);assert ckpt['spatial'] is spatial
        model=HNet(spatial=spatial).to(device);model.load_state_dict(ckpt['state_dict']);model.eval();models[name]=model
    folder=ROOT/'results';folder.mkdir(exist_ok=True)
    for sub in ['h_maps','errors','outputs','journal']: (folder/sub).mkdir(exist_ok=True)
    lock=folder/'test_started.json'
    if not lock.exists():
        with lock.open('x') as f:json.dump(dict(frozen_sha256=digest(ROOT/'manifests'/'frozen.json')),f)
    else: assert json.loads(lock.read_text())['frozen_sha256']==digest(ROOT/'manifests'/'frozen.json')
    rows=[r for r in load_json('split.json')['prepared'] if r['split']=='test'];scores=[]
    try:
        for row in sorted(rows,key=lambda r:r['id']):
            journal=folder/'journal'/f"{row['id']}.json"
            if journal.exists():scores.append(json.loads(journal.read_text()));continue
            dirty,clean=load_pair(row);x=torch.from_numpy(preprocess(dirty,cal['pipeline']))[None,None].to(device)
            target=torch.from_numpy(clean)[None,None].to(device);result=dict(image=row['id'])
            with torch.no_grad():
                for name,model in models.items():
                    out,h=restore(x,model,cal['h_lo'],cal['h_hi'])
                    result[name+'_psnr']=float(psnr(out,target));result[name+'_ssim']=float(ssim(out,target))
                    result[name+'_h_mean']=float(h.mean());result[name+'_h_std']=float(h.std(unbiased=False));result[name+'_h_min']=float(h.min());result[name+'_h_max']=float(h.max())
                    a=out[0,0].cpu().numpy();hm=h[0,0].cpu().numpy();error=np.abs(a-clean)
                    # Float artifacts are authoritative; PNGs are display-only.
                    np.save(folder/'outputs'/f"{row['id']}_{name}.npy",a)
                    np.save(folder/'h_maps'/f"{row['id']}_{name}.npy",hm)
                    np.save(folder/'errors'/f"{row['id']}_{name}.npy",error)
                    Image.fromarray(np.uint8(np.clip(a*255,0,255).round())).save(folder/'outputs'/f"{row['id']}_{name}.png")
                result['delta_psnr']=result['spatial_psnr']-result['scalar_psnr'];result['delta_ssim']=result['spatial_ssim']-result['scalar_ssim']
            save_json(journal.relative_to(ROOT),result);scores.append(result);print(row['id'],result['delta_psnr'],flush=True);torch.cuda.empty_cache()
        mean={k:float(np.mean([r[k] for r in scores])) for k in ['scalar_psnr','spatial_psnr','delta_psnr','scalar_ssim','spatial_ssim','delta_ssim']}
        with (folder/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
        save_json('results/metrics.json',dict(status='completed',n_test=len(scores),means=mean,per_image=scores,
            frozen_sha256=digest(ROOT/'manifests'/'frozen.json'),aggregation='equal-weight per-image mean, one real dirty-clean observation per image',
            intensity='decoded grayscale /255, continuous outputs, peak=1',ssim='baseline Wang 11x11 Gaussian sigma1.5, zero-pad, C1=.01^2,C2=.03^2',
            valid_region='common prepared rectangle per pair, identical for both systems',training_seed_count=1))
        print('TEST MEANS',mean,flush=True)
    except Exception:
        save_json('results/evaluation_failure.json',dict(status='failed',traceback=traceback.format_exc(),completed_images=[r['image'] for r in scores]));raise

if __name__=='__main__':main()

