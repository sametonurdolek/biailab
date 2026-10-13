"""Identical initial weights, crop schedule, and budget; only h pooling differs."""
import json,traceback,time
import numpy as np
import torch
from config import ROOT,STEPS,BATCH,CROP,LR,VAL_EVERY,SEED
from common import seed_all,load_json,save_json,load_pair
from audit_dataset import digest
from document_pipeline import preprocess,postprocess,restore
from models import HNet
from nlm import nlm
from metrics import psnr

def main():
    seed_all();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cal=load_json('calibration.json');lo,hi=cal['h_lo'],cal['h_hi'];rows=load_json('split.json')['prepared'];schedule=load_json('crops.json')['batches']
    # TEST is not loaded by training.
    data={}
    for row in rows:
        if row['split']=='test':continue
        d,c=load_pair(row);data[row['id']]=(preprocess(d,cal['pipeline']),c)
    val=[r for r in rows if r['split']=='val']
    def batch(item):
        xs=[];ys=[]
        for crop in item['crops']:
            d,c=data[crop['id']];x,y=crop['x'],crop['y'];d=d[y:y+CROP,x:x+CROP];c=c[y:y+CROP,x:x+CROP]
            if crop['flip_h']:d=d[:,::-1];c=c[:,::-1]
            if crop['flip_v']:d=d[::-1];c=c[::-1]
            xs.append(d.copy());ys.append(c.copy())
        return torch.from_numpy(np.stack(xs))[:,None].to(device),torch.from_numpy(np.stack(ys))[:,None].to(device)
    def validate(model):
        model.eval();scores=[]
        with torch.no_grad():
            for row in val:
                d,c=data[row['id']];x=torch.from_numpy(d)[None,None].to(device);target=torch.from_numpy(c)[None,None].to(device)
                out,_=restore(x,model,lo,hi);scores.append(float(psnr(out,target)))
        model.train();return float(np.mean(scores))
    # Shared bias centering on the same real training crops. Identical tensors for both models.
    initial=HNet(spatial=True).to(device);initial.set_constant_h(cal['h_init'],lo,hi)
    x,_=batch(schedule[0])
    with torch.no_grad():
        for _ in range(4):
            mean_h=initial(x,lo,hi).mean();fraction=((mean_h-lo)/(hi-lo)).clamp(1e-4,1-1e-4)
            desired=(cal['h_init']-lo)/(hi-lo)
            delta=np.log(desired/(1-desired))-torch.log(fraction/(1-fraction))
            initial.head.bias.add_(delta)
    shared={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()}
    del initial;torch.cuda.empty_cache()
    common=dict(seed=SEED,steps=STEPS,batch=BATCH,crop=CROP,lr=LR,loss='L1 after shared postprocessing',optimizer='Adam',
        crop_manifest_sha256=digest(ROOT/'manifests'/'crops.json'),split_sha256=digest(ROOT/'manifests'/'split.json'),
        calibration_sha256=digest(ROOT/'manifests'/'calibration.json'),nlm_sha256=digest(ROOT/'nlm.py'),
        protocol_source_sha256={name:digest(ROOT/name) for name in ['config.py','document_pipeline.py','models.py','nlm.py','metrics.py','common.py','train.py']})
    for spatial in [False,True]:
        seed_all();name='spatial' if spatial else 'scalar';folder=ROOT/'checkpoints'/name;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'completed.json').exists():raise RuntimeError('Refusing to overwrite completed run: '+name)
        log_path=folder/'train.jsonl'
        if log_path.exists():raise RuntimeError('Refusing to overwrite attempted run: '+name)
        model=HNet(spatial=spatial).to(device);model.load_state_dict(shared)
        torch.save(dict(state_dict=shared,spatial=spatial,**common),folder/'initial.pt')
        opt=torch.optim.Adam(model.parameters(),lr=LR);best=-float('inf');best_step=None
        with log_path.open('x',encoding='utf-8') as log:
            try:
                for step in range(STEPS+1):
                    event=dict(step=step)
                    if step:
                        x,target=batch(schedule[step-1]);opt.zero_grad(set_to_none=True)
                        h=model(x,lo,hi);out=postprocess(nlm(x,h));loss=(out-target).abs().mean();loss.backward()
                        event.update(loss=float(loss.detach()),gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)),h_mean=float(h.detach().mean()),h_std=float(h.detach().std()))
                        if not np.isfinite(event['gradient_norm']):raise RuntimeError('Nonfinite gradient')
                        opt.step()
                    if step==0 or step==1 or step%VAL_EVERY==0:
                        score=validate(model);event['val_psnr']=score
                        if score>best:
                            best=score;best_step=step
                            torch.save(dict(state_dict=model.state_dict(),spatial=spatial,h_lo=lo,h_hi=hi,step=step,val_psnr=best,**common),folder/'best.pt')
                        print(name,step,'val',score,'best',best_step,flush=True)
                    log.write(json.dumps(event)+'\n');log.flush()
                    if step%10==0:torch.cuda.empty_cache()
                save_json(folder.relative_to(ROOT)/'completed.json',dict(status='completed',best_step=best_step,best_val_psnr=best,checkpoint_sha256=digest(folder/'best.pt'),initial_sha256=digest(folder/'initial.pt'),**common))
            except Exception as exc:
                log.write(json.dumps(dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))+'\n');log.flush();raise
        del model,opt;torch.cuda.empty_cache()
    save_json('manifests/frozen.json',dict(status='both training budgets completed; checkpoints frozen',
        scalar=digest(ROOT/'checkpoints'/'scalar'/'best.pt'),spatial=digest(ROOT/'checkpoints'/'spatial'/'best.pt'),**common))

if __name__=='__main__':main()
