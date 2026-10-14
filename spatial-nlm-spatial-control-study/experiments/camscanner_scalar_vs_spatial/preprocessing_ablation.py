"""Preprocessing-only vs saved scalar/spatial NLM on the same Bettertest pairs.

Only A is newly scored. B/C use the authoritative previous scores and float
outputs. No checkpoint is loaded, no H-Net/NLM inference runs, and no tuning
is performed. All original tracked experiment files must stay unchanged.
"""
from pathlib import Path
import argparse,csv,datetime,json,subprocess,sys,traceback
import numpy as np
import torch
from PIL import Image
from config import ROOT
from common import seed_all,load_json,environment
from audit_dataset import digest
from document_pipeline import preprocess,postprocess
from metrics import psnr,ssim

SOURCE=ROOT/'results'/'bettertest'
DEFAULT=SOURCE/'preprocessing_ablation'

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def protected():
    repo=ROOT.parents[2];prefix=ROOT.relative_to(repo).as_posix()+'/'
    names=subprocess.check_output(['git','ls-files','-z'],cwd=repo).split(b'\0')
    return {name:digest(repo/name) for item in names if item and (name:=item.decode('utf-8')).startswith(prefix)
            and '/preprocessing_ablation/' not in name and not name.endswith('preprocessing_ablation.py')}

def render(folder,rows,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (folder/'figures').mkdir(exist_ok=True)
    for row,pair in zip(rows,pairs):
        name=row['id']
        dirty=np.asarray(Image.open(SOURCE/pair['prepared']['dirty']['path']),dtype=np.float32)/255.
        clean=np.asarray(Image.open(SOURCE/pair['prepared']['clean']['path']),dtype=np.float32)/255.
        a=np.load(folder/'outputs'/f'{name}_preprocess_only.npy')
        b=np.load(SOURCE/'outputs'/f'{name}_scalar.npy');c=np.load(SOURCE/'outputs'/f'{name}_spatial.npy')
        fig,axes=plt.subplots(1,5,figsize=(18,7),constrained_layout=True)
        titles=['Prepared dirty','Clean target',f"A Preprocess only\n{row['preprocess_psnr']:.4f} dB / {row['preprocess_ssim']:.6f}",
                f"B Scalar NLM (saved)\n{row['scalar_psnr']:.4f} dB / {row['scalar_ssim']:.6f}",
                f"C Spatial NLM (saved)\n{row['spatial_psnr']:.4f} dB / {row['spatial_ssim']:.6f}"]
        for ax,array,title in zip(axes,[dirty,clean,a,b,c],titles):
            ax.imshow(array,cmap='gray',vmin=0,vmax=1);ax.set_title(title,fontsize=10);ax.axis('off')
        fig.suptitle(name+' | same target, preprocessing and prepared geometry; only NLM removed in A')
        fig.savefig(folder/'figures'/f'{name}_ablation.png',dpi=150);plt.close(fig)

def report(folder,metrics):
    means=metrics['means'];rows=metrics['per_image']
    text='# Preprocessing-only ablation on Bettertest\n\n'
    text+='Three prepared dirty/clean pairs; same frozen local_31_0.0 preprocessing and [0,1] clipping. A removes H-Net/NLM. B/C reuse the previously evaluated frozen scalar/spatial outputs and scores. No training, calibration, checkpoint loading, model inference or new geometry preparation occurred. Old results remain unchanged.\n\n'
    text+='## Mean scores\n\n| System | PSNR (dB) | SSIM | ΔPSNR versus A | ΔSSIM versus A |\n|---|---:|---:|---:|---:|\n'
    for branch,label in [('preprocess','A Preprocess only'),('scalar','B Preprocess + scalar NLM'),('spatial','C Preprocess + spatial NLM')]:
        dp=0 if branch=='preprocess' else means[branch+'_minus_preprocess_psnr'];ds=0 if branch=='preprocess' else means[branch+'_minus_preprocess_ssim']
        text+=f"| {label} | {means[branch+'_psnr']:.9f} | {means[branch+'_ssim']:.9f} | {dp:+.9f} | {ds:+.9f} |\n"
    text+='\n## Per-image scores\n\n| Pair | A PSNR | B PSNR | C PSNR | A SSIM | B SSIM | C SSIM | B−A PSNR | C−A PSNR | B−A SSIM | C−A SSIM |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    keys=['preprocess_psnr','scalar_psnr','spatial_psnr','preprocess_ssim','scalar_ssim','spatial_ssim','scalar_minus_preprocess_psnr','spatial_minus_preprocess_psnr','scalar_minus_preprocess_ssim','spatial_minus_preprocess_ssim']
    for row in rows:text+='| '+row['id']+' | '+' | '.join(f'{row[k]:.9f}' for k in keys)+' |\n'
    text+='\nSource: [metrics.json](metrics.json) -> `means`, `per_image`; [CSV](per_image.csv). Means give each image equal weight. Scores use the same peak1, float32 domain and zero-padded Wang SSIM as Bettertest. Existing B/C scores are imported rather than rescored or regenerated.\n\n'
    text+='## What NLM adds on these three pairs\n\n'
    for branch in ['scalar','spatial']:
        p=sum(r[branch+'_minus_preprocess_psnr']>0 for r in rows);s=sum(r[branch+'_minus_preprocess_ssim']>0 for r in rows)
        text+=f"{branch.capitalize()} NLM improves PSNR on {p}/3 and SSIM on {s}/3 relative to preprocessing only. Mean differences: {means[branch+'_minus_preprocess_psnr']:+.9f} dB and {means[branch+'_minus_preprocess_ssim']:+.9f} SSIM.\n\n"
    text+='These paired differences directly measure adding the existing NLM controllers after the fixed preprocessing, on this three-pair diagnostic. They do not establish the usefulness or uselessness of NLM for all documents, other bandwidths/windows, or the original 20-image cohort. Three deliberately selected pairs, large white margins, application-produced targets, grayscale evaluation and one original training seed limit generalization. No significance claim is made. No region or parameter was selected after seeing the scores.\n\n'
    text+='## Files and reproduction\n\n'
    for row in rows:
        name=row['id'];text+=f'* [{name} comparison](figures/{name}_ablation.png); [preprocessing-only PNG](outputs/{name}_preprocess_only.png); float32 NPY and error map are also saved.\n'
    text+='\n`provenance.json` records checkpoint/source/prepared-pair/cached-output hashes and the exact command. `preservation_verification.json` confirms original tracked experiment files are unchanged; `environment.json` records deterministic settings and versions. Logs retain failure tracebacks.\n\n'
    text+='Run in the existing experiment directory with an empty output folder:\n\n```powershell\n$env:OPENBLAS_NUM_THREADS = "1"\n$env:OMP_NUM_THREADS = "1"\n$env:PYTHONHASHSEED = "20261014"\npython preprocessing_ablation.py\n```\n\nFor an independent reproduction use `--output results/bettertest/preprocessing_ablation_reproduction`. Preserve the completed attempt. The original Bettertest pairs, prepared PNGs, cached B/C NPYs and scores are required. No training/evaluation driver is invoked.\n'
    (folder/'REPORT.md').write_text(text,encoding='utf-8')

def run(folder):
    seed_all();frozen=load_json('frozen.json');cal=load_json('calibration.json')
    for name,sha in frozen['protocol_source_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Frozen source changed: '+name)
    for branch in ['scalar','spatial']:
        if digest(ROOT/'checkpoints'/branch/'best.pt')!=frozen[branch]:raise RuntimeError('Frozen checkpoint changed')
    if digest(ROOT/'manifests/calibration.json')!=frozen['calibration_sha256']:raise RuntimeError('Calibration changed')
    prior=json.loads((SOURCE/'metrics.json').read_text());pairs=json.loads((SOURCE/'pairs.json').read_text())['pairs']
    if prior['status']!='completed' or len(pairs)!=3:raise RuntimeError('Require the completed three-pair Bettertest diagnostic')
    original=protected();publication=json.loads((ROOT.parents[2]/'BETTERTEST_PUBLICATION_MANIFEST.json').read_text())
    published={r['path']:r['sha256'] for r in publication['files']};references={};repo=ROOT.parents[2]
    for pair in pairs:
        for side in ['dirty','clean']:
            path=SOURCE/pair['prepared'][side]['path']
            if digest(path)!=pair['prepared'][side]['sha256']:raise RuntimeError('Prepared pair changed')
            references[path.relative_to(repo).as_posix()]=digest(path)
        for branch in ['scalar','spatial']:
            path=SOURCE/'outputs'/f"{pair['id']}_{branch}.npy";relative=path.relative_to(repo).as_posix()
            if digest(path)!=published[relative]:raise RuntimeError('Cached output differs from published result')
            references[relative]=digest(path)
    write(folder/'provenance.json',dict(command='python preprocessing_ablation.py',script_sha256=digest(Path(__file__)),
        frozen_sha256=digest(ROOT/'manifests/frozen.json'),calibration_sha256=digest(ROOT/'manifests/calibration.json'),
        original_bettertest_metrics_sha256=digest(SOURCE/'metrics.json'),pairs_sha256=digest(SOURCE/'pairs.json'),
        checkpoint_sha256={b:frozen[b] for b in ['scalar','spatial']},protocol_source_sha256=frozen['protocol_source_sha256'],
        prepared_and_cached_array_sha256=references,models_loaded=False,model_inference_runs=0,training=False,calibration=False,
        newly_scored='A: fixed preprocessing + shared clip only',B_C='reuse previous authoritative metrics and float32 arrays'))
    device='cuda' if torch.cuda.is_available() else 'cpu';rows=[]
    with torch.no_grad():
        for pair in pairs:
            name=pair['id'];old=next(r for r in prior['per_image'] if r['id']==name)
            dirty=np.asarray(Image.open(SOURCE/pair['prepared']['dirty']['path']),dtype=np.float32)/255.
            clean=np.asarray(Image.open(SOURCE/pair['prepared']['clean']['path']),dtype=np.float32)/255.
            a=postprocess(torch.from_numpy(preprocess(dirty,cal['pipeline']))[None,None].to(device))
            target=torch.from_numpy(clean)[None,None].to(device)
            array=a[0,0].cpu().numpy().copy();error=np.abs(array-clean)
            np.save(folder/'outputs'/f'{name}_preprocess_only.npy',array);np.save(folder/'errors'/f'{name}_preprocess_only.npy',error)
            Image.fromarray(np.uint8(np.clip(array*255,0,255).round())).save(folder/'outputs'/f'{name}_preprocess_only.png')
            row=dict(id=name,preprocess_psnr=float(psnr(a,target)),preprocess_ssim=float(ssim(a,target)),
                scalar_psnr=old['scalar_psnr'],scalar_ssim=old['scalar_ssim'],spatial_psnr=old['spatial_psnr'],spatial_ssim=old['spatial_ssim'])
            for branch in ['scalar','spatial']:
                row[branch+'_minus_preprocess_psnr']=row[branch+'_psnr']-row['preprocess_psnr']
                row[branch+'_minus_preprocess_ssim']=row[branch+'_ssim']-row['preprocess_ssim']
                cached=np.load(SOURCE/'outputs'/f'{name}_{branch}.npy')
                if cached.shape!=array.shape or cached.dtype!=np.float32:raise RuntimeError('Cached array convention mismatch')
                row[branch+'_mean_abs_change_from_preprocess']=float(np.abs(cached-array).mean())
            rows.append(row);print(json.dumps(row),flush=True)
    means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='id'}
    with (folder/'per_image.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics=dict(status='completed',n_images=3,means=means,per_image=rows,aggregation='arithmetic image means',
        source_bettertest_metrics_sha256=digest(SOURCE/'metrics.json'),new_model_inference=False,new_training=False,
        comparison='A fixed preprocessing+clip; B/C previously saved frozen scalar/spatial NLM outputs/scores')
    write(folder/'metrics.json',metrics);write(folder/'environment.json',environment())
    if protected()!=original:raise RuntimeError('Historical tracked files changed')
    write(folder/'preservation_verification.json',dict(historical_tracked_files_unchanged=True,protected_count=len(original),hashes=original))
    render(folder,rows,pairs);report(folder,metrics);print('MEANS',json.dumps(means),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT);args=parser.parse_args();folder=args.output.resolve()
    folder.mkdir(parents=True,exist_ok=False)
    for name in ['outputs','errors','logs']:(folder/name).mkdir()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    with (folder/'logs'/f'{stamp}_ablation.log').open('x',encoding='utf-8') as log:
        class Tee:
            def write(self,s):sys.__stdout__.write(s);log.write(s);log.flush()
            def flush(self):sys.__stdout__.flush();log.flush()
        sys.stdout=Tee()
        try:run(folder);print('EXIT_CODE=0')
        except Exception:log.write(traceback.format_exc());log.flush();raise
        finally:sys.stdout=sys.__stdout__

if __name__=='__main__':main()
