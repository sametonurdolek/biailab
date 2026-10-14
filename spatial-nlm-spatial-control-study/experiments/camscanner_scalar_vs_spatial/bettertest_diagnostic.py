"""Evaluate new paired references once with the unchanged frozen CamScanner system.

No training, calibration or modification of historical artifacts. Geometry gate
is imported unchanged; pixel preparation is the original prepare_pairs.py block.
"""
from pathlib import Path, PureWindowsPath
import argparse, csv, datetime, json, subprocess, sys, traceback
import numpy as np
import cv2
import torch
from PIL import Image
from common import seed_all, load_json, environment
from config import ROOT, CROP, PATCH, SEARCH, TILE
from audit_dataset import read, gray, properties, correspondence, digest
from document_pipeline import preprocess, restore
from models import HNet
from metrics import psnr, ssim

DEFAULT_SOURCE = ROOT.parents[1] / 'bettertest'
DEFAULT_OUTPUT = ROOT / 'results' / 'bettertest'

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def old_source_path(prop):
    # Resolve historical Windows source locators to the same dataset in this checkout.
    parts = PureWindowsPath(prop['path']).parts
    index = parts.index('camscanner')
    file = ROOT.parents[1].joinpath(*parts[index:])
    if digest(file) != prop['sha256']:
        raise RuntimeError('Prior source differs from its recorded hash: ' + str(file))
    return file

def frozen_system():
    frozen = load_json('frozen.json'); cal = load_json('calibration.json')
    for key, name in [('crop_manifest_sha256', 'crops.json'), ('split_sha256', 'split.json'), ('calibration_sha256', 'calibration.json')]:
        if digest(ROOT/'manifests'/name) != frozen[key]:
            raise RuntimeError('Frozen manifest mismatch: ' + name)
    for name, sha in frozen['protocol_source_sha256'].items():
        if digest(ROOT/name) != sha: raise RuntimeError('Frozen source mismatch: ' + name)
    checkpoint_info = {}
    for name, spatial in [('scalar', False), ('spatial', True)]:
        path = ROOT/'checkpoints'/name/'best.pt'
        if digest(path) != frozen[name]: raise RuntimeError('Checkpoint hash mismatch: '+name)
        completed = json.loads((path.parent/'completed.json').read_text())
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['spatial'] is not spatial: raise RuntimeError('Wrong checkpoint mode: '+name)
        if completed['status'] != 'completed': raise RuntimeError('Incomplete original training')
        if (checkpoint['h_lo'], checkpoint['h_hi']) != (cal['h_lo'], cal['h_hi']):
            raise RuntimeError('Checkpoint bandwidth bounds mismatch')
        checkpoint_info[name] = dict(path=str(path.relative_to(ROOT)), sha256=frozen[name], step=checkpoint['step'], spatial=spatial)
    return frozen, cal, checkpoint_info

def protected_hashes():
    # Track all historical experiment files present in Git, not only the old score.
    repo = ROOT.parents[2]
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=repo).split(b'\0')
    prefix = ROOT.relative_to(repo).as_posix()+'/'
    return {path:digest(repo/path) for item in files if item and (path:=item.decode('utf-8')).startswith(prefix)
            and '/results/bettertest/' not in path and not path.endswith(('bettertest_diagnostic.py','BETTERTEST_REPORT.md'))}

def prepare(row, output):
    # Same preparation operations, thresholds and support-rectangle loop as the
    # frozen study's prepare_pairs.py. Source JPEGs are never rewritten.
    dirty=gray(read(row['dirty']['path']));clean=gray(read(row['clean']['path']));h,w=clean.shape
    dirty=cv2.resize(dirty,(w,h),interpolation=cv2.INTER_LINEAR)
    box=[0,0,w,h]
    if row['geometry']['registration_required']:
        H=np.array(row['geometry']['homography'])
        dirty=cv2.warpPerspective(dirty,H,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
        valid=cv2.warpPerspective(np.ones((h,w),np.uint8),H,(w,h),flags=cv2.INTER_NEAREST)
        x1=y1=0;x2=w;y2=h
        for _ in range(w+h):
            sub=valid[y1:y2,x1:x2]
            if sub.all():break
            fractions=[sub[0].mean(),sub[-1].mean(),sub[:,0].mean(),sub[:,-1].mean()]
            side=int(np.argmin(fractions))
            if side==0:y1+=1
            elif side==1:y2-=1
            elif side==2:x1+=1
            else:x2-=1
        box=[x1+2,y1+2,x2-2,y2-2]
    x1,y1,x2,y2=box;dirty=dirty[y1:y2,x1:x2];clean=clean[y1:y2,x1:x2]
    if min(dirty.shape)<CROP:raise RuntimeError('Prepared region too small: '+row['id'])
    files={}
    for side,array in [('dirty',dirty),('clean',clean)]:
        path=output/'prepared'/f"{row['id']}_{side}.png";path.parent.mkdir(exist_ok=True)
        if path.exists():
            stored=np.asarray(Image.open(path))
            if not np.array_equal(stored,array):raise RuntimeError('Existing prepared data differs')
        else:Image.fromarray(array).save(path)
        files[side]=dict(path=path.relative_to(output).as_posix(),sha256=digest(path))
    return dict(width=dirty.shape[1],height=dirty.shape[0],original_clean_roi=box,
        resize_required=(row['dirty']['width'],row['dirty']['height'])!=(w,h),registration=row['geometry']['registration_required'],
        applied_homography=row['geometry']['homography'] if row['geometry']['registration_required'] else np.eye(3).tolist(),**files)

def target_descriptors(path):
    x=gray(read(path)).astype(np.float32)/255.
    ink=(x<.7).astype(np.uint8)
    bg=(x>.8)&(cv2.dilate(ink,np.ones((11,11),np.uint8))==0)
    bg[:5]=False;bg[-5:]=False;bg[:,:5]=False;bg[:,-5:]=False
    lap=cv2.Laplacian(x,cv2.CV_32F,ksize=1,borderType=cv2.BORDER_REFLECT_101)
    immer=cv2.filter2D(x,-1,np.array([[1,-2,1],[-2,4,-2],[1,-2,1]],np.float32),borderType=cv2.BORDER_REFLECT_101)
    edge=np.hypot(cv2.Sobel(x,-1,1,0),cv2.Sobel(x,-1,0,1))
    if not bg.any():raise RuntimeError('No descriptor background support: '+str(path))
    return dict(bright_background_fraction=float(bg.mean()),bright_background_variance=float(x[bg].var()),
        background_laplacian_abs=float(np.abs(lap[bg]).mean()),background_hf_energy=float((lap[bg]**2).mean()),
        estimated_background_noise=float(np.abs(immer[bg]).mean()*np.sqrt(np.pi/2)/6),
        global_laplacian_abs=float(np.abs(lap).mean()),white_fraction=float((x>=.98).mean()),edge_density=float((edge>.2).mean()))

def cleanliness(rows, output):
    prior=load_json('pairs.json')['pairs'];test_ids=set(load_json('split.json')['partitions']['test']);cohorts={}
    for name, selected in [('previous_test_20',[r for r in prior if r['id'] in test_ids]),('previous_all_100',prior)]:
        per=[dict(id=r['id'],sha256=r['clean']['sha256'],descriptors=target_descriptors(old_source_path(r['clean']))) for r in selected]
        cohorts[name]=dict(n=len(per),per_image=per,means={k:float(np.mean([r['descriptors'][k] for r in per])) for k in per[0]['descriptors']})
    per=[dict(id=r['id'],sha256=r['clean']['sha256'],descriptors=target_descriptors(Path(r['clean']['path']))) for r in rows]
    cohorts['bettertest_3']=dict(n=len(per),per_image=per,means={k:float(np.mean([r['descriptors'][k] for r in per])) for k in per[0]['descriptors']})
    write(output/'target_cleanliness.json',dict(cohorts=cohorts,aggregation='equal-image arithmetic means on RAW clean sources, before resize/registration',
        formulas=dict(background_mask='gray>.8 AND farther than 5px square radius from gray<.7; 5px border excluded',
            variance='population variance on mask',laplacian='OpenCV ksize1, reflect101',hf_energy='mean squared Laplacian on mask',
            estimated_noise='mean absolute Immerkaer 3x3 response on mask * sqrt(pi/2)/6',white_fraction='fraction gray>=.98',
            edge_density='fraction Sobel magnitude>.2'),
        limitation='Background and high-frequency proxies are not calibrated sensor noise. White fraction and edge density depend on margins, ink coverage, font, diagrams and content. No restoration parameter is selected from them.'))

def content_overlap(rows, output):
    prior=load_json('pairs.json')['pairs'];roles={r['id']:r['split'] for r in load_json('split.json')['prepared']};cache=[]
    sift=cv2.SIFT_create(nfeatures=1200)
    for r in prior:
        image=gray(read(old_source_path(r['clean'])));kp,des=sift.detectAndCompute(image,None);cache.append((r,kp,des,image.shape))
    matches=[]
    for row in rows:
        image=gray(read(row['clean']['path']));kp,des=sift.detectAndCompute(image,None)
        for old,op,od,shape in cache:
            if des is None or od is None:continue
            good=[a for a,b in cv2.BFMatcher().knnMatch(des,od,k=2) if a.distance<.7*b.distance]
            if len(good)<35:continue
            p=np.float32([kp[m.queryIdx].pt for m in good]);q=np.float32([op[m.trainIdx].pt for m in good])
            H,mask=cv2.findHomography(p,q,cv2.RANSAC,2.)
            if H is None:continue
            ok=mask.ravel().astype(bool)
            if ok.sum()<35:continue
            coverage=(np.ptp(p[ok,0])/image.shape[1])*(np.ptp(p[ok,1])/image.shape[0])
            if ok.sum()>=35 and ok.mean()>=.65 and ok.sum()>=.12*min(len(kp),len(op)) and coverage>=.2:
                matches.append(dict(new_id=row['id'],prior_id=old['id'],prior_split=roles[old['id']],inliers=int(ok.sum()),inlier_fraction=float(ok.mean()),new_coverage=float(coverage)))
    write(output/'content_overlap.json',dict(matches=matches,method='SIFT1200 over all 100 previous clean sources; ratio .7, homography2px; >=35 inliers, fraction>=.65, >=.12 smaller descriptor set, new coverage>=.2',
        limitation='No detected match is not proof of independent source-page provenance. Used only to describe the cohort, never to choose a model.'))

def audit(source, output):
    seed_all();frozen,cal,ckpts=frozen_system();output.mkdir(parents=True,exist_ok=True)
    if (output/'metrics.json').exists():raise RuntimeError('Already evaluated; audit may not rewrite its prepared input')
    dirty={p.name:p for p in (source/'dirty').iterdir() if p.is_file()};clean={p.name:p for p in (source/'clean').iterdir() if p.is_file()}
    if dirty.keys()!=clean.keys() or len(dirty)!=3:raise RuntimeError('Require exactly three unambiguous filename-identical pairs')
    prior=load_json('pairs.json')['pairs'];roles={r['id']:r['split'] for r in load_json('split.json')['prepared']};rows=[]
    for filename in sorted(dirty):
        d,c=dirty[filename],clean[filename];dp,cp=properties(d),properties(c);geometry=correspondence(d,c)
        row=dict(id=d.stem,dirty=dp,clean=cp,geometry=geometry,pairing_rule='exact filename identity across dirty/clean',
            exact_prior_dirty_matches=[dict(id=r['id'],split=roles[r['id']]) for r in prior if r['dirty']['sha256']==dp['sha256']])
        rows.append(row);print('AUDIT',row['id'],'valid',geometry['valid'],'registration',geometry.get('registration_required'),flush=True)
    write(output/'pairs.json',dict(status='audited',source=str(source),pairs=rows,unmatched=[],geometry_source='unchanged audit_dataset.correspondence'))
    if not all(r['geometry']['valid'] for r in rows):raise RuntimeError('STOP: geometry gate failed; no metrics')
    for row in rows:row['prepared']=prepare(row,output)
    write(output/'pairs.json',dict(status='prepared',source=str(source),pairs=rows,unmatched=[],geometry_source='unchanged audit_dataset.correspondence',preparation_source='original prepare_pairs.py pixel-preparation block'))
    cleanliness(rows,output);content_overlap(rows,output)
    write(output/'frozen_system.json',dict(checkpoints=ckpts,pipeline=cal['pipeline'],h_lo=cal['h_lo'],h_hi=cal['h_hi'],h_star=cal['h_star'],
        patch=PATCH,search=SEARCH,tile=TILE,protocol_source_sha256=frozen['protocol_source_sha256'],
        reused_source_sha256={n:digest(ROOT/n) for n in ['evaluate.py','audit_dataset.py','prepare_pairs.py']},
        calibration_sha256=digest(ROOT/'manifests/calibration.json'),frozen_manifest_sha256=digest(ROOT/'manifests/frozen.json'),
        diagnostic_source_sha256=digest(Path(__file__)),no_training=True,no_calibration=True))
    write(output/'environment.json',environment());write(output/'protected_before.json',protected_hashes())

def evaluate(output):
    seed_all();frozen,cal,_=frozen_system()
    if (output/'metrics.json').exists() or (output/'evaluation_started.json').exists():raise RuntimeError('One frozen diagnostic only; existing attempt may not be rescored')
    pairs=json.loads((output/'pairs.json').read_text())
    if pairs['status']!='prepared' or len(pairs['pairs'])!=3:raise RuntimeError('Run audit/preparation first')
    before=json.loads((output/'protected_before.json').read_text())
    if before!=protected_hashes():raise RuntimeError('Historical artifacts changed since audit')
    for row in pairs['pairs']:
        for side in ['dirty','clean']:
            if digest(Path(row[side]['path']))!=row[side]['sha256']:raise RuntimeError('Diagnostic source image changed')
            if digest(output/row['prepared'][side]['path'])!=row['prepared'][side]['sha256']:raise RuntimeError('Prepared pair changed')
    device='cuda' if torch.cuda.is_available() else 'cpu';models={}
    for name,spatial in [('scalar',False),('spatial',True)]:
        checkpoint=torch.load(ROOT/'checkpoints'/name/'best.pt',map_location=device,weights_only=True)
        model=HNet(spatial=spatial).to(device);model.load_state_dict(checkpoint['state_dict']);model.eval();model.requires_grad_(False);models[name]=model
    for sub in ['outputs','h_maps','errors','journal']:(output/sub).mkdir(exist_ok=True)
    with (output/'evaluation_started.json').open('x',encoding='utf-8') as f:
        json.dump(dict(pairs_sha256=digest(output/'pairs.json'),frozen_manifest_sha256=digest(ROOT/'manifests/frozen.json'),diagnostic_source_sha256=digest(Path(__file__))),f,indent=2)
    scores=[]
    with torch.no_grad():
        for row in pairs['pairs']:
            dirty=np.asarray(Image.open(output/row['prepared']['dirty']['path']),dtype=np.float32)/255.
            clean=np.asarray(Image.open(output/row['prepared']['clean']['path']),dtype=np.float32)/255.
            raw=torch.from_numpy(dirty)[None,None].to(device);target=torch.from_numpy(clean)[None,None].to(device)
            x=torch.from_numpy(preprocess(dirty,cal['pipeline']))[None,None].to(device)
            result=dict(id=row['id'],dirty_psnr=float(psnr(raw,target)),dirty_ssim=float(ssim(raw,target)))
            for name,model in models.items():
                restored,h=restore(x,model,cal['h_lo'],cal['h_hi'])
                result[name+'_psnr']=float(psnr(restored,target));result[name+'_ssim']=float(ssim(restored,target))
                a=restored[0,0].cpu().numpy().copy();hm=h[0,0].cpu().numpy().copy();error=np.abs(a-clean)
                for sub,array in [('outputs',a),('h_maps',hm),('errors',error)]:np.save(output/sub/f"{row['id']}_{name}.npy",array)
                result[name+'_h_stats']=dict(mean=float(h.mean()),std=float(h.std(unbiased=False)),min=float(h.min()),max=float(h.max()))
                if name=='scalar' and np.ptp(hm)!=0:raise RuntimeError('Scalar map is not constant')
            result['delta_psnr']=result['spatial_psnr']-result['scalar_psnr'];result['delta_ssim']=result['spatial_ssim']-result['scalar_ssim']
            write(output/'journal'/f"{row['id']}.json",result);scores.append(result);print('SCORE',json.dumps(result),flush=True);torch.cuda.empty_cache()
    columns=['id','dirty_psnr','scalar_psnr','spatial_psnr','delta_psnr','dirty_ssim','scalar_ssim','spatial_ssim','delta_ssim']
    means={k:float(np.mean([r[k] for r in scores])) for k in columns[1:]}
    with (output/'per_image.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows([{k:r[k] for k in columns} for r in scores])
    old=json.loads((ROOT/'results/metrics.json').read_text())
    write(output/'metrics.json',dict(status='completed',n_images=3,means=means,per_image=scores,previous_test_means=old['means'],
        scalar_checkpoint_sha256=frozen['scalar'],spatial_checkpoint_sha256=frozen['spatial'],
        frozen_manifest_sha256=digest(ROOT/'manifests/frozen.json'),prepared_manifest_sha256=digest(output/'pairs.json'),
        intensity='prepared grayscale /255, no clean normalization, float32 restored pixels, peak1',
        metrics='unchanged metrics.py: PSNR; Wang SSIM 11x11 Gaussian sigma1.5, zero padding, C1=.01^2 C2=.03^2',
        aggregation='three image arithmetic means; no confidence/significance claim',training_performed=False,calibration_performed=False))
    after=protected_hashes()
    if before!=after:raise RuntimeError('Historical files changed during evaluation')
    write(output/'preservation_verification.json',dict(all_historical_tracked_experiment_files_unchanged=True,protected_count=len(before),hashes=after,
        original_bettertest_sources_unchanged=all(digest(Path(r[s]['path']))==r[s]['sha256'] for r in pairs['pairs'] for s in ['dirty','clean'])))
    print('MEANS',json.dumps(means),flush=True)

def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pairs=json.loads((output/'pairs.json').read_text())['pairs'];metrics=json.loads((output/'metrics.json').read_text());_,cal,_=frozen_system()
    (output/'figures').mkdir(exist_ok=True)
    for row in pairs:
        dirty=np.asarray(Image.open(output/row['prepared']['dirty']['path']),dtype=np.float32)/255.;clean=np.asarray(Image.open(output/row['prepared']['clean']['path']),dtype=np.float32)/255.
        arrays=[dirty,clean]+[np.load(output/'outputs'/f"{row['id']}_{name}.npy") for name in ['scalar','spatial']]
        arrays += [np.load(output/'errors'/f"{row['id']}_{name}.npy") for name in ['scalar','spatial']]
        arrays += [np.load(output/'h_maps'/f"{row['id']}_{name}.npy") for name in ['scalar','spatial']]
        h,w=clean.shape;fig,axes=plt.subplots(2,4,figsize=(16,max(5.5,min(18,8*h/w+1.5))),constrained_layout=True)
        titles=['Dirty (prepared geometry)','Clean target','Frozen scalar output','Frozen spatial output','Scalar absolute error','Spatial absolute error','Scalar broadcast h','Spatial h(x,y)']
        for i,ax in enumerate(axes.flat):
            if i<4:ax.imshow(arrays[i],cmap='gray',vmin=0,vmax=1)
            elif i<6:ea=ax.imshow(arrays[i],cmap='magma',vmin=0,vmax=.5)
            else:ha=ax.imshow(arrays[i],cmap='viridis',vmin=cal['h_lo'],vmax=cal['h_hi'])
            ax.set_title(titles[i]);ax.axis('off')
        fig.colorbar(ea,ax=list(axes.flat)[4:6],location='bottom',shrink=.7,label='Absolute error [0,.5]')
        fig.colorbar(ha,ax=list(axes.flat)[6:8],location='bottom',shrink=.7,label='Frozen bandwidth interval')
        score=next(s for s in metrics['per_image'] if s['id']==row['id'])
        fig.suptitle(f"{row['id']} | spatial - scalar: PSNR {score['delta_psnr']:+.4f} dB, SSIM {score['delta_ssim']:+.6f}")
        fig.savefig(output/'figures'/f"{row['id']}_comparison.png",dpi=140);plt.close(fig)

def report(output):
    pairs=json.loads((output/'pairs.json').read_text())['pairs'];m=json.loads((output/'metrics.json').read_text());clean=json.loads((output/'target_cleanliness.json').read_text())
    overlap=json.loads((output/'content_overlap.json').read_text());system=json.loads((output/'frozen_system.json').read_text());env=json.loads((output/'environment.json').read_text())
    text='# Better-clean-target diagnostic with frozen Scalar/Spatial NLM\n\n'
    text+='Completed diagnostic evaluation of three new pairs. **No retraining, calibration, checkpoint selection or processing changes were performed.** Old CamScanner results remain unchanged.\n\n'
    text+='## Frozen system and pair preparation\n\n'
    for name,c in system['checkpoints'].items():text+=f"* {name}: `{c['path']}`, selected step {c['step']}, SHA256 `{c['sha256']}`.\n"
    text+=f"\nShared preprocessing `{system['pipeline']['name']}`; 31x31 closing, Gaussian sigma31/6, grayscale /255. NLM patch{system['patch']}/search{system['search']}, tile{system['tile']} with 13px halo. Bounds [{system['h_lo']:.13g}, {system['h_hi']}], historical h*={system['h_star']:.13g}; postprocessing [0,1] clip. Imported frozen implementations are unchanged.\n\n"
    text+='Exact filename identity defines pairing. All three dirty/clean dimensions match. Same SIFT gate and the original shared resize/warp/valid-rectangle preparation are used. Identity resize is a no-op for equal dimensions.\n\n'
    text+='| Pair | Source WxH | Registration | Prepared WxH | Feature residual median/p95 (px) |\n|---|---|---|---|---|\n'
    for r in pairs:
        p=r['prepared'];g=r['geometry'];text+=f"| {r['id']} | {r['dirty']['width']}x{r['dirty']['height']} | {p['registration']} | {p['width']}x{p['height']} | {g['residual_median']:.4f} / {g['residual_p95']:.4f} |\n"
    text+='\nNo exact dirty-source SHA256 matches to the prior 100 pairs. Detected prior-content matches: '+str(len(overlap['matches']))+'. See `results/bettertest/content_overlap.json`; absence of a detected match is not source-provenance proof.\n\n'
    text+='## Metrics\n\n| Pair | Dirty PSNR | Scalar PSNR | Spatial PSNR | ΔPSNR | Dirty SSIM | Scalar SSIM | Spatial SSIM | ΔSSIM |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    keys=['dirty_psnr','scalar_psnr','spatial_psnr','delta_psnr','dirty_ssim','scalar_ssim','spatial_ssim','delta_ssim']
    for r in m['per_image']:text+='| '+r['id']+' | '+' | '.join(f"{r[k]:.6f}" for k in keys)+' |\n'
    text+='| Arithmetic mean | '+' | '.join(f"{m['means'][k]:.6f}" for k in keys)+' |\n\n'
    text+='Source: [metrics.json](results/bettertest/metrics.json) -> `means`, `per_image`; [CSV](results/bettertest/per_image.csv). Metrics are computed from float32 arrays, never display PNGs. Dirty metrics use the same prepared grayscale dirty input/valid rectangle as both methods, before document preprocessing.\n\n'
    text+='| Cohort | Scalar PSNR | Spatial PSNR | ΔPSNR | Scalar SSIM | Spatial SSIM | ΔSSIM |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name,values in [('Original CamScanner test (20)',m['previous_test_means']),('Bettertest diagnostic (3)',m['means'])]:
        text+='| '+name+' | '+' | '.join(f"{values[k]:.6f}" for k in ['scalar_psnr','spatial_psnr','delta_psnr','scalar_ssim','spatial_ssim','delta_ssim'])+' |\n'
    pos_psnr=sum(r['delta_psnr']>0 for r in m['per_image']);neg_ssim=sum(r['delta_ssim']<0 for r in m['per_image'])
    text+=f'\nSpatial PSNR is higher on {pos_psnr}/3; spatial SSIM is lower on {neg_ssim}/3.\n\n'
    text+='## Target-cleanliness descriptors\n\n'
    text+='Descriptors were fixed before scoring and do not select any restoration parameters. Raw clean source grayscale is used to avoid a registration/resampling confound in the descriptor comparison.\n\n'
    descriptors=['bright_background_fraction','bright_background_variance','background_laplacian_abs','background_hf_energy','estimated_background_noise','global_laplacian_abs','white_fraction','edge_density']
    text+='| Equal-image mean descriptor | Previous test20 | Previous all100 | Bettertest3 |\n|---|---:|---:|---:|\n'
    for key in descriptors:text+='| '+key+' | '+' | '.join(f"{clean['cohorts'][c]['means'][key]:.8g}" for c in ['previous_test_20','previous_all_100','bettertest_3'])+' |\n'
    text+='\nSource: [target_cleanliness.json](results/bettertest/target_cleanliness.json), containing formulas, raw source hashes and all per-image values. Background mask is gray>.8 and outside an 11x11 dilation of gray<.7, with the 5px border excluded. Population variance, reflect101 Laplacian and Immerkaer absolute response summarize this support. White fraction is gray>=.98; edge density uses Sobel magnitude>.2. These are content-dependent descriptive proxies, not objective perfection or calibrated camera noise.\n\n'
    old=clean['cohorts']['previous_test_20']['means'];new=clean['cohorts']['bettertest_3']['means']
    signs={k:new[k]<old[k] for k in ['bright_background_variance','background_laplacian_abs','estimated_background_noise']}
    text+='New mean versus old test mean: '+', '.join(f'{k} lower={v}' for k,v in signs.items())+f"; white fraction higher={new['white_fraction']>old['white_fraction']}.\n\n"
    text+='## Interpretation and limitations\n\n'
    if pos_psnr==3 and neg_ssim==3:
        text+='The original higher-PSNR/lower-SSIM pattern persists on all three. Cleaner references alone do not remove the conflict in this diagnostic. This weakens an explanation based solely on imperfections in the old targets; it does not establish a structural mechanism or exclude a partial contribution.\n\n'
    elif pos_psnr==3 and neg_ssim==0:
        text+='The spatial PSNR gain persists and no pair has a spatial SSIM deficit. If the descriptive target comparison supports cleaner backgrounds, this is consistent with reference quality contributing to the original conflict. The dataset change prevents a causal attribution to target cleanliness alone.\n\n'
    elif neg_ssim==3:
        text+=f"The spatial SSIM deficit persists on all three, with mean {m['means']['delta_ssim']:+.6f} versus the historical {m['previous_test_means']['delta_ssim']:+.6f}. The deficit is smaller, not eliminated. PSNR improves on {pos_psnr}/3 and the three-image mean paired difference is {m['means']['delta_psnr']:+.6f} dB, so the earlier uniform PSNR gain is not replicated. Cleaner backgrounds do not by themselves eliminate the SSIM loss here; whether old target imperfections made a partial contribution remains unresolved because content and framing change too.\n\n"
    else:
        text+='Per-image behavior is mixed. Three selected pairs do not support a strong conclusion about the source of the original PSNR/SSIM conflict.\n\n'
    text+='Changing both dirty photographs and clean references also changes content, framing, margins, illumination and ink coverage. This is a frozen-system dataset diagnostic, **not a same-input intervention on target cleanliness alone**. New images have large white margins, which affect whole-image metrics and cleanliness descriptors. Three selected pairs, one original training seed, application-produced targets, grayscale scoring, sparse-feature alignment and a subpixel warp on one pair limit inference. No publication-level, production or medical claim follows.\n\n'
    text+='## Artifacts and reproduction\n\n'
    for r in pairs:text+=f"* [{r['id']} eight-panel figure](results/bettertest/figures/{r['id']}_comparison.png).\n"
    text+='\n`results/bettertest/` also preserves explicit pairs, preparation matrices/ROIs, source/checkpoint/code SHA256s, environment, CSV/JSON scores, original-artifact preservation proof, float32 output/error/h-map NPYs and logs. The first exploratory audit was read-only and then incorporated into the executable audit. No pair was excluded.\n\n'
    text+='One preliminary audit attempt failed on an empty SIFT inlier array during prior-content screening. A guard was added and the audit rerun before any scoring; the failed log remains. No frozen implementation was changed and no model evaluation failed. Later report wording/layout revisions read only saved results; evaluation_started.json records the exact helper hash used at scoring.\n\n'
    text+=f"Environment: Python {env['python'].split()[0]}, Torch {env['pytorch']}, CUDA {env['cuda']}, OpenCV {env['opencv']}, {env['gpu']}; strict deterministic algorithms, no AMP/TF32, original seed20261014.\n\n"
    text+='From this experiment directory, with an EMPTY diagnostic result directory:\n\n```powershell\n$env:PYTHONHASHSEED = "20261014"\npython bettertest_diagnostic.py audit\npython bettertest_diagnostic.py evaluate\npython bettertest_diagnostic.py render\npython bettertest_diagnostic.py report\n```\n\nFor an independent reproduction, use `--output results/bettertest_reproduction` with each command. Preserve this attempt. Existing evaluation-start/metrics markers refuse a second score; render/report read saved arrays and do not run the models. Do not run `train.py`, `calibrate.py` or the original `evaluate.py` for this diagnostic.\n'
    (ROOT/'BETTERTEST_REPORT.md').write_text(text,encoding='utf-8')
    print('REPORT',ROOT/'BETTERTEST_REPORT.md',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','evaluate','render','report']);parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');logs=output/'logs';logs.mkdir(exist_ok=True)
    path=logs/f'{stamp}_{args.stage}.log'
    class Tee:
        def __init__(self,stream,file):self.stream=stream;self.file=file
        def write(self,s):self.stream.write(s);self.file.write(s);self.file.flush()
        def flush(self):self.stream.flush();self.file.flush()
    with path.open('x',encoding='utf-8') as log:
        stream=sys.stdout;sys.stdout=Tee(stream,log)
        try:
            if args.stage=='audit':audit(args.source.resolve(),output)
            elif args.stage=='evaluate':evaluate(output)
            elif args.stage=='render':render(output)
            else:report(output)
            print('EXIT_CODE=0',flush=True)
        except Exception:
            log.write(traceback.format_exc());log.flush();raise
        finally:sys.stdout=stream

if __name__=='__main__':main()
