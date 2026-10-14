"""Shared geometry preparation, content grouping, and immutable data/crop manifests."""
import json,hashlib
import cv2
import numpy as np
from PIL import Image
from audit_dataset import ROOT,read,gray,digest
from config import SEED,STEPS,BATCH,CROP

def write(name,data):
    (ROOT/'manifests'/name).write_text(json.dumps(data,indent=2),encoding='utf-8')

def main():
    cv2.setNumThreads(1);cv2.setRNGSeed(SEED)
    audit=json.loads((ROOT/'manifests'/'pairs.json').read_text(encoding='utf-8'))
    if audit['unmatched']: raise SystemExit('STOP: unpaired source files')
    rows=[r for r in audit['pairs'] if r['geometry']['valid']]
    # Perceptual candidate generation across ALL source collections, then feature verification.
    parents={r['id']:r['id'] for r in rows};desc={};hashes={}
    def find(a):
        while parents[a]!=a: a=parents[a]
        return a
    def union(a,b):
        aa,bb=find(a),find(b);parents[max(aa,bb)]=min(aa,bb)
    for row in rows:
        g=gray(read(row['clean']['path']));small=cv2.resize(g,(32,32)).astype(np.float32)
        v=cv2.dct(small)[:8,:8].ravel()[1:];hashes[row['id']]=v>np.median(v)
        sift=cv2.SIFT_create(nfeatures=1200);kp,d=sift.detectAndCompute(g,None);desc[row['id']]=(kp,d,g.shape)
    duplicates=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            dist=int(np.count_nonzero(hashes[a['id']]!=hashes[b['id']]))
            if dist>32: continue
            ka,da,sa=desc[a['id']];kb,db,sb=desc[b['id']]
            if da is None or db is None: continue
            good=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.7*n.distance]
            if len(good)<35:continue
            pa=np.float32([ka[m.queryIdx].pt for m in good]);pb=np.float32([kb[m.trainIdx].pt for m in good])
            H,mask=cv2.findHomography(pa,pb,cv2.RANSAC,2.)
            if H is not None and mask.sum()>=35 and mask.mean()>=.65 and mask.sum()>=.12*min(len(ka),len(kb)):
                union(a['id'],b['id']);duplicates.append(dict(a=a['id'],b=b['id'],phash_distance=dist,inliers=int(mask.sum())))
    groups={}
    for row in rows: groups.setdefault(find(row['id']),[]).append(row['id'])
    # Hash order of stable content-group identity, never filesystem iteration order.
    ordered=sorted(groups,key=lambda k:hashlib.sha256(f'{SEED}:{k}'.encode()).hexdigest())
    n=len(ordered);nval=max(1,round(n*.2));ntest=max(1,round(n*.2))
    if n<5: raise SystemExit('Too few independent content groups: infrastructure only, no held-out claims')
    assignments={k:('val' if i<nval else 'test' if i<nval+ntest else 'train') for i,k in enumerate(ordered)}
    partitions={s:[] for s in ['train','val','test']};prepared=[]
    for row in rows:
        split=assignments[find(row['id'])];partitions[split].append(row['id'])
        dest=ROOT/'prepared'/split;dest.mkdir(parents=True,exist_ok=True)
        dirty=gray(read(row['dirty']['path']));clean=gray(read(row['clean']['path']));h,w=clean.shape
        dirty=cv2.resize(dirty,(w,h),interpolation=cv2.INTER_LINEAR)
        box=[0,0,w,h]
        if row['geometry']['registration_required']:
            H=np.array(row['geometry']['homography'])
            dirty=cv2.warpPerspective(dirty,H,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
            valid=cv2.warpPerspective(np.ones((h,w),np.uint8),H,(w,h),flags=cv2.INTER_NEAREST)
            # Conservative common rectangle: shrink sides until every included pixel has source support.
            x1=y1=0;x2=w;y2=h
            for _ in range(w+h):
                sub=valid[y1:y2,x1:x2]
                if sub.all(): break
                fractions=[sub[0].mean(),sub[-1].mean(),sub[:,0].mean(),sub[:,-1].mean()]
                side=int(np.argmin(fractions))
                if side==0:y1+=1
                elif side==1:y2-=1
                elif side==2:x1+=1
                else:x2-=1
            # Remove 2px of interpolation-boundary uncertainty in both images.
            box=[x1+2,y1+2,x2-2,y2-2]
        x1,y1,x2,y2=box;dirty=dirty[y1:y2,x1:x2];clean=clean[y1:y2,x1:x2]
        if min(dirty.shape)<CROP: raise RuntimeError('Prepared region too small: '+row['id'])
        files={}
        for side,array in [('dirty',dirty),('clean',clean)]:
            path=dest/f"{row['id']}_{side}.png";Image.fromarray(array).save(path)
            files[side]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
        prepared.append(dict(id=row['id'],split=split,content_group=find(row['id']),width=dirty.shape[1],height=dirty.shape[0],
            original_clean_roi=box,registration=row['geometry']['registration_required'],**files))
    write('split.json',dict(seed=SEED,rule='sha256(seed:content-group) ordered, 20% val / 20% test / remainder train',
        partitions=partitions,content_groups=groups,duplicate_links=duplicates,
        excluded=[dict(id=r['id'],reason=r['geometry']['reason']) for r in audit['pairs'] if not r['geometry']['valid']],prepared=prepared))
    rng=np.random.default_rng(SEED);train=[r for r in prepared if r['split']=='train'];batches=[]
    for step in range(1,STEPS+1):
        batch=[]
        for _ in range(BATCH):
            row=train[int(rng.integers(len(train)))];batch.append(dict(id=row['id'],x=int(rng.integers(row['width']-CROP+1)),
                y=int(rng.integers(row['height']-CROP+1)),flip_h=bool(rng.integers(2)),flip_v=bool(rng.integers(2))))
        batches.append(dict(step=step,crops=batch))
    write('crops.json',dict(seed=SEED,crop=CROP,batches=batches))
    print(json.dumps(dict(pairs=len(rows),content_groups=n,duplicates=duplicates,counts={s:len(v) for s,v in partitions.items()},registration=sum(r['registration'] for r in prepared))))

if __name__=='__main__':main()
