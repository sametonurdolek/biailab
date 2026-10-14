"""Read-only source audit; explicit filename proposals verified with image features."""
from pathlib import Path
import hashlib, json, re, sys
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / 'camscanner'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB'))

def gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def properties(path):
    rgb = read(path); g = gray(rgb).astype(np.float32)/255
    bg = g[g >= np.quantile(g,.75)]; ink = g[g <= np.quantile(g,.1)]
    lap = cv2.filter2D(g,-1,np.array([[1,-2,1],[-2,4,-2],[1,-2,1]],np.float32),borderType=cv2.BORDER_REFLECT_101)
    gx = cv2.Sobel(g,-1,1,0); gy = cv2.Sobel(g,-1,0,1)
    with Image.open(path) as im: mode, fmt = im.mode, im.format
    return dict(path=str(path),sha256=digest(path),width=rgb.shape[1],height=rgb.shape[0],mode=mode,format=fmt,
        decoded_dtype=str(rgb.dtype),bits_per_channel=8,channels=3,
        mean=float(g.mean()),std=float(g.std()),background_top_quartile_mean=float(bg.mean()),
        background_top_quartile_std=float(bg.std()),ink_bottom_decile_mean=float(ink.mean()),
        quantiles=np.quantile(g,[0,.01,.1,.5,.9,.99,1]).tolist(),
        near_white_fraction=float((g>=.98).mean()),near_black_fraction=float((g<=.02).mean()),
        edge_mean=float(np.hypot(gx,gy).mean()),local_contrast=float((cv2.dilate(g,np.ones((7,7),np.uint8))-cv2.erode(g,np.ones((7,7),np.uint8))).mean()),
        noise_laplacian=float(np.abs(lap).mean()*np.sqrt(np.pi/2)/6),
        histogram=np.histogram(g,bins=32,range=(0,1))[0].tolist(),
        mean_channel_spread=float((rgb.max(2).astype(float)-rgb.min(2)).mean()/255))

def correspondence(dirty, clean):
    """Gate thresholds fixed before fitting any restoration or seeing test scores."""
    d=gray(read(dirty)); c=gray(read(clean)); h,w=c.shape
    d=cv2.resize(d,(w,h),interpolation=cv2.INTER_LINEAR)
    sift=cv2.SIFT_create(nfeatures=6000)
    kd,dd=sift.detectAndCompute(d,None); kc,dc=sift.detectAndCompute(c,None)
    if dd is None or dc is None: return dict(valid=False,reason='no descriptors')
    matches=cv2.BFMatcher().knnMatch(dd,dc,k=2)
    good=[a for a,b in matches if a.distance < .7*b.distance]
    if len(good)<30: return dict(valid=False,reason='fewer than 30 feature matches',matches=len(good))
    p=np.float32([kd[m.queryIdx].pt for m in good]);q=np.float32([kc[m.trainIdx].pt for m in good])
    H,mask=cv2.findHomography(p,q,cv2.RANSAC,2.,maxIters=5000,confidence=.999)
    if H is None: return dict(valid=False,reason='homography failed')
    inliers=mask.ravel().astype(bool)
    projected=cv2.perspectiveTransform(p[:,None,:],H)[:,0,:]
    residual=np.linalg.norm(projected-q,axis=1)[inliers]
    identity=np.linalg.norm(p-q,axis=1)[inliers]
    pts=p[inliers]; coverage=(np.ptp(pts[:,0])/w)*(np.ptp(pts[:,1])/h)
    # Need broad page support and subpixel correspondence, not just a repeated word.
    valid=int(inliers.sum())>=30 and inliers.mean()>=.5 and np.median(residual)<=.6 and np.quantile(residual,.95)<=1.5 and coverage>=.2
    registration=bool(np.median(identity)>.5 or np.quantile(identity,.95)>1.25)
    return dict(valid=bool(valid),reason='accepted' if valid else 'feature geometry gate failed',matches=len(good),
        inliers=int(inliers.sum()),inlier_fraction=float(inliers.mean()),coverage=float(coverage),
        residual_median=float(np.median(residual)),residual_p95=float(np.quantile(residual,.95)),
        resized_identity_median=float(np.median(identity)),resized_identity_p95=float(np.quantile(identity,.95)),
        registration_required=registration,homography=H.tolist())

def main():
    cv2.setNumThreads(1);cv2.setRNGSeed(20261014)
    (ROOT/'manifests').mkdir(exist_ok=True);(ROOT/'results'/'figures').mkdir(parents=True,exist_ok=True)
    folders=[DATA]+[x for x in sorted(DATA.iterdir()) if x.is_dir() and (x/'dirty').is_dir() and (x/'clean').is_dir()]
    pairs=[];missing=[];inventories=[]
    for group in folders:
        dirty=sorted((group/'dirty').glob('*.jpg'));clean=list((group/'clean').glob('*.jpg'))
        by_number={}
        for f in clean:
            match=re.search(r'_(\d+)\.jpg$',f.name,re.I)
            if match: by_number.setdefault(int(match[1]),[]).append(f)
        inventories.append(dict(group=group.name,dirty_count=len(dirty),clean_count=len(clean)))
        for ordinal,d in enumerate(dirty,1):
            index=int(d.stem) if d.stem.isdigit() else ordinal
            candidates=by_number.get(index,[])
            if len(candidates)!=1:
                missing.append(dict(dirty=str(d),index=index,candidates=[str(c) for c in candidates]));continue
            c=candidates[0];geom=correspondence(d,c)
            row=dict(id=f'{group.name}_{index:03}',group=group.name,index=index,dirty=properties(d),clean=properties(c),geometry=geom,
                pairing_rule='numeric dirty identity / terminal clean suffix' if d.stem.isdigit() else 'natural source pair identifiers rank / terminal clean suffix; content verified by SIFT')
            pairs.append(row)
            print(row['id'],geom['valid'],geom.get('inliers'),geom.get('resized_identity_median'),flush=True)
        used={row['clean']['path'] for row in pairs}
        missing.extend(dict(unmatched_clean=str(c)) for c in clean if str(c) not in used)
    manifest=dict(status='audited',inventories=inventories,pairs=pairs,unmatched=missing,
        geometry_gate='SIFT ratio .7; homography RANSAC 2 px; >=30 inliers; fraction >=.5; coverage >=.2; median residual <=.6px; p95 <=1.5px',
        warning='No pairing by arbitrary filesystem order. Nested collection included explicitly. Geometry estimates are offline shared preparation, not model-specific alignment.')
    (ROOT/'manifests'/'pairs.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    # Numeric identities, not lexicographic clean ordering.
    for start in range(0,len(pairs),20):
        sheet=Image.new('RGB',(1000,10*150),'#ddd');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(pairs[start:start+20]):
            y=(i//2)*150;x=(i%2)*500
            for j,side in enumerate(['dirty','clean']):
                im=Image.fromarray(read(row[side]['path']));im.thumbnail((245,125));sheet.paste(im,(x+j*250,y+22))
            draw.text((x,y),row['id']+' valid='+str(row['geometry']['valid']),fill='black')
        sheet.save(ROOT/'results'/'figures'/f'audit_pairs_{start:03}.jpg')
    print(json.dumps(dict(total=len(pairs),valid=sum(r['geometry']['valid'] for r in pairs),unmatched=len(missing))))
    if missing: raise SystemExit('STOP: missing or ambiguous pairs; see pairs.json')

if __name__=='__main__': main()
