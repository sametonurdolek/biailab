"""Frozen test visualizations: first, best, median, worst paired PSNR; all rows available."""
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ROOT
from common import load_json,load_pair,save_json

def main():
    metrics=json.loads((ROOT/'results'/'metrics.json').read_text());scores=metrics['per_image']
    order=sorted(scores,key=lambda r:(r['delta_psnr'],r['image']))
    reps=list(dict.fromkeys([sorted(scores,key=lambda r:r['image'])[0]['image'],order[-1]['image'],order[len(order)//2]['image'],order[0]['image']]))
    rows={r['id']:r for r in load_json('split.json')['prepared']};cal=load_json('calibration.json');artifacts=[]
    # Precommitted from the dataset audit, before test scoring: two triangle diagrams.
    diagram_id='camscanner_004'
    # Render every test image so losses are available as readily as wins.
    for name in sorted(rows):
        if rows[name]['split']!='test':continue
        dirty,clean=load_pair(rows[name]);arrays=[dirty,clean]
        for model in ['scalar','spatial']: arrays.append(np.load(ROOT/'results'/'outputs'/f'{name}_{model}.npy'))
        arrays.extend([np.abs(arrays[2]-clean),np.abs(arrays[3]-clean)])
        arrays.extend([np.load(ROOT/'results'/'h_maps'/f'{name}_{model}.npy') for model in ['scalar','spatial']])
        titles=['Dirty (prepared geometry)','Clean target','Scalar output','Spatial output','Scalar absolute error','Spatial absolute error','Scalar broadcast h','Spatial h(x,y)']
        page_h,page_w=clean.shape
        figure_height=max(5.5,min(18.,8*page_h/page_w+1.5))
        fig,axes=plt.subplots(2,4,figsize=(16,figure_height),constrained_layout=True)
        for i,ax in enumerate(axes.flat):
            if i<4:ax.imshow(arrays[i],cmap='gray',vmin=0,vmax=1)
            elif i<6:error_artist=ax.imshow(arrays[i],cmap='magma',vmin=0,vmax=.5)
            else:h_artist=ax.imshow(arrays[i],cmap='viridis',vmin=cal['h_lo'],vmax=cal['h_hi'])
            ax.set_title(titles[i]);ax.axis('off')
        fig.colorbar(error_artist,ax=list(axes.flat)[4:6],location='bottom',shrink=.7,label='Absolute error (same range)')
        fig.colorbar(h_artist,ax=list(axes.flat)[6:8],location='bottom',shrink=.7,label='Bandwidth h (same range)')
        row=next(r for r in scores if r['image']==name)
        fig.suptitle(f'{name} | spatial - scalar PSNR = {row["delta_psnr"]:+.4f} dB')
        path=ROOT/'results'/'figures'/f'{name}_comparison.png';fig.savefig(path,dpi=140);plt.close(fig)
        if name==diagram_id:
            hh,ww=clean.shape;x1=int(.38*ww);x2=int(.98*ww);y1=int(.05*hh);y2=int(.95*hh)
            df,da=plt.subplots(1,4,figsize=(14,5),constrained_layout=True)
            for i,ax in enumerate(da):ax.imshow(arrays[i][y1:y2,x1:x2],cmap='gray',vmin=0,vmax=1);ax.set_title(titles[i]);ax.axis('off')
            df.suptitle('Precommitted diagram crop: '+name)
            df.savefig(ROOT/'results'/'figures'/f'{name}_diagrams.png',dpi=160);plt.close(df)
        if name not in reps:continue
        h,w=clean.shape;s=min(128,h,w);edge=np.hypot(cv2.Sobel(clean,-1,1,0),cv2.Sobel(clean,-1,0,1))
        # ROIs selected solely from target geometry, not model error/gain; fixed rule.
        edge_score=cv2.boxFilter(edge,-1,(s,s));bg_score=cv2.boxFilter(clean,-1,(s,s))-cv2.boxFilter(edge,-1,(s,s))
        interior=(slice(s//2,h-s//2),slice(s//2,w-s//2))
        boxes=[]
        for title,score in [('text_edges',edge_score),('background',bg_score)]:
            yy,xx=np.unravel_index(np.argmax(score[interior]),score[interior].shape);yy+=s//2;xx+=s//2
            boxes.append((title,xx-s//2,yy-s//2))
        # Thin-stroke center: maximum edge energy among pixels in low-width ink regions.
        ink=(clean<.5).astype(np.uint8);width=cv2.distanceTransform(ink,cv2.DIST_L2,3)
        thin=(width>0)&(width<=1.5);thin_score=cv2.boxFilter((thin*edge).astype('float32'),-1,(s,s))
        yy,xx=np.unravel_index(np.argmax(thin_score[interior]),thin_score[interior].shape);boxes.append(('thin_strokes',xx,yy))
        # Fixed center context, includes diagrams when they occupy the page center.
        boxes.append(('center_context',max(0,(w-s)//2),max(0,(h-s)//2)))
        fig,axes=plt.subplots(4,4,figsize=(12,12),constrained_layout=True)
        for j,(label,x,y) in enumerate(boxes):
            x=min(max(x,0),w-s);y=min(max(y,0),h-s)
            for i in range(4):
                axes[j,i].imshow(arrays[i][y:y+s,x:x+s],cmap='gray',vmin=0,vmax=1);axes[j,i].axis('off')
                axes[j,i].set_title(label.replace('_',' ')+' / '+['Dirty','Clean','Scalar','Spatial'][i],fontsize=10)
        path2=ROOT/'results'/'figures'/f'{name}_closeups.png';fig.savefig(path2,dpi=160);plt.close(fig)
        artifacts.append(dict(image=name,comparison=str(path.relative_to(ROOT)),closeups=str(path2.relative_to(ROOT)),roi_rule='max clean edge, max bright/low-edge background, max thin-stroke edge, fixed center'))
    save_json('results/representatives.json',dict(rule='first test filename, best/upper-median/worst delta PSNR; every test also has a comparison figure',representatives=reps,artifacts=artifacts,
        diagram_example=dict(image=diagram_id,path=f'results/figures/{diagram_id}_diagrams.png',rule='precommitted before test; x [.38,.98], y [.05,.95] of prepared page')))
    print('representatives',reps)

if __name__=='__main__':main()
