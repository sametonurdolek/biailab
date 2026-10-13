import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json,random,sys,platform,hashlib
import numpy as np
import cv2
import torch
from PIL import Image
from config import ROOT,SEED
from audit_dataset import digest

def seed_all():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cv2.setNumThreads(1);cv2.setRNGSeed(SEED);torch.set_num_threads(4)

def load_json(name):return json.loads((ROOT/'manifests'/name).read_text(encoding='utf-8'))
def save_json(path,value):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def load_pair(row):
    arrays=[]
    for side in ['dirty','clean']:
        file=ROOT/row[side]['path']
        if digest(file)!=row[side]['sha256']:raise RuntimeError('Prepared data modified: '+str(file))
        arrays.append(np.asarray(Image.open(file),dtype=np.float32)/255.)
    return arrays

def environment():
    return dict(python=sys.version,executable=sys.executable,os=platform.platform(),numpy=np.__version__,
        opencv=cv2.__version__,pytorch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        seeds=dict(python=SEED,numpy=SEED,torch=SEED,cuda=SEED),deterministic_algorithms=True,
        cudnn_benchmark=False,tf32=False,cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        precision='float32, no AMP',limitations='Deterministic algorithms enforced; numerical equality across different hardware/software is not promised.',
        source_hashes={p.name:digest(p) for p in sorted(ROOT.glob('*.py'))})

