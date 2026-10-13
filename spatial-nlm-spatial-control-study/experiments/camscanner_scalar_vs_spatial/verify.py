"""Scientific invariants: copied NLM, tile neighborhoods, gradients and shared weights."""
import torch
from common import seed_all
from nlm import nlm,_self_test
from models import HNet
import document_pipeline as pipeline

def main():
    seed_all();_self_test()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    x=torch.rand(1,1,59,67,device=device);h=torch.full_like(x,.12,requires_grad=True)
    pipeline.TILE=24
    tiled=pipeline.nlm_tiled(x,h);direct=nlm(x,h)
    err=float((tiled-direct).abs().max().detach())
    assert err<2e-6,err
    tiled.mean().backward();assert h.grad is not None and h.grad.abs().sum()>0
    spatial=HNet(spatial=True).to(device);scalar=HNet(spatial=False).to(device);scalar.load_state_dict(spatial.state_dict())
    assert all(torch.equal(a,b) for a,b in zip(scalar.parameters(),spatial.parameters()))
    hm=scalar(x,.001,.3);assert float((hm.max()-hm.min()).detach())==0
    print('shared weights identical; scalar constant; tile max error',err,'h gradient nonzero',flush=True)

if __name__=='__main__':main()

