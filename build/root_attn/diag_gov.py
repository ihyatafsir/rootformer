import sys
import numpy as np, torch
sys.path.insert(0,'/workspace/hf_v19_2_release'); sys.path.insert(0,'/workspace/hf_v19_2_release/models')
import nrmp_vocab as nv
WIN, STRIDE = 128, 64
def starts_for(t,n,seed):
    s=list(range(0,t.numel()-WIN-1,STRIDE)); np.random.default_rng(seed).shuffle(s); return sorted(s[:n])
V=next(v for k,v in vars(nv).items() if isinstance(v,type) and 'MorphemicVocab' in k)
vocab=V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
tr4=[t.long() for t in torch.load('/workspace/head_fix/nrmp_cache_9490_aligned/train.pt',map_location='cpu')]
va4=[t.long() for t in torch.load('/workspace/head_fix/nrmp_cache_9490_aligned/val.pt',map_location='cpu')]
tr_st=starts_for(tr4[1],3000,0); va_st=starts_for(va4[1],300,1)
def wt(t,st): return torch.stack([t[j:j+WIN] for j in st])
Pr,Tr,Wr,Sr=(wt(tr4[i],tr_st) for i in (0,1,2,3))
Pv,Tv,Wv,Sv=(wt(va4[i],va_st) for i in (0,1,2,3))
from sibawayh_governor import SibawayhGovernor
def op_of(gov,P,R,W,n):
    rows=[gov.op_ids_for_ids(rr,ww,pp) for pp,rr,ww in zip(P.tolist()[:n],R.tolist()[:n],W.tolist()[:n])]
    return np.array(rows)
g_fresh=SibawayhGovernor(vocab); fresh=op_of(g_fresh,Pv,Tv,Wv,20)
g_used=SibawayhGovernor(vocab)
tr=op_of(g_used,Pr,Tr,Wr,3000)
after=op_of(g_used,Pv,Tv,Wv,20)
g_fresh2=SibawayhGovernor(vocab); fresh2=op_of(g_fresh2,Pv,Tv,Wv,20)
print('train op rows shape',tr.shape)
print('val op ids: fresh == fresh(2nd instance):', np.array_equal(fresh,fresh2))
print('val op ids: fresh == after-train-call    :', np.array_equal(fresh,after))
print('mismatch count:', int((fresh!=after).sum()), 'of', fresh.size)
print('sample fresh[:2,50:60]', fresh[:2,50:60].tolist())
print('sample after[:2,50:60]', after[:2,50:60].tolist())
