import json, sys, torch
sys.path.insert(0,'/workspace/scratch_curve'); sys.path.insert(0,'/workspace/hf_v19_2_release'); sys.path.insert(0,'/workspace/hf_v19_2_release/models')
import curve_lm as C, scratch_lm as S
D='/workspace/sf_data'; meta=json.load(open(f'{D}/meta.json'))
data=torch.load(f'{D}/streams.pt', weights_only=False)
dev=torch.device('cpu')
m=S.MorphemicLM((meta['n_roots'],meta['n_awzan'],meta['n_prefixes'],meta['n_suffixes'])).to(dev)
m.load_state_dict(torch.load('/workspace/scratch_lm_run/sf_morph.pt',map_location=dev,weights_only=True)); m.eval()
ht=torch.zeros(meta['n_roots'],dtype=torch.bool)
for i in meta['hold_roots']: ht[i]=True
for ns in (8000,):
    r=C.held_out_metrics(m,data,meta['n_roots'],meta['special_root_ids'],dev,ns,seed=11,hold_ids=ht)
    rc=r['root_cats']
    print("n_sent=%d: root t1 %.2f | heldout n=%d t1 %.3f | real_seen n=%d t1 %.3f (pub 1.468) | special n=%d t1 %.3f (pub 39.202)"%(
        ns, r['r']['top1']*100, rc['heldout_root']['n'], rc['heldout_root']['top1']*100,
        rc['real_seen_root']['n'], rc['real_seen_root']['top1']*100,
        rc['special']['n'], rc['special']['top1']*100))
