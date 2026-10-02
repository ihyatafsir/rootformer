import json, sys, torch
sys.path.insert(0,'/workspace/hf_v19_2_release')
import scratch_lm as S
D='/workspace/sf_data'
meta=json.load(open(f'{D}/meta.json'))
hold=sorted(meta['hold_roots'])
sizes=(meta['n_roots'],meta['n_awzan'],meta['n_prefixes'],meta['n_suffixes'])
# train() does: torch.manual_seed(0) then MorphemicLM(sizes) -> reproduce that EXACT init
torch.manual_seed(0)
fresh=S.MorphemicLM(sizes).state_dict()
ck=torch.load('/workspace/scratch_lm_run/sf_morph.pt', map_location='cpu', weights_only=True)
hold_t=torch.tensor(hold)
seen=[i for i in range(sizes[0]) if i not in set(hold) and i not in set(meta['special_root_ids'])][:200]
seen_t=torch.tensor(seen)
for name in ('e_r.weight','h_r.weight'):
    a,b=fresh[name],ck[name]
    print(f'{name}:')
    print(f'   held-out rows identical to random init : {torch.equal(a[hold_t], b[hold_t])}'
          f'   (max|delta|={(a[hold_t]-b[hold_t]).abs().max():.3e})')
    print(f'   seen-root rows identical to random init: {torch.equal(a[seen_t], b[seen_t])}'
          f'   (max|delta|={(a[seen_t]-b[seen_t]).abs().max():.3e})')
# how many held-out root tokens were seen as INPUT in train?
tr=torch.load(f'{D}/streams.pt', weights_only=False)['train']['R']
hs=set(hold); n=sum(1 for seq in tr for x in seq if x in hs)
print(f'held-out root tokens in train input stream = {n}')
