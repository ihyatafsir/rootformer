import collections, json, random, re, sys
sys.path.insert(0,'/workspace/hf_v19_2_release'); sys.path.insert(0,'/workspace/hf_v19_2_release/models'); sys.path.insert(0,'/workspace/qiyas')
from qiyas_engine import Qiyas, ILLAS
import nrmp_vocab as nv
V=next(v for k,v in vars(nv).items() if isinstance(v,type) and 'MorphemicVocab' in k)
vocab=V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
SPEC=('<NONE>','<PAD>','<UNK>','')
af=lambda x:'' if (x is None or x in SPEC) else x
du=json.load(open('/workspace/lisan_root_fix/du_after.json',encoding='utf-8'))['words']
bok={w:bool(v[0]) for w,v in du.items()}; types=list(du.keys())
gold=json.load(open('/workspace/qiyas/deriv_gold.json',encoding='utf-8'))
PUNCT=re.compile(r'[^\u0621-\u064a\u0670\u064b-\u0652]')
pos=[{'word':x['word'],'root':x['root'],'wazn':x['wazn'],'prefix':af(x['prefix']),'suffix':af(x['suffix'])} for x in gold['positions']]
hold=sorted({x['root'] for x in pos}); hset=set(hold)
obs=[]
for w in types:
    p_id,r_id,wz_id,s_id=vocab.encode_word(w)
    r=vocab.id2root[r_id]; wz=vocab.id2wazn[wz_id]
    p=af(vocab.id2prefix[p_id]); s=af(vocab.id2suffix[s_id])
    if r.startswith('<') or not r or r in hset: continue
    if not wz or wz in ('<NONE>','<PAD>','<UNK>'): continue
    if not bok.get(w): continue
    if len(p)+len(s)>len(w): continue
    stem=w[len(p):len(w)-len(s)] if s else w[len(p):]
    if not stem or p+stem+s!=w: continue
    obs.append((r,wz,stem))
q=Qiyas('strict7',backoff=True)
for r,w,s in obs: q.observe(r,w,s)
q.induce()

def score(plist,label):
    S=collections.Counter(); prov=collections.Counter()
    for x in plist:
        w,r,wz,p,s=x['word'],x['root'],x['wazn'],x['prefix'],x['suffix']
        base=vocab.base_tok.realize_root_and_wazn(r,wz); b=(p+base+s==w)
        res=q.realize(r,wz)
        if res is None: qo=False; pv='NONE'
        else:
            qy,pv,su,tot,d=res; qo=(p+qy+s==w)
        S['b']+=b; S['q']+=qo; prov[pv.split(':')[0]]+=1
    n=len(plist)
    print('%-28s n=%d  shipped=%.4f%%  qiyas=%.4f%%  delta=%+d  prov=%s'%(
      label,n,100*S['b']/n,100*S['q']/n,S['q']-S['b'],dict(prov)))
    return S
score(pos,'ALL 3588 (as scored)')
clean=[x for x in pos if not PUNCT.search(x['word'])]
score(clean,'punctuation-free subset')

# ---- Task B on the punctuation-free subset
wazn_all=[w for w in vocab.awzan_list if isinstance(w,str) and not w.startswith('<')]
grid={}
for r in hold:
    row={}
    for wz in wazn_all:
        rr=q.realize(r,wz)
        if rr is not None: row[wz]=rr[:5]
    grid[r]=row
def ident(plist):
    t1=t5=0
    for x in plist:
        w,gr,p,sf=x['word'],x['root'],x['prefix'],x['suffix']
        c=[]
        for r in hold:
            for wz,rr in grid[r].items():
                if (p+rr[0]+sf)==w or rr[0]==w: c.append(r)
        seen=[]
        for r in c:
            if r not in seen: seen.append(r)
        if gr in seen:
            rk=seen.index(gr)+1
            if rk==1: t1+=1
            if rk<=5: t5+=1
    n=len(plist)
    print('TaskB %-24s n=%d top1=%.4f%% top5=%.4f%%'%('',n,100*t1/n,100*t5/n))
ident(pos); ident(clean)

# ---- TADDI (4th condition): held-out (root,wazn) PAIRS, roots seen
rng=random.Random(5)
pairs=sorted({(r,wz) for r,wz,_ in obs})
rng.shuffle(pairs); k=int(0.10*len(pairs)); test=set(pairs[:k]); train=pairs[k:]
train_obs=[o for o in obs if (o[0],o[1]) not in test]
test_obs=[o for o in obs if (o[0],o[1]) in test]
print('\nTier-1 pair holdout: %d train pairs, %d held-out pairs, %d held-out observations'%(
  len(train),len(test),len(test_obs)))
for name in ('identity','wazn_only','coarse_weak','strict7','positional'):
    qq=Qiyas(name,backoff=True)
    for r,w,s in train_obs: qq.observe(r,w,s)
    qq.induce()
    prim=0; miss=0; ok=0; n=0
    for r,wz,s in test_obs:
        n+=1
        rule=qq.rule(r,wz)
        if rule is None: miss+=1; continue
        if rule[1].split(':')[0]==name and not rule[1].startswith('GLOBAL'): prim+=1
        rr=qq.realize(r,wz)
        if rr is not None and rr[0]==s: ok+=1
    print('  %-11s primary-ʿilla firing (taʿaddī)=%.4f  realisation=%.4f  no-analogue=%.4f'%(
      name,prim/n,ok/n,miss/n))
