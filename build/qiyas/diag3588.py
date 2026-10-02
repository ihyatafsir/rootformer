import json,sys,collections
sys.path.insert(0,'/workspace/hf_v19_2_release'); sys.path.insert(0,'/workspace/hf_v19_2_release/models'); sys.path.insert(0,'/workspace/qiyas')
from qiyas_engine import Qiyas
import nrmp_vocab as nv
V=next(v for k,v in vars(nv).items() if isinstance(v,type) and 'MorphemicVocab' in k)
vocab=V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
gold=json.load(open('/workspace/qiyas/deriv_gold.json',encoding='utf-8'))
pos=gold['positions']; hold=sorted({x['root'] for x in pos})
du=json.load(open('/workspace/lisan_root_fix/du_after.json',encoding='utf-8'))['words']
bok={w:bool(v[0]) for w,v in du.items()}
obs=[]
for w in du:
    p_id,r_id,wz_id,s_id=vocab.encode_word(w)
    r=vocab.id2root[r_id]; wz=vocab.id2wazn[wz_id]
    p=vocab.id2prefix[p_id]; s=vocab.id2suffix[s_id]
    p='' if p in ('<NONE>','<PAD>','<UNK>') else p
    s='' if s in ('<NONE>','<PAD>','<UNK>') else s
    if r.startswith('<') or not r or r in set(hold): continue
    if not wz or wz in ('<NONE>','<PAD>','<UNK>'): continue
    if not bok.get(w): continue
    if len(p)+len(s)>len(w): continue
    stem=w[len(p):len(w)-len(s)] if s else w[len(p):]
    if not stem or p+stem+s!=w: continue
    obs.append((r,wz,stem))
q=Qiyas('strict7');
for r,w,s in obs: q.observe(r,w,s)
q.induce()
print('asl',len(obs))
fails=[]; oks=0
for x in pos:
    w,r,wz,p,s=x['word'],x['root'],x['wazn'],x['prefix'],x['suffix']
    res=q.realize(r,wz)
    stem=w[len(p):len(w)-len(s)] if s else w[len(p):]
    if res and p+res[0]+s==w: oks+=1
    elif len(fails)<25: fails.append((w,r,wz,p,s,stem,res[0] if res else None,res[1] if res else None))
print('ok',oks,'/',len(pos))
print('--- failures ---')
for f in fails: print(f)
print('--- ok sample (wazn agreement) ---')
n_du=collections.Counter()
for x in pos:
    n_du[(x['wazn'], x['wazn'] in vocab.wazn2id)] += 1
print('wazn values not in vocab:', sum(1 for x in pos if x['wazn'] not in vocab.wazn2id))
print('top wazn:', collections.Counter(x['wazn'] for x in pos).most_common(8))
# how many positions' gold word is in du_after?
print('gold words present in du_after keys:', sum(1 for x in pos if x['word'] in du), '/', len(pos))
