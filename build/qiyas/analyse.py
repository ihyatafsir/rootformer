import collections, json, sys, time
sys.path.insert(0,'/workspace/hf_v19_2_release'); sys.path.insert(0,'/workspace/hf_v19_2_release/models'); sys.path.insert(0,'/workspace/qiyas')
from qiyas_engine import Qiyas, ILLAS
import qiyas_wire, nrmp_vocab as nv
V=next(v for k,v in vars(nv).items() if isinstance(v,type) and 'MorphemicVocab' in k)
vocab=V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
SPEC=('<NONE>','<PAD>','<UNK>','')
af=lambda x:'' if (x is None or x in SPEC) else x
du=json.load(open('/workspace/lisan_root_fix/du_after.json',encoding='utf-8'))['words']
bok={w:bool(v[0]) for w,v in du.items()}; types=list(du.keys())
gold=json.load(open('/workspace/qiyas/deriv_gold.json',encoding='utf-8'))
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
print('asl',len(obs))

# --- (a)/(b) task A breakdown
fix=collections.Counter(); fail=collections.Counter(); fail_ex=[]; noan=[]
hamzaish=0
for x in pos:
    w,r,wz,p,s=x['word'],x['root'],x['wazn'],x['prefix'],x['suffix']
    base=vocab.base_tok.realize_root_and_wazn(r,wz); b_ok=(p+base+s==w)
    res=q.realize(r,wz)
    if res is None:
        noan.append((w,r,wz)); continue
    qy,pv,sup,tot,dist=res; q_ok=(p+qy+s==w)
    if q_ok and not b_ok:
        fix[(r,wz)]+=1
        d=(p+qy+s, p+base+s)
        if abs(len(d[0])-len(d[1]))<=1 and (d[0].replace('أ','ا').replace('إ','ا').replace('آ','ا')==d[1].replace('أ','ا').replace('إ','ا').replace('آ','ا') or d[0].replace('ا','')==d[1].replace('أ','')):
            hamzaish+=1
    elif not q_ok:
        fail[(r,wz)]+=1
        if len(fail_ex)<25: fail_ex.append((w,r,wz,p,s,'shipped='+p+base+s,'qiyas='+p+qy+s,pv))
print('fixes',sum(fix.values()),'hamza/orthography-shaped',hamzaish)
print('fix top cells'); [print('  ',k,v) for k,v in fix.most_common(8)]
print('no-analogue',len(noan)); print('  sample',noan[:8])
print('remaining failures',sum(fail.values()))
print('fail top cells'); [print('  ',k,v) for k,v in fail.most_common(12)]
print('fail examples'); [print('  ',e) for e in fail_ex[:14]]

# --- (c) round-trip restricted to held-out-root types + overall, with the hook
qfull=q
def run(mode, subset):
    qiyas_wire.install(vocab.base_tok, enabled=False, engine=qfull, mode=mode, log_cap=0)
    off=sum(1 for w in subset if vocab.decode_word(*vocab.encode_word(w))==w)
    qiyas_wire.set_enabled(vocab.base_tok, True)
    on=sum(1 for w in subset if vocab.decode_word(*vocab.encode_word(w))==w)
    st=qiyas_wire.stats(vocab.base_tok)
    qiyas_wire.uninstall(vocab.base_tok)
    return off,on,st['n_replaced']
sub_hold=[w for w in types if vocab.id2root[vocab.encode_word(w)[1]] in hset]
sub_rest=[w for w in types if vocab.id2root[vocab.encode_word(w)[1]] not in hset]
print('types with a held-out root:',len(sub_hold),' rest:',len(sub_rest))
for mode in ('fallback','replace'):
    for nm,sub in (('HELDOUT-ROOT TYPES',sub_hold),('REST',sub_rest),('ALL',types)):
        off,on,rep=run(mode,sub)
        print('  %-9s %-19s off=%d on=%d delta=%+d (%.4f%% -> %.4f%%) replaced=%d'%(
          mode,nm,off,on,on-off,100*off/len(sub),100*on/len(sub),rep))
