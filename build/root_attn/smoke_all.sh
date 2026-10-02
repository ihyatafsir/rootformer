#!/bin/bash
set -u
cd /workspace/hf_v19_2_release || exit 90
PY=/workspace/venvs/rootformer/bin/python
CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
OUT=/workspace/root_attn/smoke; mkdir -p $OUT
SM=/workspace/discrete_path/smoke_cache
FLAGS="--feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln"
COMMON="--checkpoint $CK --cache $SM --head-init remap --steps 40 --batch-size 8 --eval-every 40 --train-windows 13 --val-windows 13 --seed 7 $FLAGS"
echo "=== (a) ORIGINAL trainer, default flags ==="
$PY nrmt_train_ORIGINAL.py $COMMON --tag SMOKE_ORIG --probe-out $OUT/trace_SMOKE_ORIG.jsonl --out $OUT/res_SMOKE_ORIG.json --save $OUT/head_SMOKE_ORIG.pt > $OUT/log_SMOKE_ORIG.txt 2>&1
echo "rc=$?"
echo "=== (b) PATCHED trainer, default flags (must be identical) ==="
$PY nrmt_train.py $COMMON --tag SMOKE_DEF --probe-out $OUT/trace_SMOKE_DEF.jsonl --out $OUT/res_SMOKE_DEF.json --save $OUT/head_SMOKE_DEF.pt > $OUT/log_SMOKE_DEF.txt 2>&1
echo "rc=$?"
echo "=== (c) PATCHED trainer, LIVE: RCA top4 + unfreeze top4 ==="
$PY nrmt_train.py $COMMON --tag SMOKE_LIVE --probe-out $OUT/trace_SMOKE_LIVE.jsonl --out $OUT/res_SMOKE_LIVE.json --save $OUT/head_SMOKE_LIVE.pt --root-cross-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.1 --rca-lr-scale 1.0 --rca-ablate-eval --h-drift-probe 5 > $OUT/log_SMOKE_LIVE.txt 2>&1
echo "rc=$?"
echo "=== TRACE EQUIVALENCE (a) vs (b) ==="
$PY - <<'PY'
import json
def load(p):
    return [json.loads(l) for l in open(p)]
a=load('/workspace/root_attn/smoke/trace_SMOKE_ORIG.jsonl')
b=load('/workspace/root_attn/smoke/trace_SMOKE_DEF.jsonl')
keys=['step','loss','root','wazn','prefix','suffix','grad_norm','lr','p_ss','skipped']
print('rows', len(a), len(b))
bad=0
for x,y in zip(a,b):
    for k in keys:
        if x.get(k)!=y.get(k):
            print('DIFF step',x['step'],k,x.get(k),y.get(k)); bad+=1
print('mismatched fields:',bad)
print('IDENTICAL' if bad==0 and len(a)==len(b)==40 else 'NOT IDENTICAL')
PY
echo "=== LIVE EVIDENCE ==="
grep -E "TRUNK LIVE|trainable parameters|optimizer groups|RCA init|proof|eval @" $OUT/log_SMOKE_LIVE.txt | head -20
echo SMOKE_ALL_DONE
