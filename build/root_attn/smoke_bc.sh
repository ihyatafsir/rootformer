#!/bin/bash
set -u
cd /workspace/hf_v19_2_release || exit 90
PY=/workspace/venvs/rootformer/bin/python
CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
OUT=/workspace/root_attn/smoke
SM=/workspace/discrete_path/smoke_cache
FLAGS="--feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln"
COMMON="--checkpoint $CK --cache $SM --head-init remap --steps 40 --batch-size 8 --eval-every 40 --train-windows 13 --val-windows 13 --seed 7 $FLAGS"
echo "=== (b) PATCHED trainer, default flags, FINAL code ==="
$PY nrmt_train.py $COMMON --tag SMOKE_DEF --probe-out $OUT/trace_SMOKE_DEF.jsonl --out $OUT/res_SMOKE_DEF.json --save $OUT/head_SMOKE_DEF.pt > $OUT/log_SMOKE_DEF.txt 2>&1
echo "rc=$?"
echo "=== (c) PATCHED trainer, LIVE: RCA top4 + unfreeze top4 ==="
$PY nrmt_train.py $COMMON --tag SMOKE_LIVE --probe-out $OUT/trace_SMOKE_LIVE.jsonl --out $OUT/res_SMOKE_LIVE.json --save $OUT/head_SMOKE_LIVE.pt --root-cross-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.1 --rca-lr-scale 1.0 --rca-ablate-eval --h-drift-probe 5 > $OUT/log_SMOKE_LIVE.txt 2>&1
echo "rc=$?"
echo "=== TRACE EQUIVALENCE (a) ORIGINAL vs (b) PATCHED-DEFAULT ==="
$PY - <<'PY'
import json
def load(p): return [json.loads(l) for l in open(p)]
a=load('/workspace/root_attn/smoke/trace_SMOKE_ORIG.jsonl')
b=load('/workspace/root_attn/smoke/trace_SMOKE_DEF.jsonl')
keys=['step','loss','root','wazn','prefix','suffix','grad_norm','lr','p_ss','skipped']
bad=[(x['step'],k,x.get(k),y.get(k)) for x,y in zip(a,b) for k in keys if x.get(k)!=y.get(k)]
print('rows',len(a),len(b),'mismatched fields:',len(bad))
print('IDENTICAL' if not bad and len(a)==len(b)==40 else ('DIFFS '+str(bad[:5])))
PY
echo "=== LIVE EVIDENCE ==="
grep -E "TRUNK LIVE|trainable parameters|optimizer groups|RCA init|\[proof\]|\[eval @" $OUT/log_SMOKE_LIVE.txt | head -25
echo "=== LIVE nan/inf ==="
grep -ciE "nan|inf" $OUT/trace_SMOKE_LIVE.jsonl
echo SMOKE_BC_DONE
