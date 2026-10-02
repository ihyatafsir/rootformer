#!/bin/bash
# chain_final.sh -- final measurement, three stages, ONE GPU process at a time.
#
#   stage 0  default-path equivalence: ORIGINAL nrmt_train.py vs the patched one, 40 steps on
#            the smoke cache, seeded.  Must be field-for-field IDENTICAL (this is the guarantee
#            that --root-cross-attn none --unfreeze-last 0 preserves the FIX baseline).
#   stage 1  RUN A2: RCA top4 + UNFREEZE top4, STABILISED.
#            lr split: head 1e-3 | RCA 3e-4 (0.3x) | trunk 1e-5 (0.01x)
#            out-norm ON (bounded residual), RCA dropout 0.1
#            WHY 0.01x: the first attempt at 0.1x (RUN A) drove the TRAIN root CE to 0.019 by
#            step 5200 (baseline ~3.8) -- total memorisation of the 381,000 training positions --
#            while the bounded, trunk-frozen arm (RUN B) stayed on the baseline trajectory.  The
#            memorisation is therefore the TRUNK, and 0.01x is the "reduced LR" the brief asked
#            for.  (Both early runs also had a broken held-out metric -- see stage 3.)
#   stage 2  RUN B2: RCA top4, trunk FROZEN, lr head 1e-3 | RCA 1e-4 (0.1x), out-norm, dropout
#            0.1.  Attribution: can the cross-attention move the number with no trunk adaptation
#            at all?
#
# NOTE the eval-stream bugfix: the first two runs fed the live evaluator the TRAIN windows
# (`_ls = (Pr,Tr,Wr,Sr)`) while pairing them with the VAL targets, so their held-out numbers
# (0.23 % / 0.70 %) were measurement noise, not model collapse.  Fixed to (Pv,Tv,Wv,Sv) and
# proven by verify_live_eval.py: max|H_live - H_cached| = 0.000e+00 on all 300 val windows.
set -u
cd /workspace/hf_v19_2_release || exit 90
OUT=/workspace/root_attn; mkdir -p $OUT
PY=/workspace/venvs/rootformer/bin/python
CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
SM=/workspace/discrete_path/smoke_cache
FIXFLAGS="--feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln"

run_arm () {  # $1 tag  $2 extra flags
  local TAG=$1; shift
  echo "=== ARM $TAG : $* ==="
  local T0=$(date +%s)
  ( M=0; while true; do
      V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      [ -n "${V:-}" ] && [ "$V" -gt "$M" ] 2>/dev/null && M=$V
      echo "$M" > $OUT/vram_${TAG}.txt
      sleep 1
    done ) &
  local SP=$!
  setsid nohup $PY nrmt_train.py --checkpoint $CK --cache $CACHE --head-init remap \
      --steps 20000 --batch-size 32 --eval-every 1000 --tag $TAG \
      --probe-out $OUT/trace_${TAG}.jsonl --out $OUT/results_${TAG}.json \
      --save $OUT/head_${TAG}.pt $FIXFLAGS "$@" > $OUT/train_${TAG}.log 2>&1 < /dev/null &
  local P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 120
    echo "[chain $(date -u +%H:%M:%S)] $TAG :: $(grep -E '^  \[eval|^  step|^  \[proof' $OUT/train_${TAG}.log | tail -1 | cut -c1-190)"
  done
  wait $P; local RC=$?
  kill $SP 2>/dev/null; wait $SP 2>/dev/null
  local T1=$(date +%s); local PEAK=$(cat $OUT/vram_${TAG}.txt 2>/dev/null)
  echo "[chain] $TAG rc=$RC wall_s=$((T1-T0)) peak_vram_mib=$PEAK"
  echo "[chain] $TAG nan/inf steps: $(grep -ciE 'nan|inf' $OUT/trace_${TAG}.jsonl)"
  grep -E 'trainable parameters|TRUNK LIVE|optimizer groups|RCA init' $OUT/train_${TAG}.log
  $PY - "$TAG" <<'PYEOF'
import json, sys
tag = sys.argv[1]
h = json.load(open(f'/workspace/root_attn/results_{tag}.json'))['history'][-1]
print(f'  [{tag}] FINAL step {h["step"]} nan_steps {h.get("nan_steps")}')
for k in ('ALL_val', 'ALL_val_RCA_OFF', 'NOVEL_only', 'NOVEL_only_RCA_OFF'):
    if k in h:
        v = h[k]
        print(f'    {k}: acc@1 {100*v["acc@1"]:.3f}%  acc@5 {100*v["acc@5"]:.3f}%  '
              f'CE_z {v["ce_z"]:.4f}  n={v["n"]}')
print('    gates', [round(x, 5) for x in h.get('rca_gate', [])], 'h_drift', h.get('h_drift'))
PYEOF
}

echo "=== STAGE 0: default-path equivalence on the FINAL code ==="
$PY nrmt_train_ORIGINAL.py --checkpoint $CK --cache $SM --head-init remap --steps 40 \
    --batch-size 8 --eval-every 40 --train-windows 13 --val-windows 13 --seed 7 $FIXFLAGS \
    --tag EQ_ORIG --probe-out $OUT/eq_ORIG.jsonl --out $OUT/eq_ORIG.json \
    --save $OUT/eq_ORIG.pt > $OUT/eq_ORIG.log 2>&1
echo "orig rc=$?"
$PY nrmt_train.py --checkpoint $CK --cache $SM --head-init remap --steps 40 \
    --batch-size 8 --eval-every 40 --train-windows 13 --val-windows 13 --seed 7 $FIXFLAGS \
    --tag EQ_PATCHED --probe-out $OUT/eq_PATCHED.jsonl --out $OUT/eq_PATCHED.json \
    --save $OUT/eq_PATCHED.pt > $OUT/eq_PATCHED.log 2>&1
echo "patched rc=$?"
$PY - <<'PYEOF'
import json
def load(p): return [json.loads(l) for l in open(p)]
a = load('/workspace/root_attn/eq_ORIG.jsonl'); b = load('/workspace/root_attn/eq_PATCHED.jsonl')
keys = ['step','loss','root','wazn','prefix','suffix','grad_norm','lr','p_ss','skipped']
bad = [(x['step'], k) for x, y in zip(a, b) for k in keys if x.get(k) != y.get(k)]
print('EQUIVALENCE rows', len(a), len(b), 'mismatched fields', len(bad),
      '->', 'IDENTICAL' if not bad and len(a) == len(b) == 40 else bad[:5])
PYEOF

run_arm RCA_UNFREEZE_A2 --root-cross-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.01 \
    --rca-lr-scale 0.3 --rca-out-norm --rca-dropout 0.1 --rca-ablate-eval --h-drift-probe 500

run_arm RCA_FROZEN_B2 --root-cross-attn top4 --unfreeze-last 0 \
    --rca-lr-scale 0.1 --rca-out-norm --rca-dropout 0.1 --rca-ablate-eval --h-drift-probe 500

echo CHAIN_FINAL_DONE
