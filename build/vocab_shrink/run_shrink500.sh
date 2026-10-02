#!/bin/bash
# run_shrink500.sh [STEPS] [BATCH] [EVAL_EVERY] [TAG]
#
# The ONE discriminating run: the shipped FIX configuration (same checkpoint, same ALIGNED
# cache, same flags, same 20 000 steps / batch 32 / lr 1e-3 / 3000 train windows) with the ROOT
# CLASSIFIER shrunk from 9490 to 500 classes (top-500 real roots by TRAIN frequency; gold roots
# outside the top-500 are DROPPED from the root loss and from the root metrics).
#
# Nothing else changes: the cache is untouched, the backbone is frozen, the hidden states and
# the morphemic/history/operator conditioning are indexed by ORIGINAL root ids exactly as in the
# full-vocab arm.  Only root_head's output width, its warm-start rows, and the root CE target
# change -- which is precisely the cardinality manipulation.
#
# Usage:  bash run_shrink500.sh                 # the 20k run
#         bash run_shrink500.sh 60 8 20 SMOKE   # cheap end-to-end smoke test
set -u
STEPS=${1:-20000}
BS=${2:-32}
EV=${3:-1000}
TAG=${4:-SHRINK500}

cd /workspace/hf_v19_2_release || exit 90
OUT=/workspace/vocab_shrink; mkdir -p "$OUT"
PY=/workspace/venvs/rootformer/bin/python
export PYTHONPATH=/workspace/hf_v19_2_release
CK=/workspace/hf_v19_2_release/checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
MAP=$OUT/root_map_top500.pt

# the patched copy resolves the blueprint as <its own dir>/data/...; read-only symlink, the
# shipped blueprint is never written.
[ -e "$OUT/data" ] || ln -s /workspace/hf_v19_2_release/data "$OUT/data" || exit 92

if pgrep -f 'nrmt_train_shrink[.]py' >/dev/null; then
  echo "[shrink] ABORT: another shrunk trainer is already running"; exit 91
fi
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -ne 0 ]; then
  echo "[shrink] ABORT: another GPU process is running"; exit 93
fi

echo "[shrink] tag=$TAG steps=$STEPS bs=$BS eval_every=$EV"
echo "[shrink] ckpt=$CK"
echo "[shrink] cache=$CACHE map=$MAP"
echo "[shrink] starts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
T0=$(date +%s)

( M=0; while true; do
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "${V:-}" ] && [ "$V" -gt "$M" ] 2>/dev/null && M=$V
    echo "$M" > "$OUT/vram_${TAG}.txt"
    sleep 1
  done ) &
SP=$!

setsid nohup $PY "$OUT/nrmt_train_shrink.py" --checkpoint "$CK" --cache "$CACHE" \
    --head-init remap --steps "$STEPS" --batch-size "$BS" --eval-every "$EV" --tag "$TAG" \
    --root-classes 500 --root-map "$MAP" --oov drop \
    --probe-out "$OUT/trace_${TAG}.jsonl" --out "$OUT/results_${TAG}.json" \
    --save "$OUT/head_${TAG}.pt" \
    --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln \
    > "$OUT/train_${TAG}.log" 2>&1 < /dev/null &
P=$!
echo "[shrink] REMOTE_PID=$P"

while kill -0 $P 2>/dev/null; do
  sleep 30
  L=$(grep -E "^  \[eval|^  step" "$OUT/train_${TAG}.log" | tail -1)
  echo "[shrink $(date -u +%H:%M:%S)] ${L:0:150}"
done
wait $P; RC=$?
kill $SP 2>/dev/null; wait $SP 2>/dev/null
T1=$(date +%s)
echo "[shrink] rc=$RC wall_s=$((T1-T0)) peak_vram_mib=$(cat "$OUT/vram_${TAG}.txt" 2>/dev/null)"
echo "[shrink] $(grep -E 'trainable parameters|training windows|SHRUNK ROOT VOCAB|kept-root rows|forbidden classes' "$OUT/train_${TAG}.log" | tr '\n' '|')"
echo "[shrink] NaN/Inf in log: $(grep -ciE 'nan|inf' "$OUT/train_${TAG}.log") hits"
tail -4 "$OUT/train_${TAG}.log"
echo "[shrink] RESULTS:"
$PY -c "
import json
d=json.load(open('$OUT/results_${TAG}.json'))
h=d['history'][-1]; s=d.get('shrunk',{})
print(' final step',h['step'],'| shrunk meta:',s)
for k in ('ALL_val','NOVEL_only'):
    v=h[k]
    print(f\"  {k}: acc@1 {100*v['acc@1']:.3f}% acc@5 {100*v['acc@5']:.3f}% CE_z {v['ce_z']:.4f} (n={v['n']})\")
m=s.get('marginal_shrunk_eval')
if m:
    print(f'  SHRUNK MARGINAL {100*m:.3f}% -> LIFT {h[\"ALL_val\"][\"acc@1\"]/m:.3f}x')
"
echo SHRINK_DONE
