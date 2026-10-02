#!/bin/bash
# gate_q20k.sh -- run arm Q (F_head kappa=10) for the full 20,000 steps, alone, to reach the
# brief's >=19.5% endpoint that the 6,000-step re-scope could not.  KILLS NOTHING.
set -u
cd /workspace/hf_v19_2_release
PY=/workspace/venvs/rootformer/bin/python
O=/workspace/ghazali_forget
CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
FLAGS="--feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln"
LOG=$O/gate_q20k.log
log(){ echo "[q20k $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
foreign_py(){ local n=0 p exe; for p in $(pgrep -f 'nrmt_train' 2>/dev/null); do exe=$(readlink /proc/$p/exe 2>/dev/null || echo none); case "$exe" in *python*) n=$((n+1));; esac; done; echo "$n"; }
used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
LAM=$($PY -c "import json;print(repr(10*json.load(open('$O/compare_report.json'))['lambda']['head']['lambda_kappa_1']))")
log "START q20k. lambda F_head kappa=10 -> $LAM"
DL=$(( $(date +%s) + 21600 ))
while :; do
  U=$(used); F=$(foreign_py)
  if [ -n "${U:-}" ] && [ "$F" -le 1 ] && [ $(( U + 11800 )) -le 30500 ]; then log "slot: used=${U} foreign=${F}"; break; fi
  [ "$(date +%s)" -gt "$DL" ] && { log "TIMEOUT (used=$U foreign=$F)"; exit 90; }
  sleep 20
done
TAG=EWC_Q20K_LHEAD
( M=0; while true; do V=$(used); [ -n "${V:-}" ] && [ "$V" -gt "$M" ] 2>/dev/null && M=$V; echo "$M" > $O/vram_$TAG.txt; sleep 2; done ) &
setsid nohup $PY $O/nrmt_train_ewc.py --checkpoint $CK --cache $CACHE --head-init remap \
  --steps 20000 --batch-size 32 --eval-every 1000 --tag $TAG \
  --probe-out $O/trace_$TAG.jsonl --out $O/results_$TAG.json --save $O/head_$TAG.pt \
  $FLAGS --root-cross-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.1 --rca-lr-scale 0.3 \
  --rca-out-norm --rca-dropout 0.1 --rca-ablate-eval --h-drift-probe 500 \
  --ewc-lambda "$LAM" --ewc-fisher $O/fisher_head.pt > $O/train_$TAG.log 2>&1 < /dev/null &
P=$!; echo $P > $O/pid_$TAG.txt; log "$TAG REMOTE_PID=$P"
while [ -d /proc/$P ]; do sleep 120; log "$(grep -E '^  \[eval' $O/train_$TAG.log | tail -1 | cut -c1-110)"; done
log "$TAG finished; peak VRAM $(cat $O/vram_$TAG.txt)MiB"
log "Q20K_DONE"
