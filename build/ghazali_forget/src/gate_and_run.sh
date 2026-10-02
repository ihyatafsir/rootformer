#!/bin/bash
# gate_and_run.sh -- durable gate v2.
#
# Waits for a SAFE SLOT, not for an empty card: at most ONE foreign python nrmt_train process may be
# resident, and adding this arm's ~11.5 GB must keep the card under ~30 GB.  That keeps at most two
# 32-batch arms resident at once -- the measured-safe footprint -- and never creates a third.
# KILLS NOTHING.  Launches a 6,000-step arm at Run A's exact config with the pre-registered penalty,
# so the comparison is made at the step (5000) where the collapse was actually measured.
set -u
cd /workspace/hf_v19_2_release
PY=/workspace/venvs/rootformer/bin/python
O=/workspace/ghazali_forget
CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
FIXFLAGS="--feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln"
LOG=$O/gate.log
MAXWAIT=${GATE_MAX_WAIT:-21600}     # 6 h
MY_GB=11800                          # MiB this arm needs
CARD=30500                           # MiB ceiling for the sum
STEPS=${GATE_STEPS:-6000}

log(){ echo "[gate $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
foreign_py(){
  local n=0 p exe
  for p in $(pgrep -f 'nrmt_train' 2>/dev/null); do
    exe=$(readlink /proc/$p/exe 2>/dev/null || echo none)
    case "$exe" in *python*) n=$((n+1));; esac
  done
  echo "$n"
}
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

wait_slot(){ # $1 = label
  local deadline=$(( $(date +%s) + MAXWAIT )) U F
  while :; do
    U=$(gpu_used); F=$(foreign_py)
    if [ -n "${U:-}" ] && [ "$F" -le 1 ] && [ $(( U + MY_GB )) -le "$CARD" ]; then
      log "slot free for $1: used=${U}MiB foreign=${F} (would reach $(( U + MY_GB ))MiB)"; return 0
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
      log "GATE TIMEOUT waiting for $1 (used=${U}MiB foreign=${F}). NOT launching. Report this."
      return 90
    fi
    sleep 20
  done
}

LAM_RET=$($PY -c "import json;print(repr(json.load(open('$O/compare_report.json'))['lambda']['ret']['lambda_kappa_1']))")
LAM_HEAD10=$($PY -c "import json;print(repr(10*json.load(open('$O/compare_report.json'))['lambda']['head']['lambda_kappa_1']))")
log "START gate v2. steps=${STEPS}. lambda F_ret kappa=1 -> ${LAM_RET} ; F_head kappa=10 (AMENDMENT 1) -> ${LAM_HEAD10}"

peak_poller(){ ( M=0
    while [ -d /proc/$2 ]; do
      V=$(gpu_used); [ -n "${V:-}" ] && [ "$V" -gt "$M" ] 2>/dev/null && M=$V
      echo "$M" > "$O/vram_$1.txt"; sleep 2
    done
    echo "$M" > "$O/vram_$1.txt" ) & }

launch(){ local TAG=$1; shift
  log "launching $TAG : $*"
  setsid nohup $PY $O/nrmt_train_ewc.py --checkpoint $CK --cache $CACHE --head-init remap \
      --steps $STEPS --batch-size 32 --eval-every 1000 --tag $TAG \
      --probe-out $O/trace_${TAG}.jsonl --out $O/results_${TAG}.json --save $O/head_${TAG}.pt \
      $FIXFLAGS "$@" > $O/train_${TAG}.log 2>&1 < /dev/null &
  local P=$!; echo "$P" > $O/pid_${TAG}.txt; peak_poller "$TAG" "$P"; log "$TAG REMOTE_PID=$P"
  while [ -d /proc/$P ]; do sleep 60; done
  log "$TAG finished"
}

# ---- ARM P: Run A verbatim + F_ret penalty at the pre-registered kappa=1 ----------------------
if wait_slot "arm P ($STEPS steps)"; then
  launch EWC_A_RUNA${STEPS}_LRET \
      --root-cross-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.1 --rca-lr-scale 1.0 \
      --rca-ablate-eval --h-drift-probe 500 \
      --ewc-lambda "$LAM_RET" --ewc-fisher $O/fisher_retained.pt
else
  log "aborting without running arm P"; exit 90
fi

# ---- ARM Q: A2 stabilised config at 0.1x + F_head kappa=10, same step budget ------------------
if wait_slot "arm Q ($STEPS steps)"; then
  launch EWC_B_A2${STEPS}_LHEAD \
      --root-cross-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.1 --rca-lr-scale 0.3 \
      --rca-out-norm --rca-dropout 0.1 --rca-ablate-eval --h-drift-probe 500 \
      --ewc-lambda "$LAM_HEAD10" --ewc-fisher $O/fisher_head.pt
else
  log "arm Q skipped (no slot)"
fi

for T in EWC_A_RUNA${STEPS}_LRET EWC_B_A2${STEPS}_LHEAD; do
  log "$T peak VRAM $(cat $O/vram_$T.txt 2>/dev/null)MiB | nan/inf $(grep -ciE 'nan|inf' $O/trace_$T.jsonl 2>/dev/null)"
  log "$T trainable: $(grep -m1 'trainable parameters' $O/train_$T.log 2>/dev/null | cut -c1-190)"
  log "$T $(grep -m1 'EWC \[al-tarjih\]' $O/train_$T.log 2>/dev/null)"
  log "$T $(grep -m1 'EWC self-check' $O/train_$T.log 2>/dev/null)"
done
log "GATE_DONE"
