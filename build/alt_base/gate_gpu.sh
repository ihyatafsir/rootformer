#!/usr/bin/env bash
# gate_gpu.sh -- wait until the card has enough FREE VRAM, then run exactly one GPU job.
# Counts every training/eval process by full argv (the standard pgrep pattern misses
# nrmt_train_width.py / nrmt_train_ewc.py / alt_*.py and under-counts by 1-2), AND requires
# real free VRAM, because process count alone does not bound memory.
set -u
NEED_FREE_MIB=${NEED_FREE_MIB:-6000}
ARMS_INCL=${ARMS_INCL:-1}      # how many OTHER resident arms are tolerated
POLL=${POLL:-45}
MAXWAIT=${MAXWAIT:-9000}
LOG=${LOG:-/workspace/alt_base/logs/gate_gpu.log}
TOTAL_MIB=${TOTAL_MIB:-32623}

count_arms() {
  ps -eo pid,args 2>/dev/null | grep -v grep \
    | grep -E 'nrmt_train[a-z_]*\.py|alt_train\.py|alt_probe\.py|verify_alignment\.py|ewc_head_probe\.py' \
    | grep -v 'gate_gpu.sh' | wc -l
}
free_mib() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo $(( TOTAL_MIB - ${used:-TOTAL_MIB} ))
}
t0=$(date +%s)
while true; do
  arms=$(count_arms); free=$(free_mib)
  echo "[$(date -u +%H:%M:%S)] arms=$arms free=${free}MiB (need arms<=$ARMS_INCL free>=$NEED_FREE_MIB)" >> "$LOG"
  if [ "$arms" -le "$ARMS_INCL" ] && [ "$free" -ge "$NEED_FREE_MIB" ]; then
    echo "[$(date -u +%H:%M:%S)] SLOT OK -> $*" >> "$LOG"
    exec "$@"
  fi
  [ $(( $(date +%s) - t0 )) -ge "$MAXWAIT" ] && { echo "gate timeout" >> "$LOG"; exit 3; }
  sleep "$POLL"
done
