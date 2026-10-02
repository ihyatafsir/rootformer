#!/usr/bin/env bash
# gate_train.sh -- wait for a SAFE GPU slot, then run one alt_base arm.
#
# The pod's other gates use  pgrep -f "nrmt_train[.]py --checkpoint"  which does NOT match
# `nrmt_train_width.py` or `nrmt_train_ewc.py`, so they under-count by 1-2.  This one counts
# every resident training/eval python process by inspecting full argv, and additionally requires
# the GPU's own used-memory to be below a ceiling.  Two concurrent 32-batch arms is the measured
# safe limit, so we wait for <=1 resident arm AND >= 9 GiB free.
set -u
OUTDIR=${OUTDIR:-/workspace/alt_base}
LOG=${LOG:-$OUTDIR/logs/gate_train.log}
NEED_ARMS_MAX=${NEED_ARMS_MAX:-1}
NEED_FREE_MIB=${NEED_FREE_MIB:-9000}
POLL=${POLL:-60}
MAXWAIT=${MAXWAIT:-14400}

count_arms() {
  # any python running one of OUR/their training or eval entry points, excluding this gate
  ps -eo pid,args 2>/dev/null | grep -v grep | grep -E \
    'nrmt_train[a-z_]*\.py|nrmt_train_width\.py|nrmt_train_ewc\.py|alt_train\.py|ewc_head_probe\.py|alt_probe\.py|verify_alignment\.py' \
    | grep -v 'gate_train.sh' | wc -l
}
free_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
}

echo "[$(date -u +%H:%M:%S)] gate start: need_arms<=$NEED_ARMS_MAX free>=${NEED_FREE_MIB}MiB poll=${POLL}s" | tee -a "$LOG"
t0=$(date +%s)
while true; do
  arms=$(count_arms); used=$(free_mib); free=$((32623 - ${used:-32623}))
  echo "[$(date -u +%H:%M:%S)] arms=$arms gpu_used=${used}MiB free=${free}MiB" | tee -a "$LOG"
  if [ "$arms" -le "$NEED_ARMS_MAX" ] && [ "$free" -ge "$NEED_FREE_MIB" ]; then
    echo "[$(date -u +%H:%M:%S)] SLOT FREE -> launching: $*" | tee -a "$LOG"
    exec "$@"
  fi
  now=$(date +%s)
  if [ $((now - t0)) -ge "$MAXWAIT" ]; then
    echo "[$(date -u +%H:%M:%S)] gate timed out after ${MAXWAIT}s; NOT launching" | tee -a "$LOG"
    exit 3
  fi
  sleep "$POLL"
done
