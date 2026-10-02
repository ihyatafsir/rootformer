#!/bin/bash
# chain_depth_t12.sh -- RETRY wrapper around the top12 depth arm.
#
# WHY: attempt 1 (18:58:33 -> 19:08:34 UTC) was killed by an EXTERNAL SIGTERM after 601 s
# (rc=143, no CUDA-OOM traceback, no dmesg OOM entry) at the exact moment another agent
# started a new GPU arm -- the GPU then held two OTHER arms.  Nothing in this experiment
# signals other processes.  Because the trainer has no resume, an interrupted arm restarts
# from step 0, so this wrapper re-gates and retries.
#
# Each attempt goes through run_depth.sh's durable gate: at most 1 other nrmt_train arm and
# >= 12 GiB free VRAM, verified twice.  Three concurrent 32-batch arms is never attempted.
set -u
OUT=/workspace/root_attn_depth
exec >> $OUT/chain_t12.out 2>&1
echo "=== chain_depth_t12 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
for A in 1 2 3 4 5; do
  echo "=== ATTEMPT $A start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  if [ -f $OUT/logs/train_RCA_DEPTH_T12.log ]; then
    P=$((A-1))
    cp -f $OUT/logs/train_RCA_DEPTH_T12.log $OUT/logs/train_RCA_DEPTH_T12_attempt${P}_killed.log 2>/dev/null
    cp -f $OUT/trace_RCA_DEPTH_T12.jsonl $OUT/trace_RCA_DEPTH_T12_attempt${P}_killed.jsonl 2>/dev/null
    rm -f $OUT/logs/train_RCA_DEPTH_T12.log $OUT/trace_RCA_DEPTH_T12.jsonl
  fi
  OUTDIR=$OUT TAG=RCA_DEPTH_T12 RCA=top12 STEPS=20000 EVAL_EVERY=1000 bash $OUT/run_depth.sh
  RC=$?
  echo "=== ATTEMPT $A exit=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  if [ -f $OUT/results_RCA_DEPTH_T12.json ]; then
    echo "COMPLETED on attempt $A"
    break
  fi
  echo "[retry] no results json -- the arm was interrupted (a clean finish always writes it); re-gating in 60 s"
  sleep 60
done
echo "CHAIN_T12_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
