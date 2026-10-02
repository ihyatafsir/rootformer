#!/bin/bash
# probe_run.sh -- run the (read-only) attention-dynamics probe on a list of checkpoints, in ONE
# GPU process at a time, so the comparison is between matched steps.
#
# The probe rebuilds the RCA module at the width recorded in the checkpoint metadata (`rca_dim`),
# so it works unchanged for W=896 and W=1792 arms.
#
# Usage:  bash probe_run.sh OUTDIR  TAG=ckptpath [TAG=ckptpath ...]
set -u
OUT=${1:?OUTDIR}; shift
PY=/workspace/venvs/rootformer/bin/python
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
mkdir -p $OUT/probe
for spec in "$@"; do
  TAG=${spec%%=*}; CK=${spec#*=}
  if [ ! -f "$CK" ]; then echo "[probe] SKIP $TAG: $CK missing"; continue; fi
  echo "[probe] $TAG <- $CK  $(date -u +%H:%M:%S)"
  $PY /workspace/root_attn_width/probe_width.py --ckpt "$CK" --tag "$TAG" --cache $CACHE \
      --windows 8 --out $OUT/probe/probe_${TAG}.json > $OUT/probe/probe_${TAG}.log 2>&1
  RC=$?
  echo "[probe] $TAG rc=$RC"
  grep -E "metadata|gates:|max\|h\(live\)|RCA load|entropy_norm_mean|eff_positions_mean|delta" \
      $OUT/probe/probe_${TAG}.log | tail -12
  [ "$RC" != "0" ] && { echo "[probe] FAILED $TAG:"; tail -20 $OUT/probe/probe_${TAG}.log; }
done
echo "PROBE_RUN_DONE"
