#!/bin/bash
# chain_depth.sh -- DEPTH curve: top8 first (the primary single-variable step vs CTL's top4),
# then top12 if the budget allows.  Sequential; each arm re-runs the durable GPU gate in
# run_depth.sh, so a third 32-batch arm is never started.
#
#   stage 1  RCA_DEPTH_T8   --root-cross-attn top8   (layers 16..23, ~19.27 M RCA params)
#   stage 2  RCA_DEPTH_T12  --root-cross-attn top12  (layers 12..23, ~28.91 M RCA params)
#
# NO MODULE WAS EDITED (top8/top12 are native to the shipped parse_layer_spec).
set -u
OUT=/workspace/root_attn_depth
mkdir -p $OUT/logs
exec >> $OUT/chain_depth.out 2>&1
echo "=== chain_depth.sh start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

run_stage () {  # $1 tag  $2 layer spec
  echo "=== STAGE $1 rca=$2 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  OUTDIR=$OUT TAG=$1 RCA=$2 STEPS=20000 EVAL_EVERY=1000 bash $OUT/run_depth.sh
  echo "=== STAGE $1 exit=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

run_stage RCA_DEPTH_T8  top8
run_stage RCA_DEPTH_T12 top12

echo "CHAIN_DEPTH_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
