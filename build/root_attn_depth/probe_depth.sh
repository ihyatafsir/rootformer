#!/bin/bash
# probe_depth.sh -- attention-entropy / degeneracy probe on a saved depth-arm RCA payload.
# Uses the EXISTING READ-ONLY probe (/workspace/root_attn_ctl/probe_rca_dynamics.py); nothing
# under /workspace/root_attn or /workspace/root_attn_ctl is written.
#   usage: bash probe_depth.sh <ckpt.pt.trunk.pt> <tag> <comma-separated layers>
set -u
CKPT=$1; TAG=$2; LAYERS=$3
PY=/workspace/venvs/rootformer/bin/python
OUT=/workspace/root_attn_depth
cd /workspace/hf_v19_2_release || exit 90
$PY /workspace/root_attn_ctl/probe_rca_dynamics.py \
    --ckpt "$CKPT" --tag "$TAG" --layers "$LAYERS" --windows 8 \
    --out "$OUT/probe_${TAG}.json" 2>&1 | tail -40
