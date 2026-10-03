#!/usr/bin/env bash
# run_matched_arms.sh -- the floor arm (A) and the early-root arm (C) at MATCHED LR and STEPS.
#
# THE ONLY DIFFERENCE BETWEEN A AND C IS THE ROOT PATHWAY.  Everything else -- checkpoint, cache,
# LR, step count, batch, head, feature branch, eval cadence, seeds -- is identical, so a
# difference in the eval is attributable to the pathway and not to the training recipe.
#
#   A  FLOOR      base + head, NO root pathway                          (root-cross-attn none)
#   C  EARLYROOT  A + the residual injector on ALL 24 layers 0..23      (root-cross-attn all)
#                 = "root structure from the input stage upward"
#   D  C + the native root SCORE BIAS on all 24 layers                  (--ishtiqaq-root-bias all)
#
# OCCUPANCY RULE (the one every gate on this pod has got wrong)
# ------------------------------------------------------------
# The standard guard `pgrep -f "nrmt_train[.]py --checkpoint"` does NOT match `nrmt_train_width.py`,
# `nrmt_train_ewc.py` or `nrmt_train_v13.py` -- it under-counts by 1-2.  This script counts with a
# BROADENED `ps` AND checks VRAM, and it NEVER kills anything.
#
# The measured-safe ceiling is TWO 32-batch arms (~21 GB of 32.6 GB).  So this waits until
# `occupants <= 1` AND `free_vram >= 12000 MiB` -- i.e. until this run is the SECOND occupant at
# most -- and then runs A and C SEQUENTIALLY.  Pass --concurrent only when you have verified both
# are safe; concurrent is what fits a short window, sequential is what cannot go wrong.
set -u

POD=${POD:-root@213.173.104.76}
PORT=${PORT:-46758}
KEY=${KEY:-/home/grem3/Documents/deepseek-harness/default-workspace/rootformer/.secrets/id_ed25519_runpod}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-32}
LR=${LR:-1e-3}
TRUNK_LR_SCALE=${TRUNK_LR_SCALE:-1.0}
CKPT=${CKPT:-/workspace/hf_v19_2_release/checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors}
CACHE=${CACHE:-/workspace/head_fix/nrmp_cache_9490_aligned}
OUTDIR=${OUTDIR:-/workspace/transmute_v2/arms}
PY=${PY:-/workspace/venvs/rootformer/bin/python}
TRAIN=${TRAIN:-/workspace/transmute_v2/nrmt_train_v13.py}
TOTAL_VRAM_MIB=32623
PER_ARM_MIB=12000

SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no "$POD")
run() { "${SSH[@]}" "$@"; }

occupants() {
  # BROADENED: any python whose command line names a trainer, whatever the file is called.
  run 'ps -eo args | grep -E "^/workspace/venvs/rootformer/bin/python" \
       | grep -E "nrmt_train|alt_train|train_rootformer|nrmt_governed" | grep -v grep | wc -l'
}
free_vram() {
  run "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" \
    | head -1 | tr -d ' \r' | awk -v t=$TOTAL_VRAM_MIB '{print t-$1}'
}

wait_for_slot() {
  local need=${1:-1}
  while true; do
    local occ free
    occ=$(occupants); free=$(free_vram)
    echo "[gate] $(date -u +%H:%M:%S) occupants=$occ free_vram=${free}MiB (need occupants+$need <= 2 and free >= $PER_ARM_MIB)"
    if [ "$occ" -le $((2 - need)) ] && [ "$free" -ge $PER_ARM_MIB ]; then
      echo "[gate] SLOT OK -- starting (this run is occupant #$((occ + 1)) of 2)"
      return 0
    fi
    echo "[gate] waiting 60s. NOT killing anything."
    sleep 60
  done
}

common_flags() {
  cat <<EOF
--model-v13 --flash-hooks off
--checkpoint $CKPT --cache $CACHE
--head-init remap --unfreeze-trunk-all --trunk-lr-scale $TRUNK_LR_SCALE
--lr $LR --steps $STEPS --batch-size $BATCH --eval-every 1000
--feat-gate --feat-gate-proj-std 1e-3 --margin-ramp 10 --grad-warmup 10
--logit-scale ln --grad-clip 1.0 --hist 3 --dropout 0.1
--rca-out-norm --rca-dropout 0.1 --rca-heads 8
--h-drift-probe 500
EOF
}

launch_A() {
  run "cd /workspace/transmute_v2 && mkdir -p $OUTDIR && setsid nohup $PY -u $TRAIN \
    $(common_flags) --root-cross-attn none --ishtiqaq-root-bias none \
    --tag FLOOR_A --out $OUTDIR/results_FLOOR_A.json \
    --probe-out $OUTDIR/trace_FLOOR_A.jsonl --save $OUTDIR/head_FLOOR_A.pt \
    < /dev/null > $OUTDIR/log_FLOOR_A.txt 2>&1 & disown; sleep 2; echo launched FLOOR_A"
}

launch_C() {
  run "cd /workspace/transmute_v2 && mkdir -p $OUTDIR && setsid nohup $PY -u $TRAIN \
    $(common_flags) --root-cross-attn all --rca-lr-scale 1.0 --rca-ablate-eval \
    --tag EARLYROOT_C --out $OUTDIR/results_EARLYROOT_C.json \
    --probe-out $OUTDIR/trace_EARLYROOT_C.jsonl --save $OUTDIR/head_EARLYROOT_C.pt \
    < /dev/null > $OUTDIR/log_EARLYROOT_C.txt 2>&1 & disown; sleep 2; echo launched EARLYROOT_C"
}

launch_D() {
  run "cd /workspace/transmute_v2 && mkdir -p $OUTDIR && setsid nohup $PY -u $TRAIN \
    $(common_flags) --root-cross-attn all --rca-lr-scale 1.0 --rca-ablate-eval \
    --ishtiqaq-root-bias all --ishtiqaq-root-source shared --ishtiqaq-gate-init 0.0 \
    --ishtiqaq-pillar-gate-init 0.0 --ishtiqaq-lr-scale 1.0 --ishtiqaq-ablate-eval \
    --tag BIAS_D --out $OUTDIR/results_BIAS_D.json \
    --probe-out $OUTDIR/trace_BIAS_D.jsonl --save $OUTDIR/head_BIAS_D.pt \
    < /dev/null > $OUTDIR/log_BIAS_D.txt 2>&1 & disown; sleep 2; echo launched BIAS_D"
}

case "${1:---sequential}" in
  --concurrent)
    wait_for_slot 2
    launch_A; launch_C
    echo "A and C concurrent. Watch VRAM: two arms must stay under ~21 GB." ;;
  --with-bias)
    wait_for_slot 1; launch_A; wait_done_A
    wait_for_slot 1; launch_C
    wait_for_slot 1; launch_D ;;
  --status)
    echo "occupants=$(occupants) free_vram=$(free_vram)MiB"
    run "ls -la $OUTDIR 2>/dev/null | head -20" ;;
  *)
    wait_for_slot 1
    launch_A
    echo "[*] A launched. Poll with: $0 --status ; when it finishes, run: $0 --arm-c"
    ;;
esac
