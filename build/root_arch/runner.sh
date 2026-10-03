#!/usr/bin/env bash
# runner.sh -- the queue.  It runs ON THE POD under `setsid`, so an SSH drop cannot kill it and
# no human decision gates GPU use.
#
# WHY THIS EXISTS
# ---------------
# Every idle hour tonight came from a human or an agent deciding WHEN to launch, while the card was
# rented.  This script makes the decision mechanical: as soon as an occupant slot is free, the next
# STAGED arm starts.  When A finishes, C starts without waiting for anyone.
#
# THE STAGES (Phase 1)
# --------------------
#   FLOOR_A       trunk TRAINED at normal LR, NO root pathway                  <- mandatory floor
#   EARLYROOT_C   A + root pathway on ALL 24 layers, BOTH mechanisms:
#                   residual injector  (57.86 M)  --root-cross-attn all
#                   native score bias  (11.01 M)  --ishtiqaq-root-bias all --ishtiqaq-root-source shared
#   RESIDUAL_R    A + the residual injector ONLY -- attributes the effect between mechanisms
#
# THE ONLY DIFFERENCE between FLOOR_A and EARLYROOT_C is the root pathway.  Checkpoint, cache,
# LR, steps, batch, head, feature branch, eval cadence and seed are identical, so a difference in
# the eval is attributable to the pathway and not to the recipe.
#
# OCCUPANCY RULE (the one every gate on this pod has got wrong)
# ------------------------------------------------------------
# `pgrep -f "nrmt_train[.]py --checkpoint"` does NOT match `nrmt_train_width.py`, `nrmt_train_ewc.py`
# or `nrmt_train_v13_sdpa.py`.  This counts with a BROADENED `ps` AND checks VRAM.  A third
# 32-batch arm OOMs a RUNNING arm, which is strictly worse than waiting, so it waits.  It NEVER
# kills anything: EWC_Q20K_LHEAD and RCA_NATIVE_A2 are other agents' work.
set -u

PY=${PY:-/workspace/venvs/rootformer/bin/python}
ROOT=${ROOT:-/workspace/root_arch}
TRAIN=${TRAIN:-$ROOT/nrmt_train_v13_sdpa.py}
OUTDIR=${OUTDIR:-$ROOT/arms}
QUEUE=${QUEUE:-$ROOT/queue}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-32}
LR=${LR:-1e-3}
TRUNK_LR_SCALE=${TRUNK_LR_SCALE:-1.0}
CKPT=${CKPT:-/workspace/hf_v19_2_release/checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors}
CACHE=${CACHE:-/workspace/head_fix/nrmp_cache_9490_aligned}
TOTAL_VRAM_MIB=${TOTAL_VRAM_MIB:-32623}
# MEASURED, not estimated.  One `--unfreeze-trunk-all` 32-batch arm peaks at 17.69 GiB (the CUDA
# OOM message from the two-arm attempt reports it exactly), because ALL 24 layers retain
# activations.  The card is 31.37 GiB, so TWO such arms CANNOT coexist: the second one OOM'd with
# 14.25 MiB free while the first held 13.65 GiB.
#   * The "two 32-batch arms ~21 GB" ceiling in the brief was measured on `--unfreeze-last 4` arms
#     (10.6 GiB each).  It does NOT transfer to full-trunk training.
#   * A third 32-batch arm is impossible either way, which is the rule that actually matters.
# So the gate demands 20 GiB free: a second arm starts only when the first has EXITED, which makes
# this queue SEQUENTIAL.  That is the outcome the brief calls for over an OOM ("strictly worse than
# running sequentially").
PER_ARM_MIB=${PER_ARM_MIB:-20000}
# LARGE CHECKPOINTS GO TO /tmp, NEVER /workspace.  `/workspace` is an rclone mount with a PER-USER
# QUOTA: `df` shows 466 T free while `dd` of 700 MB fails with "Disk quota exceeded".  A 640 MiB
# trunk checkpoint (24 unfrozen layers) silently KILLED a smoke arm this way, with no traceback.
# /tmp is a 50 GB overlay with no quota.  Only small JSON/jsonl/log artefacts stay on /workspace.
CKPTDIR=${CKPTDIR:-/tmp/root_arch_arms}
MAX_OCCUPANTS=${MAX_OCCUPANTS:-2}      # the measured-safe ceiling.  NEVER a third.
POLL=${POLL:-45}
STAGES=${STAGES:-"FLOOR_A EARLYROOT_C RESIDUAL_R SCOREBIAS_D LATE_X FLOOR_A_S2 EARLYROOT_C_S2"}

mkdir -p "$OUTDIR" "$QUEUE" "$QUEUE/state" "$CKPTDIR"
LOG="$QUEUE/runner.log"
PIDFILE="$QUEUE/runner.pid"
STATE="$QUEUE/state"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# --------------------------------------------------------------------------- occupancy
# BROADENED counter.  The first grep anchors on the interpreter path at the START of the args, so
# the pattern text inside THIS script (or in grep's own argv) can never match itself.
# NOTE: `grep -c` PRINTS 0 and EXITS 1 when there are no matches, so it must NOT be followed by
# `|| echo 0` -- that yields "0\n0" and every numeric comparison on it breaks.
occupants() {
  local n
  n=$(ps -eo pid=,args= 2>/dev/null \
      | grep -E '^ *[0-9]+ +/workspace/venvs/rootformer/bin/python' \
      | grep -cE 'nrmt_train|alt_train|train_rootformer' 2>/dev/null)
  echo "${n:-0}"
}
occupant_detail() {
  ps -eo pid=,etime=,args= 2>/dev/null \
    | grep -E '^ *[0-9]+ +[0-9:]+ +/workspace/venvs/rootformer/bin/python' \
    | grep -E 'nrmt_train|alt_train|train_rootformer' \
    | sed -E 's/^ *([0-9]+) +([0-9:]+) +.*--tag ([^ ]+).*/  pid=\1 etime=\2 tag=\3/'
}
free_vram() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r'
}
used_vram() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r'
}
# Per-PROCESS VRAM, timestamped, appended every poll: this is how "peak VRAM per arm" is obtained
# without instrumenting the trainer.  Joined to a tag through $QUEUE/pid_<tag>.
VRAM_CSV="$QUEUE/vram.csv"
vram_snapshot() {
  local ts; ts=$(date -u +%s)
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' \r' | awk -v t="$ts" -F, 'NF==2 {print t","$1","$2}' >> "$VRAM_CSV"
}

# The pid that actually holds GPU memory is the PYTHON process, not the `setsid` wrapper that
# `$!` captured.  Resolve it from the unique `--tag`.
arm_pid() {
  ps -eo pid=,args= 2>/dev/null \
    | grep -E '^ *[0-9]+ +/workspace/venvs/rootformer/bin/python' \
    | grep -E -- "--tag[= ]$1([[:space:]]|$)" | awk '{print $1}' | head -1
}

# ps-based liveness on the unique `--tag` the trainer echoed into its argv.  No PID reuse hazard.
arm_alive() {
  ps -eo args= 2>/dev/null | grep -E '^/workspace/venvs/rootformer/bin/python' \
    | grep -qE -- "--tag[= ]$1([[:space:]]|$)"
}

# --------------------------------------------------------------------------- stages
common_flags() {
  echo -n "--model-v13 --flash-hooks off --sdpa --pillar-freeze"
  echo -n " --checkpoint $CKPT --cache $CACHE"
  echo -n " --head-init remap --unfreeze-trunk-all --trunk-lr-scale $TRUNK_LR_SCALE"
  echo -n " --lr $LR --steps $STEPS --batch-size $BATCH --eval-every 1000"
  echo -n " --feat-gate --feat-gate-proj-std 1e-3 --margin-ramp 10 --grad-warmup 10"
  echo -n " --logit-scale ln --grad-clip 1.0 --hist 3 --dropout 0.1"
  echo -n " --rca-out-norm --rca-dropout 0.1 --rca-heads 8 --h-drift-probe 500"
}

# THE LADDER.  Every arm shares `common_flags` verbatim -- the SAME checkpoint, cache, LR,
# trunk_lr_scale, steps, batch, head, feature branch, eval cadence and SDPA/pillar settings -- so
# the ONLY variable is which root mechanism is attached, and where.
#   FLOOR_A      no root pathway at all                              (the mandatory floor)
#   EARLYROOT_C  residual injector + native score bias, ALL 24 layers
#   RESIDUAL_R   residual injector alone, all 24 layers              (mechanism isolation)
#   SCOREBIAS_D  native score bias alone, all 24 layers              (mechanism isolation)
#   LATE_X       both mechanisms, but only on the LATE layers 20-23  (early vs late, direct)
# `--*-ablate-eval` is passed ONLY where the corresponding mechanism is actually attached;
# ablating an unattached mechanism is meaningless and the evaluator has no modules to zero.
stage_flags() {
  local C_BOTH="--ishtiqaq-root-source shared --ishtiqaq-gate-init 0.0 \
--ishtiqaq-pillar-gate-init 0.0 --ishtiqaq-lr-scale 1.0"
  case "$1" in
    FLOOR_A)
      echo "--root-cross-attn none --ishtiqaq-root-bias none" ;;
    # RE-RUN of the floor arm with the CURRENT implementation.  FLOOR_A started at 00:04:18Z, before
    # `_sdpa` switched from `enable_gqa=True` to explicit `repeat_interleave`; measured, those two
    # formulations differ by max|dh| = 2.07e-01 on the full trunk (test_gqa_identity.py), the same
    # magnitude as the eager-vs-SDPA difference.  Arm A would therefore differ from arm C in the
    # ATTENTION IMPLEMENTATION as well as in the root pathway.  This re-run removes the confound:
    # same trainer md5, hence the same attention code, as every other arm in the ladder.
    FLOOR_A_V2)
      echo "--root-cross-attn none --ishtiqaq-root-bias none" ;;
    EARLYROOT_C)
      echo "--root-cross-attn all --rca-lr-scale 1.0 --rca-ablate-eval \
--ishtiqaq-root-bias all $C_BOTH --ishtiqaq-ablate-eval" ;;
    RESIDUAL_R)
      echo "--root-cross-attn all --rca-lr-scale 1.0 --rca-ablate-eval \
--ishtiqaq-root-bias none" ;;
    SCOREBIAS_D)
      echo "--root-cross-attn none --ishtiqaq-root-bias all $C_BOTH --ishtiqaq-ablate-eval" ;;
    LATE_X)
      echo "--root-cross-attn top4 --rca-lr-scale 1.0 --rca-ablate-eval \
--ishtiqaq-root-bias top4 $C_BOTH --ishtiqaq-ablate-eval" ;;
    # SECOND SEED.  The A-vs-C conclusion is the whole point of the session, so it must not rest on
    # a single unseeded run: `--seed 1` reseeds torch/numpy/cuda and therefore the batch order and
    # the dropout stream.  Identical in every other respect to FLOOR_A / EARLYROOT_C.
    FLOOR_A_S2)
      echo "--root-cross-attn none --ishtiqaq-root-bias none --seed 1" ;;
    LATE_X_S2)
      echo "--root-cross-attn top4 --rca-lr-scale 1.0 --rca-ablate-eval \
--ishtiqaq-root-bias top4 $C_BOTH --ishtiqaq-ablate-eval --seed 1" ;;
    # SCHEDULE-LENGTH PROBE.  Every arm in the ladder ran the brief's 20 000 steps and EARLYROOT_C
    # was still creeping upward at the end (18.99 %) while its train loss was already 0.0000.  This
    # asks whether 20 k is BINDING: same flags as RESIDUAL_R (the cheapest arm that reaches ~19 %)
    # but 30 000 steps.  `--steps` is appended AFTER common_flags, and argparse takes the last
    # occurrence, so this overrides the common 20 000.
    RESIDUAL_R_30K)
      echo "--root-cross-attn all --rca-lr-scale 1.0 --rca-ablate-eval \
--ishtiqaq-root-bias none --steps 30000" ;;
    EARLYROOT_C_S2)
      echo "--root-cross-attn all --rca-lr-scale 1.0 --rca-ablate-eval \
--ishtiqaq-root-bias all $C_BOTH --ishtiqaq-ablate-eval --seed 1" ;;
    *) echo "UNKNOWN_STAGE" ;;
  esac
}

launch() {
  local tag=$1
  local flags; flags="$(stage_flags "$tag")"
  if [ "$flags" = "UNKNOWN_STAGE" ]; then log "FATAL: unknown stage $tag"; return 1; fi
  local logf="$OUTDIR/log_$tag.txt"
  log "LAUNCH $tag  -> $logf"
  # setsid + nohup + all three std streams redirected: the arm survives an SSH drop AND survives
  # this runner exiting.  APPEND-ONLY: `>>`.
  ( cd "$ROOT" && PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
      setsid nohup "$PY" -u "$TRAIN" $(common_flags) $flags \
      --tag "$tag" \
      --out "$OUTDIR/results_$tag.json" \
      --probe-out "$OUTDIR/trace_$tag.jsonl" \
      --save "$CKPTDIR/head_$tag.pt" \
      < /dev/null >> "$logf" 2>&1 & echo $! > "$QUEUE/pid_$tag" )
  sleep 3
  if arm_alive "$tag"; then
    local ppid_; ppid_=$(arm_pid "$tag")
    echo "${ppid_:-?}" > "$QUEUE/pid_$tag"          # the pid holding VRAM, not the setsid wrapper
    log "  $tag confirmed ALIVE (python pid ${ppid_:-?})"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$STATE/$tag.started"
    wait_for_alloc "$tag"
    return 0
  fi
  log "  FATAL: $tag is NOT alive 3s after launch -- see $logf"
  tail -20 "$logf" >> "$LOG" 2>/dev/null
  return 1
}

# Trainer processes that are ALIVE but NOT yet in `nvidia-smi --query-compute-apps` -- i.e. still
# building the model and holding no memory.  `free` alone cannot see them, which is how a second arm
# got admitted 49 s after the first at 01:28 and OOM'd it.  The gate refuses to launch while this is
# non-zero, which covers the COLD-START case too (another arm already initialising when this runner
# starts, as EARLYROOT_C was).
starting_arms() {
  local pid n=0 listed
  listed=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \r')
  for pid in $(ps -eo pid=,args= 2>/dev/null \
               | grep -E '^ *[0-9]+ +/workspace/venvs/rootformer/bin/python' \
               | grep -E 'nrmt_train|alt_train|train_rootformer' | awk '{print $1}'); do
    if ! echo "$listed" | grep -qx "$pid"; then n=$((n + 1)); fi
  done
  echo "$n"
}

# A NEWLY LAUNCHED ARM HOLDS NO GPU MEMORY FOR ~60 s (model build + checkpoint load).  The VRAM
# gate therefore reads "32 GB free" and would admit a second arm, which then OOMs the first.  This
# is exactly what happened at 01:28:05 -> 01:28:54.  So after every launch, wait until the arm is
# actually resident in `nvidia-smi --query-compute-apps` before the loop may consider another slot.
wait_for_alloc() {
  local tag=$1 t=0 pid
  while [ "$t" -lt 420 ]; do
    if ! arm_alive "$tag"; then
      log "  $tag died before allocating GPU memory -- see $OUTDIR/log_$tag.txt"
      return 1
    fi
    pid=$(arm_pid "$tag")
    if [ -n "$pid" ] && nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
         | tr -d ' \r' | grep -qx "$pid"; then
      log "  $tag resident on the GPU (python pid $pid) after ${t}s -- slot accounting is accurate"
      return 0
    fi
    sleep 10; t=$((t + 10))
  done
  log "  WARN $tag did not appear in compute-apps within 420s; forcing a cooldown instead"
  return 1
}

# --------------------------------------------------------------------------- preflight
preflight() {
  log "PREFLIGHT"
  local ok=1
  [ -x "$PY" ] && log "  interpreter OK: $PY" || { log "  FAIL interpreter $PY"; ok=0; }
  [ -f "$TRAIN" ] && log "  trainer OK: $TRAIN" || { log "  FAIL trainer $TRAIN"; ok=0; }
  [ -f "$CKPT" ] && log "  checkpoint OK: $(stat -c%s "$CKPT") bytes" \
    || { log "  FAIL checkpoint $CKPT"; ok=0; }
  [ -f "$CACHE/train.pt" ] && log "  cache OK: $CACHE" || { log "  FAIL cache $CACHE"; ok=0; }
  grep -q -- '--sdpa' "$TRAIN" && log "  trainer has --sdpa" || { log "  FAIL no --sdpa"; ok=0; }
  grep -q -- '--pillar-freeze' "$TRAIN" && log "  trainer has --pillar-freeze" \
    || { log "  FAIL no --pillar-freeze"; ok=0; }

  # the silent-shrink trap: 9445 instead of 9490 makes every number quietly wrong
  local rc
  rc=$("$PY" - <<'EOF' 2>&1
import sys
for p in ('/workspace/hf_v19_2_release', '/workspace/hf_v19_2_release/models',
          '/workspace/transmute_v2', '/workspace/root_arch', '/workspace/ishtiqaq_check'):
    if p not in sys.path:
        sys.path.insert(0, p)
import nrmp_vocab as nv
V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
v = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
print(f'roots={v.num_roots} awzan={v.num_awzan}')
raise SystemExit(0 if (int(v.num_roots), int(v.num_awzan)) == (9490, 142) else 9)
EOF
)
  local rcst=$?
  log "  vocab preflight: $rc (exit $rcst)"
  [ "$rcst" -eq 0 ] || { log "  FAIL silent vocab shrink -- refusing to launch"; ok=0; }

  # the SDPA install must actually import and wire the score-bias subclass
  local sd
  sd=$("$PY" - <<'EOF' 2>&1
import sys
for p in ('/workspace/hf_v19_2_release', '/workspace/hf_v19_2_release/models',
          '/workspace/transmute_v2', '/workspace/root_arch', '/workspace/ishtiqaq_check'):
    if p not in sys.path:
        sys.path.insert(0, p)
import sdpa_attention as SA
r = SA.install(verbose=False)
print('patched:', len(r['patched']), '| root-bias class:',
      SA.SdpaIshtiqaqRootBiasAttention.__mro__[1].__name__)
raise SystemExit(0 if SA._ORIG_ROOT_BIAS is not None and len(r['patched']) >= 1 else 9)
EOF
)
  local sdst=$?
  log "  sdpa preflight: $sd (exit $sdst)"
  [ "$sdst" -eq 0 ] || { log "  FAIL sdpa install"; ok=0; }

  # The in-place clamp at root_cross_attn.py:136 killed arm C on its first backward.  The verified
  # module is NOT edited; `rca_compat` wraps the method instead.  So the check is not "is the
  # module fixed" but "is the WORKAROUND installed and working".
  if grep -q "root_ids.clamp_(0, root_weight" /workspace/root_attn/root_cross_attn.py 2>/dev/null; then
    log "  root_cross_attn.py still has the in-place clamp_ (module left UNEDITED, as required)"
    if ! grep -q "rca_compat.install" "$TRAIN"; then
      log "  FAIL the in-place clamp_ is present and rca_compat is NOT wired into the trainer"; ok=0
    elif ! "$PY" - <<'PYCK' >/dev/null 2>&1
import sys
for p in ('/workspace/root_attn', '/workspace/root_arch'):
    if p not in sys.path:
        sys.path.insert(0, p)
import rca_compat
r = rca_compat.install(verbose=False)
raise SystemExit(0 if r.get('installed') or r.get('reason') == 'upstream already fixed' else 9)
PYCK
    then
      log "  FAIL rca_compat.install() did not report a successful wrap"; ok=0
    else
      log "  rca_compat workaround installed and verified -> OK"
    fi
  else
    log "  root_cross_attn.py clamp is already OUT-OF-PLACE upstream -> no workaround needed"
  fi

  local dfs
  dfs=$(df -Pm /workspace | awk 'NR==2{print $4}')
  log "  /workspace df says ${dfs} MiB free (df LIES here: per-user quota)"
  [ "${dfs:-0}" -ge 2000 ] || { log "  FAIL less than 2 GiB free (df)"; ok=0; }

  # THE HONEST CHECK: actually write the checkpoint size we need, to the directory we will use.
  local need_mb=750
  if dd if=/dev/zero of="$CKPTDIR/.quota_probe" bs=1M count=$need_mb 2>/dev/null; then
    log "  checkpoint dir OK: wrote ${need_mb} MiB to $CKPTDIR"
  else
    log "  FAIL cannot write ${need_mb} MiB to $CKPTDIR -- checkpoints would die mid-arm"; ok=0
  fi
  rm -f "$CKPTDIR/.quota_probe"
  # and prove /workspace does NOT have room, so nobody 'helpfully' moves them back
  if dd if=/dev/zero of="$OUTDIR/.quota_probe" bs=1M count=$need_mb 2>/dev/null; then
    log "  NOTE /workspace accepted ${need_mb} MiB (quota headroom exists today)"
  else
    log "  confirmed: /workspace CANNOT take a ${need_mb} MiB checkpoint (quota) -- using $CKPTDIR"
  fi
  rm -f "$OUTDIR/.quota_probe"

  # NOT `[ ... ] && log PASS || log FAIL`: `log` pipes through tee, so a non-zero tee makes the
  # || branch fire and prints a contradictory "PREFLIGHT FAIL" right after "PREFLIGHT PASS".
  if [ "$ok" -eq 1 ]; then log "PREFLIGHT PASS"; else log "PREFLIGHT FAIL"; fi
  return $((1 - ok))
}

# --------------------------------------------------------------------------- main
# Three disjoint lists, and a stage moves unstarted -> running -> finished and can go back to
# unstarted if a launch fails.  An earlier draft dropped a stage from the pending list as soon as
# it started, which meant a RUNNING arm was never polled for completion and the runner declared
# "ALL STAGES RETIRED" while two arms were still training.
main() {
  echo $$ > "$PIDFILE"
  log "=============================================================="
  log "runner.sh START pid=$$ stages=[$STAGES] steps=$STEPS batch=$BATCH lr=$LR"
  log "  trunk_lr_scale=$TRUNK_LR_SCALE  max_occupants=$MAX_OCCUPANTS  per_arm_MiB=$PER_ARM_MIB"
  log "=============================================================="
  preflight || { log "aborting: preflight failed"; exit 1; }

  local unstarted="$STAGES" running="" finished=""
  local last_launch=0 now_s
  while :; do
    now_s=$(date -u +%s)
    local occ free used starting
    occ=$(occupants); free=$(free_vram); used=$(used_vram); starting=$(starting_arms)
    vram_snapshot

    # ---- retire finished arms ---------------------------------------------------
    local still_running=""
    for t in $running; do
      if arm_alive "$t"; then
        still_running="$still_running $t"
      else
        if [ -s "$OUTDIR/results_$t.json" ]; then
          log "DONE $t -- results_$t.json present ($(stat -c%s "$OUTDIR/results_$t.json") bytes)"
        else
          log "DIED $t -- no results_$t.json; see $OUTDIR/log_$t.txt (NOT restarting; not killing)"
        fi
        date -u +%Y-%m-%dT%H:%M:%SZ > "$STATE/$t.finished"
        finished="$finished $t"
      fi
    done
    running="${still_running# }"

    if [ -z "$unstarted" ] && [ -z "$running" ]; then
      log "ALL STAGES FINISHED:${finished:- none}.  runner.sh exiting."
      break
    fi

    # ---- launch the next unstarted stage if a slot is free ----------------------
    if [ -n "$unstarted" ] && [ "$occ" -le $((MAX_OCCUPANTS - 1)) ] \
       && [ "${free:-0}" -ge "$PER_ARM_MIB" ] && [ "${starting:-0}" -eq 0 ] \
       && [ $((now_s - last_launch)) -ge "${LAUNCH_COOLDOWN:-240}" ]; then
      local nxt rest
      nxt=$(echo "$unstarted" | awk '{print $1}')
      rest=$(echo "$unstarted" | cut -d' ' -f2-)
      if [ "$rest" = "$unstarted" ]; then rest=""; fi
      log "SLOT FREE (occupants=$occ <= $((MAX_OCCUPANTS-1)), free=${free}MiB >= ${PER_ARM_MIB}MiB)"
      log "occupied now:"; occupant_detail | tee -a "$LOG"
      last_launch=$(date -u +%s)
      if launch "$nxt"; then
        unstarted="$rest"; running="$running $nxt"; running="${running# }"
      else
        log "launch of $nxt FAILED -- leaving it first in the queue, retrying next poll"
      fi
    else
      log "WAIT occupants=$occ/$MAX_OCCUPANTS used=${used}MiB free=${free}MiB initialising=${starting} started=[$running] unstarted=[$unstarted]"
    fi
    sleep "$POLL"
  done
  log "runner.sh EXIT"
}

status() {
  echo "runner pid=$(cat "$PIDFILE" 2>/dev/null || echo none)"
  echo "occupants=$(occupants)/$MAX_OCCUPANTS used=$(used_vram)MiB free=$(free_vram)MiB"
  echo "occupied now:"; occupant_detail
  echo "state markers:"; ls -1 "$STATE" 2>/dev/null | sed 's/^/  /'
  echo "--- logs (tail) ---"
  for f in "$OUTDIR"/log_*.txt; do
    [ -e "$f" ] || continue
    echo "  == $(basename "$f") ($(wc -l < "$f") lines)"
    tail -2 "$f" | sed 's/^/     /'
  done
  echo "--- results ---"; ls -la "$OUTDIR"/results_*.json 2>/dev/null | sed 's/^/  /'
}

case "${1:-run}" in
  run)     main ;;
  status)  status ;;
  check)   preflight ;;
  # `one TAG` launches exactly ONE arm with the SAME flags the queue would use, then exits.  It
  # exists so a single arm can start the instant a card frees, without a second runner fighting
  # over runner.pid / vram.csv.
  one)
    echo $$ > "$PIDFILE"
    preflight || { log "aborting: preflight failed"; exit 1; }
    [ $# -ge 2 ] || { echo "usage: $0 one TAG"; exit 2; }
    launch "$2"
    ;;
  *)       echo "usage: $0 [run|status|check|one TAG]"; exit 2 ;;
esac
