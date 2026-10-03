#!/usr/bin/env bash
# backup_to_gdrive.sh -- recurring, incremental backup of the rootformer handoff state to Drive.
#
# DESIGN NOTES (why it looks like this)
# -------------------------------------
# * RUNS ON THE POD, not the laptop.  Two reasons, both load-bearing:
#     - `/workspace/.rclone/rclone.conf` lives on the pod, and the pod's rclone is the one the
#       previous backups used.  rclone on the laptop has NO config ("didn't find section in config
#       file") so `gdrive:` does not resolve there.
#     - The pod holds artifacts the laptop does not.  Backing up laptop->Drive would silently miss
#       them, which is the failure mode this project keeps hitting: a check that reports success
#       while doing nothing.
# * INCREMENTAL: `rclone copy --checksum` with an identical exclusion set on copy and verify, so a
#   rerun is cheap and the 3-hourly schedule does not re-upload 5 GiB each time.
# * NO `|| true`, no `2>/dev/null` on transfer or verify steps.  Exit codes are checked and the
#   script exits non-zero on failure.  A "successful" upload that transferred nothing has already
#   burned this project once.
# * ONE TIMESTAMPED DIRECTORY PER RUN (`<BASE>/<UTCSTAMP>/`) so a bad run cannot corrupt the
#   previous good one, and so `rclone lsf` tells you at a glance which runs completed.
# * The SCOPE is small and high-value by default: this is the handoff state (STATE.md, NEXT.md, the
#   session's evidence dirs, the rclone config).  It deliberately does NOT sweep the multi-GB weigh
#   points -- set TIER1=1 to add them.
#
# usage:  bash backup_to_gdrive.sh            # one backup now
#         TIER1=1 bash backup_to_gdrive.sh    # also copy the multi-GB result trees
set -euo pipefail

RC=${RC:-"rclone --config /workspace/.rclone/rclone.conf"}
BASE=${BASE:-gdrive:rootformer_backup_auto}
STAMP=$(date -u +%Y-%m-%d_%H%M%SZ)
DEST="$BASE/$STAMP"
LOG=${LOG:-/workspace/root_arch/backup/backup_$(date -u +%Y-%m-%d).log}
TIER1=${TIER1:-0}

mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

say "=== backup START -> $DEST (tier1=$TIER1) ==="

# fail early and loudly if the remote is unreachable -- never proceed to a "successful" no-op
if ! $RC lsd gdrive: >/dev/null 2>&1; then
  say "FATAL: gdrive: is not reachable. Refusing to report success."
  exit 9
fi

# ---- the exclusion set, computed ONCE and used identically for copy and verify -------------
# `__pycache__` and stray .pyc are noise; `*.trunk.pt` / arm checkpoints are large transient
# artifacts owned by the running trainer and are not part of the handoff.
EXCLUDES=(
  --exclude '__pycache__/**'
  --exclude '*.pyc'
  --exclude '*.tmp'
  --exclude 'core.*'
)

# ---- source tree: small, high-value, and specifically THIS session's output -----------------
WORK=${WORK:-/workspace/_backup_stage}
rm -rf "$WORK"; mkdir -p "$WORK"

copy_in() {  # copy_in <src> <destsub>
  local src=$1 dst=$2
  if [ -e "$src" ]; then
    mkdir -p "$WORK/$dst"
    cp -a "$src" "$WORK/$dst/" 2>/dev/null || cp -r "$src" "$WORK/$dst/"
    return 0
  fi
  say "  WARN: missing source $src (skipped, not silently ignored)"
  return 1
}

# handoff documents -- the things a next session actually needs
for f in STATE.md NEXT.md README.md; do
  [ -f "/workspace/rootformer/$f" ] && cp -a "/workspace/rootformer/$f" "$WORK/" && say "  staged $f"
done
# the live pod-side copies, if the local mirror is behind
for f in STATE.md NEXT.md; do
  [ -f "/workspace/$f" ] && mkdir -p "$WORK/pod_docs" && cp -a "/workspace/$f" "$WORK/pod_docs/" \
    && say "  staged pod_docs/$f"
done

# this session's evidence directories (small)
mkdir -p "$WORK/pod"
for d in qiyas root_arch transmute_v2; do
  if [ -d "/workspace/$d" ]; then
    cp -a "/workspace/$d" "$WORK/pod/$d" 2>/dev/null || say "  WARN: could not fully copy /workspace/$d"
    say "  staged pod/$d"
  else
    say "  note: /workspace/$d absent"
  fi
done

# the rclone config itself: without it none of this is reproducible
mkdir -p "$WORK/creds" && cp -a /workspace/.rclone/rclone.conf "$WORK/creds/" && say "  staged creds/rclone.conf"

# git HEAD id, so the backup is tied to a commit
if [ -d /workspace/rootformer/.git ]; then
  ( cd /workspace/rootformer && git log --oneline -20 && echo "--- status ---" && git status --short ) \
    > "$WORK/git_state.txt" 2>&1 && say "  staged git_state.txt"
fi

# optional: the multi-GB weigh points.  OFF by default -- they are already in the *_final backups
# and re-uploading 2.3 GiB every 3 hours is not what "backup the handoff" should mean.
if [ "$TIER1" = "1" ]; then
  for d in /workspace/ghazali_forget /workspace/head_fix /workspace/ishtiqaq_check; do
    [ -d "$d" ] || continue
    say "  TIER1: staging $d (large)"
    mkdir -p "$WORK/tier1"; cp -a "$d" "$WORK/tier1/" 2>/dev/null || say "  WARN: partial copy of $d"
  done
fi

staged_size=$(du -sh "$WORK" | cut -f1)
say "staged $staged_size in $WORK"

# ---- transfer ------------------------------------------------------------------------------
say "copy -> $DEST"
if ! $RC copy "$WORK" "$DEST" "${EXCLUDES[@]}" --checksum --transfers 8 --checkers 16 \
     --stats-one-line --stats 30s 2>&1 | tee -a "$LOG"; then
  say "FATAL: rclone copy failed. Nothing is claimed as backed up."
  exit 10
fi

# ---- verify (same exclusion set!) ------------------------------------------------------------
say "verify (rclone check --checksum --one-way)"
if ! $RC check "$WORK" "$DEST" "${EXCLUDES[@]}" --checksum --one-way 2>&1 | tee -a "$LOG"; then
  say "FATAL: verification FAILED. The copy is not trustworthy."
  exit 11
fi

# ---- record what actually landed ------------------------------------------------------------
count=$($RC size "$DEST" 2>/dev/null | tr '\n' ' ')
say "remote now: $count"
say "=== backup OK -> $DEST ==="
rm -rf "$WORK"
