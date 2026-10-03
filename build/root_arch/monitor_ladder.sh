#!/usr/bin/env bash
# monitor_ladder.sh -- read-only timeline.  Records one compact line every interval so the ladder's
# progress is legible without an agent polling at exactly the right second.
# It NEVER signals a process and NEVER launches anything.
set -u
OUT=/workspace/root_arch
LOG="$OUT/queue/timeline.log"
POLL=${POLL:-300}
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }
say "monitor_ladder START pid=$$ poll=${POLL}s"
while true; do
  line=""
  for tag in FLOOR_A_V2 EARLYROOT_C RESIDUAL_R SCOREBIAS_D LATE_X EARLYROOT_C_S2 FLOOR_A; do
    lg="$OUT/arms/log_$tag.txt"
    [ -f "$lg" ] || continue
    st=$(wc -l < "$OUT/arms/trace_$tag.jsonl" 2>/dev/null || echo 0)
    ev=$(grep -a "eval @" "$lg" 2>/dev/null | tail -1 | sed -E 's/.*eval @([0-9]+)\] ALL_val: acc@1 ([0-9.]+)%.*NOVEL_only: acc@1 ([0-9.]+)%.*/\1:\2\/\3/')
    alive=$(ps -eo args= | grep -E -- "--tag[= ]$tag([[:space:]]|$)" | grep -q . && echo A || echo D)
    line="$line $tag=$alive,steps=$st,last=$ev;"
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
  say "vram=${used}MiB |$line"
  sleep "$POLL"
done
