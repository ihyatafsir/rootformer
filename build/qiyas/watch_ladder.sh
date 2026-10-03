#!/usr/bin/env bash
# watch_ladder.sh -- self-reporting watcher for the arm ladder's NEXT transition.
#
# WHY: EARLYROOT_C died from the two-occupant CUDA OOM at launch (17.69 GiB vs a 13.65 GiB
# neighbour) and was only noticed ~5 minutes later.  Waiting for a human or an agent to poll at
# the right second is what the runner's own header calls "every idle hour tonight".
#
# This does NOT touch any process.  It only reads ps / nvidia-smi / logs, and appends a timeline
# to /workspace/root_arch/queue/watch.log so the transition is legible after the fact.
#
# Detects, for each arm that appears:
#   * the step it reached
#   * whether it died (process gone) and, if so, the last CUDA/error line from its log
#   * the VRAM peak seen while it was alive
set -u

OUTDIR=${OUTDIR:-/workspace/root_arch/arms}
LOG=/workspace/root_arch/queue/watch.log
POLL=${POLL:-30}
PY=/workspace/venvs/rootformer/bin/python

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> "$LOG"; }

say "watch_ladder.sh START pid=$$ poll=${POLL}s"

seen=""
done_=""
declare -A peak
declare -A laststep

while true; do
  # broadened occupant census -- the narrow pgrep missed _width/_ewc/_v13 trainers
  line=$(ps -eo pid=,etime=,args= 2>/dev/null \
         | grep -E '^ *[0-9]+ +[0-9:]+ +/workspace/venvs/rootformer/bin/python' \
         | grep -E 'nrmt_train|alt_train|train_rootformer' || true)
  cur=$(echo "$line" | grep -oE -- '--tag [A-Za-z0-9_]+' | awk '{print $2}' | sort -u | tr '\n' ' ')
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
  # occupant count and a hard alarm on >=3 (the rule that matters)
  n=$(echo "$line" | grep -c . )
  if [ "$n" -ge 3 ]; then
    say "ALARM occupants=$n used=${used}MiB -- THREE CUDA OCCUPANTS (rule violated) tags=[$cur]"
  fi

  for t in $cur; do
    # track peak VRAM per tag while alive
    p=$(ps -eo pid=,args= 2>/dev/null | grep -E -- "--tag[= ]$t([[:space:]]|$)" | grep -v grep | awk '{print $1}' | head -1)
    if [ -n "${p:-}" ]; then
      m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
          | tr -d ' \r' | awk -F, -v P="$p" '$1==P{print $2}' | head -1)
      if [ -n "${m:-}" ]; then
        if [ "${peak[$t]:-0}" -lt "$m" ]; then peak[$t]=$m; fi
      fi
      # last step reached, from the trace jsonl
      tr="$OUTDIR/trace_$t.jsonl"
      if [ -f "$tr" ]; then
        s=$(tail -1 "$tr" 2>/dev/null | sed -E 's/.*"step": *([0-9]+).*/\1/')
        [ -n "${s:-}" ] && laststep[$t]=$s
      fi
    fi
  done

  # announce arms newly seen
  for t in $cur; do
    case " $seen " in *" $t "*) ;; *)
      seen="$seen $t"
      say "APPEARED tag=$t used=${used}MiB peak=${peak[$t]:-?}MiB step=${laststep[$t]:-none} \
log=$OUTDIR/log_$t.txt" ;;
    esac
  done

  # announce arms that vanished
  for t in $seen; do
    case " $cur " in *" $t "*) continue ;; esac
    case " $done_ " in *" $t "*) continue ;; esac
    done_="${done_:-} $t"
    lg="$OUTDIR/log_$t.txt"
    verdict="clean exit (no traceback found)"
    if grep -q "OutOfMemoryError" "$lg" 2>/dev/null; then
      verdict="DIED: CUDA OutOfMemoryError"
    elif grep -q "Traceback" "$lg" 2>/dev/null; then
      verdict="DIED: traceback ($(grep -m1 -A0 'Error' "$lg" 2>/dev/null | tail -1))"
    fi
    say "EXITED  tag=$t peak=${peak[$t]:-?}MiB laststep=${laststep[$t]:-none} -> $verdict"
    if grep -q "OutOfMemoryError" "$lg" 2>/dev/null; then
      say "  OOM detail: $(grep -m1 -oE 'this process has [0-9.]+ GiB memory in use' "$lg" 2>/dev/null)"
      say "  free at death: used=${used}MiB"
    fi
  done

  sleep "$POLL"
done
