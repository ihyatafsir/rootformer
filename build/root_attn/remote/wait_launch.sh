#!/bin/bash
# Wait for a free GPU using a PRECISE pattern.  NOTE: `pgrep -f nrmt_train[.]py` also matches a
# stale bash whose argv contains "md5sum nrmt_train.py ...", which is why the match is anchored
# on "nrmt_train[.]py --checkpoint" (only a real trainer has that).
for i in $(seq 1 400); do
  T=$(pgrep -f "nrmt_train[.]py --checkpoint" | wc -l)
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "$T" -eq 0 ] && [ -n "$MEM" ] && [ "$MEM" -lt 800 ]; then
    sleep 5
    if [ "$(pgrep -f 'nrmt_train[.]py --checkpoint' | wc -l)" -ne 0 ]; then continue; fi
    echo "[wait] GPU free at $(date -u +%H:%M:%S) mem=${MEM}MiB after $i polls"
    exec bash /workspace/head_fix/run_aligned.sh
  fi
  sleep 15
done
echo "[wait] TIMEOUT waiting for a free GPU"; exit 1
