#!/bin/bash
# chain_ngram.sh -- durable GPU gate for the single discrete-pathway 20k run.
#
# It waits until the GPU has NO compute process AT ALL and near-zero used memory, confirms that
# twice 60 s apart, and only then runs run_ngram.sh.  It never signals, renices or kills another
# agent's process -- the 120,000-step curve_lm.py run is left completely alone.
#
# Mirrors the previous agent's chain_fix.sh, but keys on nvidia-smi's compute-app list rather
# than on a fixed set of script names, so it also waits for GPU work started by other agents.
LOG=/workspace/discrete_path/chain_ngram.log
echo "[chain_ngram] armed $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $LOG
for i in $(seq 1 5000); do
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${N:-99}" -eq 0 ] && [ -n "${MEM:-}" ] && [ "$MEM" -lt 500 ]; then
    sleep 60
    N2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    M2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ "${N2:-99}" -eq 0 ] && [ -n "${M2:-}" ] && [ "$M2" -lt 500 ]; then
      echo "[chain_ngram] GPU FREE at $(date -u +%Y-%m-%dT%H:%M:%SZ) after $i polls (mem ${M2} MiB)" >> $LOG
      bash /workspace/discrete_path/run_ngram.sh >> $LOG 2>&1
      RC=$?
      echo "[chain_ngram] run_ngram.sh finished rc=$RC at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $LOG
      exit 0
    fi
  fi
  if [ $((i % 30)) -eq 0 ]; then
    echo "[chain_ngram] poll $i $(date -u +%H:%M:%S) gpu_procs=$N mem=${MEM}MiB" >> $LOG
  fi
  sleep 20
done
echo "[chain_ngram] TIMEOUT at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $LOG
exit 1
