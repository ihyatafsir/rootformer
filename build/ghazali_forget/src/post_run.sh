#!/bin/bash
# post_run.sh -- fires when the two EWC arms are done and produces the head-independent measure
# (FIX head on each arm's saved trunk, RCA off) with the same code that reproduced 6.641 %/2.433 %,
# then a combined table.  CPU only; touches no GPU and kills nothing.
set -u
cd /workspace/hf_v19_2_release
PY=/workspace/venvs/rootformer/bin/python
O=/workspace/ghazali_forget
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=
LOG=$O/post_run.log
log(){ echo "[post $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "waiting for the gate to launch the arms (pid files appear only then)"
for T in EWC_A_RUNA6000_LRET EWC_B_A26000_LHEAD; do
  i=0
  while [ ! -f $O/pid_$T.txt ] && [ $i -lt 960 ]; do sleep 30; i=$((i+1)); done
  if [ ! -f $O/pid_$T.txt ]; then log "$T never launched (gate timeout?)"; continue; fi
  log "$T launched (pid $(cat $O/pid_$T.txt)); waiting for it to finish"
  P=$(cat $O/pid_$T.txt)
  while [ -n "$P" ] && [ "$P" != "0" ] && [ -d /proc/$P ]; do sleep 60; done
  log "$T finished (pid $P gone) after ${i} polls of setup"
done

for T in EWC_A_RUNA6000_LRET EWC_B_A26000_LHEAD; do
  if [ -f $O/head_$T.pt.trunk.pt ]; then
    log "probing FIX head on $T trunk (live, no RCA)"
    $PY $O/ewc_head_probe.py --live --val-windows 300 --tag $T \
        --trunk $O/head_$T.pt.trunk.pt --out $O/probe_head_$T.json >> $LOG 2>&1
    log "$T probe rc=$?"
  else
    log "$T has no trunk payload saved (still at step <1000?)"
  fi
done

log "writing combined arm report"
$PY $O/ewc_arm_report.py EWC_A_RUNA6000_LRET EWC_B_A26000_LHEAD >> $LOG 2>&1
log "ARM_REPORT rc=$?"
log POST_RUN_DONE
