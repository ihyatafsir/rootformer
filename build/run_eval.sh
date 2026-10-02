set -x
cd /workspace
PY=/workspace/venvs/rootformer/bin/python
rm -f /workspace/eval_heldout_sample.json /workspace/eval_heldout_results.json
nice -n 10 env ROOTFORMER_VALIDATED_SEG=0 $PY eval_heldout.py --per-file-cap 50000 --force-scan
nice -n 10 env ROOTFORMER_VALIDATED_SEG=1 $PY eval_heldout.py --per-file-cap 50000
nice -n 10 env ROOTFORMER_VALIDATED_SEG=2 $PY eval_heldout.py --per-file-cap 50000
nice -n 10 $PY eval_heldout.py --report
echo "ALL_DONE rc=$?"
