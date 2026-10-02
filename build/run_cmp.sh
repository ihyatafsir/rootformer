set -x
cd /workspace
PY=/workspace/venvs/rootformer/bin/python
for m in 0 1 2; do
  nice -n 10 env ROOTFORMER_VALIDATED_SEG=$m $PY eval_heldout.py --dump-words /workspace/du_m$m.json
done
nice -n 10 $PY eval_heldout.py --compare /workspace/du_m0.json /workspace/du_m1.json
nice -n 10 $PY eval_heldout.py --compare /workspace/du_m1.json /workspace/du_m2.json
nice -n 10 $PY eval_heldout.py --compare /workspace/du_m0.json /workspace/du_m2.json
nice -n 10 $PY eval_heldout.py --report
echo "CMP_DONE rc=$?"
