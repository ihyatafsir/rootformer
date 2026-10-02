#!/bin/bash
# run_seven_suites.sh -- the seven regression suites the brief requires after a module edit.
cd /workspace/hf_v19_2_release || exit 90
PY=/workspace/venvs/rootformer/bin/python
OUT=/workspace/discrete_path/suites_after_discrete.log
: > $OUT
run() {
  echo "########## $1 ##########" >> $OUT
  timeout 900 $PY "$1" >> $OUT 2>&1
  echo "exit=$?" >> $OUT
  tail -2 $OUT | sed 's/^/    /'
}
run test_grammar_impl.py
run verify_v2.py
run andalusian_realizer.py
run test_sibawayh_governor.py
run ibn_malik_automaton.py
run test_khalil_orbits.py
run test_awzan_order.py
echo "SUITES_DONE" >> $OUT
grep -nE "RESULT|checks pass|PASS.*FAIL|exit=" $OUT | tail -40
echo SUITES_ALL_DONE
