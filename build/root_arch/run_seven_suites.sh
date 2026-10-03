#!/usr/bin/env bash
# The seven suites the brief requires after ANY module edit.  root_cross_attn.py is not exercised
# by any of them, so a PASS here shows the edit did not disturb the shared module tree.
PY=/workspace/venvs/rootformer/bin/python
REL=/workspace/hf_v19_2_release
S=/workspace/transmute_v2/suites
run() {
  local name="$1"; shift
  local out; out=$(cd "$2" && timeout 900 "$PY" "$1" 2>&1); local rc=$?
  local last; last=$(echo "$out" | grep -aiE "passed|failed|[0-9]+/[0-9]+|OK|PASS" | tail -2 | tr "\n" " ")
  printf "  %-28s exit=%-3s %s\n" "$name" "$rc" "$last"
}
run "test_grammar_impl.py"      "$S/test_grammar_impl.py"      "$S"
run "verify_v2.py"              "$S/verify_v2.py"              "$S"
run "andalusian_realizer.py"    "$REL/andalusian_realizer.py"  "$REL"
run "test_sibawayh_governor.py" "$S/test_sibawayh_governor.py" "$S"
run "ibn_malik_automaton.py"    "$REL/ibn_malik_automaton.py"  "$REL"
run "test_khalil_orbits.py"     "$S/test_khalil_orbits.py"     "$S"
run "test_awzan_order.py"       "$S/test_awzan_order.py"       "$S"
