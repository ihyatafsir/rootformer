taxonomy_out/ -- closed taxonomy of mode-1 (shipped default) round-trip failures
Pinned release: /workspace/taxpin (== the revision that produced du_m1.json, proven by pin_verify.py, 0/188722 mismatches)

failures_m1.jsonl   83,111 lines, the COMPLETE failing distinct-form set:
                    {w, count, p, r, wazn, s, dec, cause, stratum, side, resp}
taxonomy.json       cause -> types/occurrences/shares/strata/10 examples; meta has module md5s
taxonomy.md         the human report (table, residual, fixes, projection, judgement)
cumulative.json     projected token/type round-trip for cumulative top-N fixes (assumed + simulated)
sim_fixes.json      simulated per-fix gains and the union (what a real fix stack achieves)
e1_probe.json       seeded 1,500-form probe of the E1 (<PARTICLE>) residual
mode_delta.json     mode 0/1/2 paired deltas (mode-2 root re-assignment regressions)
resources.json      the tokenizer's own tables (roots, tokens, particles, awzan, implemented wazn)
