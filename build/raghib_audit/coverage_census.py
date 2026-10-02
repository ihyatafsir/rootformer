#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deliverable 1 (v2): exhaustive coverage census, with a *principled* match.

Rules
-----
A lexicon root R is counted as HAVING a Mufradat entry iff either

  EXACT   : norm(R) == norm(H) for some entry header H, or
  SHORT2  : H is a 2-letter surface form of a 3-radical weak root, i.e.
            len(plain(H)) == 2, norm(R)[:2] == norm(H), and R's third
            radical is a weak letter (و/ي).  e.g. H='أب' for R='أبو',
            H='أخ' for R='أخو', H='يد' for R='يدي'.

A looser "drop all weak letters" rule was tested and REJECTED: it collapses
genuinely distinct roots (e.g. R='بوغ' would match H='بغي'; R='جور' would
match H='أجر').  No weak-letter substitution (و<->ي) is permitted.
"""
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/workspace/raghib_audit")
from extract_mufradat_index import extract_headers, strip_diac, norm

LEX = "/workspace/rootformer_v12/v18_next_root_morph/data/lisan_3pillar_master_roots.jsonl"
OUT = "/workspace/raghib_audit/coverage_census.json"
WEAK = set("وي")


def plain(s):
    p = strip_diac(s)
    return (p.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
             .replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و").replace("ة", "ه"))


def main():
    lines, headers = extract_headers()
    by_norm = defaultdict(list)
    for h in headers:
        by_norm[h["norm"]].append(h)

    recs = [json.loads(l) for l in open(LEX, encoding="utf-8")]
    n = len(recs)

    results = []
    for r in recs:
        root = r["root"]
        nr = norm(root)
        m = None
        how = None
        if nr in by_norm:
            m = by_norm[nr][0]
            how = "exact"
        else:
            p = plain(root)
            if len(p) == 3 and p[2] in WEAK:
                for cand in (p[:2],):
                    if cand in by_norm:
                        m = by_norm[cand][0]
                        how = "short2"
                        break
        results.append({"root": root, "match": how or "none",
                        "header": m["header"] if m else None,
                        "line": m["line"] if m else None})

    cov = [x for x in results if x["match"] != "none"]
    nocov = [x for x in results if x["match"] == "none"]
    print("=== COVERAGE CENSUS (exhaustive, n=%d) ===" % n)
    print("Mufradat entry headers parsed        : %d" % len(headers))
    print("roots WITH a Mufradat entry          : %d  (%.2f%%)" % (len(cov), 100.0 * len(cov) / n))
    print("   of which exact-form match         : %d" % sum(1 for x in cov if x["match"] == "exact"))
    print("   of which 2-letter weak-root match : %d" % sum(1 for x in cov if x["match"] == "short2"))
    print("roots WITHOUT a Mufradat entry       : %d  (%.2f%%)" % (len(nocov), 100.0 * len(nocov) / n))

    # by root length
    print("\n--- coverage by lexicon root length ---")
    bylen = defaultdict(lambda: [0, 0])
    for x in results:
        L = len(plain(x["root"]))
        bylen[L][1] += 1
        if x["match"] != "none":
            bylen[L][0] += 1
    for L in sorted(bylen):
        c, t = bylen[L]
        print("   len=%d : %4d / %4d covered  (%.1f%%)" % (L, c, t, 100.0 * c / t))

    print("\n--- roots with NO Mufradat entry, sample of 25 ---")
    for x in nocov[:25]:
        print("   ", x["root"])

    # Mufradat headers that the lexicon does not contain -> the converse gap
    lex_norms = set(norm(x["root"]) for x in results)
    unused = [h for h in headers if h["norm"] not in lex_norms]
    print("\n--- Mufradat headers ABSENT from the 9015 lexicon: %d ---" % len(unused))
    for h in unused[:60]:
        print("   ", h["header"], "@", h["line"])

    per_letter = Counter(h["plain"][0] for h in headers)
    print("\nMufradat headers per first letter:", sorted(per_letter.items()))

    json.dump({"n": n, "headers": len(headers),
               "covered": len(cov), "uncovered": len(nocov),
               "by_len": {str(k): v for k, v in bylen.items()},
               "results": results,
               "unused_headers": [{"header": h["header"], "line": h["line"]} for h in unused]},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
