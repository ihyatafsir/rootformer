#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VERIFY the adjudication output.

Three mechanical checks, because an unverified quote is worthless:
  1. root set / ordering matches the sample exactly;
  2. every UNCHECKABLE verdict corresponds to a source that genuinely has no
     entry, and vice versa;
  3. every non-empty quote is an EXACT substring of the real source text
     (tested raw, and after diacritic/orthography normalisation), and is at
     least 15 Arabic characters long.

Anything failing (2) or (3) is a fabricated or misplaced quotation and is
reported for manual re-adjudication.
"""
import json
import os
import re
import glob
import sys
from collections import Counter, defaultdict

ENR = "/workspace/raghib_audit/fidelity_sample_enriched.json"
VD = "/workspace/raghib_audit/verdicts"

DIAC = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")


def norm(s):
    s = DIAC.sub("", s or "")
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
          .replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و").replace("ة", "ه"))
    return re.sub(r"[^\u0621-\u064A]", "", s)


def main():
    S = json.load(open(ENR, encoding="utf-8"))
    bundle = {e["root"]: e for e in S["bundle"]}
    expected = [e["root"] for e in S["bundle"]]

    cases = []
    for p in sorted(glob.glob(os.path.join(VD, "batch_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        for c in d["cases"]:
            c["_file"] = os.path.basename(p)
            cases.append(c)
    print("verdicts loaded: %d (expected %d)" % (len(cases), len(expected)))

    got = [c["root"] for c in cases]
    if got != expected:
        miss = [r for r in expected if r not in set(got)]
        extra = [r for r in got if r not in set(expected)]
        print("!! ROOT SET MISMATCH  missing=%d extra=%d" % (len(miss), len(extra)))
        print("   missing:", miss[:20])
        print("   extra  :", extra[:20])
    else:
        print("OK: root set and order match the sample exactly")

    checks = {"mufradat": "mufradat", "maqayis": "maqayis", "asas": "asas"}
    fails = []
    stats = defaultdict(Counter)
    for c in cases:
        e = bundle.get(c["root"])
        if e is None:
            fails.append((c["root"], "root-not-in-bundle", ""))
            continue
        for k, srckey in checks.items():
            v = c.get(k + "_verdict")
            q = (c.get(k + "_quote") or "").strip()
            src = e["sources"].get(srckey)
            has = bool(src)
            stats[k][v] += 1
            if v == "UNCHECKABLE":
                if has:
                    fails.append((c["root"], k + ":UNCHECKABLE-but-source-exists", ""))
                if q:
                    fails.append((c["root"], k + ":UNCHECKABLE-with-nonempty-quote", q[:60]))
                continue
            if not has:
                fails.append((c["root"], k + ":verdict-but-no-source", ""))
                continue
            if k == "asas":
                text = " ||| ".join(x["text"] for x in src["entries"])
            else:
                text = src["text"]
            if len(norm(q)) < 15:
                fails.append((c["root"], k + ":quote-too-short(%d)" % len(norm(q)), q[:60]))
                continue
            if q in text or norm(q) in norm(text):
                continue
            # allow the quote to be verified against the file itself
            fails.append((c["root"], k + ":QUOTE-NOT-FOUND", q[:90]))

    print("\n=== verdict distributions (as returned) ===")
    for k in checks:
        tot = sum(stats[k].values())
        print("  %-9s n=%d  " % (k, tot) + "  ".join(
            "%s=%d(%.1f%%)" % (v, stats[k][v], 100.0 * stats[k][v] / tot)
            for v in ("AGREES", "PARTIAL", "WRONG", "UNCHECKABLE")))

    print("\n=== verification failures: %d ===" % len(fails))
    kind = Counter(f[1].split(":")[1].split("(")[0] for f in fails)
    for k, v in kind.most_common():
        print("   %-32s %d" % (k, v))
    for f in fails[:80]:
        print("   %-8s %-38s %s" % (f[0], f[1], f[2]))

    json.dump({"cases": cases, "fails": fails},
              open("/workspace/raghib_audit/verdicts_verified.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    print("\nwrote /workspace/raghib_audit/verdicts_verified.json")


if __name__ == "__main__":
    main()
