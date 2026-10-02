#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deliverable 4: do al-Dari'a / Tafsil al-Nash'atayn extend Mufradat coverage?

Measured for the 7,720 roots that have NO al-Mufradat entry:
  (a) does the root string occur as a whitespace-delimited token (after
      diacritic stripping) anywhere in the work?   -> lexical occurrence
  (b) is the occurrence an actual lexicographic entry (the root on a line by
      itself, as in al-Mufradat)?                   -> structural entry

Also measures the same two things against the Asas al-Balaghah corpus, which
IS a root-keyed lexicon and therefore can genuinely extend coverage.
"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/workspace/raghib_audit")
from extract_mufradat_index import extract_headers, strip_diac, norm

LEX = "/workspace/rootformer_v12/v18_next_root_morph/data/lisan_3pillar_master_roots.jsonl"
WORKS = {
    "dhariah": "/workspace/scholastic_masters/Raghib_Al_Dhariah.txt",
    "tafsil": "/workspace/scholastic_masters/Raghib_Tafsil_al_Nashatayn.txt",
}
SAN = {
    "dhariah_sanitized": "/workspace/scholastic_sanitized/Raghib_Al_Dhariah.txt",
    "tafsil_sanitized": "/workspace/scholastic_sanitized/Raghib_Tafsil_al_Nashatayn.txt",
}
ZAM = ["/workspace/rootformer/data/rootformer_v7_6_zamakhshari_train.jsonl",
       "/workspace/rootformer/data/rootformer_v7_6_zamakhshari_val.jsonl"]
OUT = "/workspace/raghib_audit/other_works.json"
WEAK = set("وي")


def plain(s):
    p = strip_diac(s)
    return (p.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
             .replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و").replace("ة", "ه"))


def main():
    cov = json.load(open("/workspace/raghib_audit/coverage_census.json", encoding="utf-8"))
    missing = [r["root"] for r in cov["results"] if r["match"] == "none"]
    present = [r["root"] for r in cov["results"] if r["match"] != "none"]
    print("roots missing from al-Mufradat: %d ; present: %d" % (len(missing), len(present)))

    out = {"n_missing": len(missing), "n_present": len(present)}

    # -- other Raghib works -------------------------------------------------
    tokcache, linecache = {}, {}
    for name, path in list(WORKS.items()) + list(SAN.items()):
        raw = open(path, encoding="utf-8").read()
        toks = set()
        for w in re.findall(r"\S+", raw):
            toks.add(plain(w))
        toks.discard("")
        tokcache[name] = toks
        lines = set()
        for l in raw.split("\n"):
            s = l.strip()
            if s and " " not in s and 2 <= len(plain(s)) <= 12:
                lines.add(plain(s))
        linecache[name] = lines
        print("%-20s distinct tokens=%d  short standalone lines=%d"
              % (name, len(toks), len(lines)))

    for name in tokcache:
        occ = [r for r in missing if plain(r) in tokcache[name]]
        ent = [r for r in missing if plain(r) in linecache[name]]
        out[name] = {"token_occurrence": len(occ), "standalone_entry": len(ent),
                     "examples_token": occ[:15], "examples_entry": ent[:15]}
        print("\n%s : of the %d roots MISSING from al-Mufradat," % (name, len(missing)))
        print("   occur as a token somewhere : %d (%.2f%%)" % (len(occ), 100.0 * len(occ) / len(missing)))
        print("   occur as a standalone line : %d (%.2f%%)" % (len(ent), 100.0 * len(ent) / len(missing)))
        print("   e.g. standalone:", ent[:15])

    # -- Asas corpus (a real root-keyed lexicon) ---------------------------
    asas_roots = set()
    for p in ZAM:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            for r in json.loads(line).get("target_roots", []) or []:
                asas_roots.add(plain(r))
    hit = [r for r in missing if plain(r) in asas_roots]
    out["asas"] = {"token_occurrence": len(hit), "standalone_entry": len(hit),
                   "examples": hit[:20]}
    print("\nAsas al-Balaghah corpus: of the %d roots MISSING from al-Mufradat," % len(missing))
    print("   covered by Asas : %d (%.2f%%)" % (len(hit), 100.0 * len(hit) / len(missing)))
    print("   examples:", hit[:20])

    # how many of ALL 9015 does Asas cover
    allroots = [r["root"] for r in cov["results"]]
    hitall = [r for r in allroots if plain(r) in asas_roots]
    out["asas_all"] = len(hitall)
    print("   Asas covers %d of all 9015 lexicon roots (%.2f%%)"
          % (len(hitall), 100.0 * len(hitall) / len(allroots)))
    print("   Raghib/Asas union covers %d (%.2f%%)"
          % (len(set(hitall) | set(present)), 100.0 * len(set(hitall) | set(present)) / len(allroots)))

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
