#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recon: what do the auxiliary resources actually contain?

1. zamakhshari_asas_balaghah JSONL  -> is Asas al-Balaghah held, and how many
   distinct roots does it cover?
2. raghib__al_mufradat_..._translated.json -> structure, translation fidelity
3. Raghib_Al_Dhariah.txt / Raghib_Tafsil_al_Nashatayn.txt -> do they carry
   root-keyed entries at all?
"""
import json
import re
from collections import Counter

ZAM = "/workspace/rootformer/data/rootformer_v7_6_zamakhshari_train.jsonl"
ZAMV = "/workspace/rootformer/data/rootformer_v7_6_zamakhshari_val.jsonl"
MT = "/workspace/rootformer_v12/raw_translations/raghib__al_mufradat_fi_gharib_al_quran_translated.json"


def zam():
    print("=" * 70)
    print("ZAMAKHSHARI / ASAS AL-BALAGHAH DATA")
    for p in (ZAM, ZAMV):
        n = 0
        modes = Counter()
        books = Counter()
        authors = Counter()
        roots = set()
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                n += 1
                modes[d.get("mode")] += 1
                books[d.get("source_book")] += 1
                authors[d.get("author")] += 1
                for r in d.get("target_roots", []) or []:
                    roots.add(r)
        print("  %s" % p)
        print("    records=%d distinct target_roots=%d" % (n, len(roots)))
        print("    modes=%s" % modes.most_common())
        print("    source_book=%s" % books.most_common())
        print("    author=%s" % authors.most_common())
        print("    sample roots: %s" % sorted(roots)[:25])


def mt():
    print("=" * 70)
    print("MUFRADAT TRANSLATION JSON")
    d = json.load(open(MT, encoding="utf-8"))
    print("  top-level type:", type(d).__name__)
    if isinstance(d, dict):
        print("  keys:", list(d.keys())[:20])
        for k in list(d.keys())[:3]:
            v = d[k]
            print("   %s -> %s %s" % (k, type(v).__name__,
                                      (str(v)[:300] if not isinstance(v, list) else "len=%d" % len(v))))
    elif isinstance(d, list):
        print("  len:", len(d))
        print("  first item:", json.dumps(d[0], ensure_ascii=False)[:1500])


if __name__ == "__main__":
    zam()
    mt()
