#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deliverable 3: the citation question.

(a) Which authorities/sources do the 9,015 records actually NAME in their prose?
(b) For each named source, is the text held on the pod?
(c) Per-record: what fraction of the record's own Arabic is verbatim present in
    a held source (character-level coverage, not just the longest run)?
"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/workspace/raghib_audit")
from extract_mufradat_index import strip_diac

LEX = "/workspace/rootformer_v12/v18_next_root_morph/data/lisan_3pillar_master_roots.jsonl"

PATTERNS = {
    "Asas al-Balaghah (al-Zamakhshari)": r"As[āa]s al-Bal[āa]ghah|Asās al-Balāghah|Asas al-Balaghah",
    "al-Mufradat (al-Raghib)": r"[Aa]l-Mufrad[āa]t|Mufradāt|Mufradat",
    "Maqayis al-Lugha (Ibn Faris)": r"Maq[āa]y[īi]s|Maqāyīs|Aṣl al-Wadʿ|Asl al-Wad",
    "Lisan al-Arab (Ibn Manzur)": r"Lis[āa]n al-ʿArab|Lisan al-Arab|Lisān",
    "Kitab al-Ayn (al-Khalil)": r"al-ʿAyn|Kitab al-Ayn|al-Ayn",
    "al-Zamakhshari (by name)": r"Zamakhshar[īi]",
    "al-Raghib (by name)": r"R[āa]ghib|al-Raghib",
    "Ibn Faris (by name)": r"Ibn F[āa]ris",
    "al-Khalil (by name)": r"al-Khal[īi]l",
    "Sibawayh": r"S[īi]bawayh",
    "Quranic verse citation": r"Qur'?[āa]n|\[[A-Z][a-z]+/\s?\d+\]",
}


def main():
    recs = [json.loads(l) for l in open(LEX, encoding="utf-8")]
    n = len(recs)
    fields = ("ontological_core", "haqiqah_literal", "majaz_scholastic")
    counts = Counter()
    per_field = {f: Counter() for f in fields}
    for r in recs:
        blob = " ".join(str(r.get(f) or "") for f in fields)
        for name, pat in PATTERNS.items():
            if re.search(pat, blob):
                counts[name] += 1
                for f in fields:
                    if re.search(pat, str(r.get(f) or "")):
                        per_field[f][name] += 1
    print("=== named sources in the 9,015 records (n=%d) ===" % n)
    for name, c in counts.most_common():
        print("   %-32s %5d  (%.2f%%)" % (name, c, 100.0 * c / n))
    print("\n--- breakdown by field ---")
    for f in fields:
        print("  %s:" % f, per_field[f].most_common())

    print("\n=== 'according to' attributions in haqiqah_literal ===")
    pat = re.compile(r"according to ([^,.;]{3,60})")
    ac = Counter()
    for r in recs:
        for m in pat.finditer(str(r.get("haqiqah_literal") or "")):
            ac[m.group(1).strip()] += 1
    for k, v in ac.most_common(20):
        print("   %5d  %s" % (v, k))

    print("\n=== 'Aṣl al-Wadʿ' framing in ontological_core ===")
    for probe in ["Aṣl al-Wadʿ", "Asl al-Wad", "foundational physical origin"]:
        c = sum(1 for r in recs if probe in str(r.get("ontological_core") or ""))
        print("   %-32s %5d (%.2f%%)" % (probe, c, 100.0 * c / n))

    print("\n=== identical-text reuse (template contamination) ===")
    for f in fields:
        vals = Counter(str(r.get(f) or "") for r in recs)
        dup = {k: v for k, v in vals.items() if v > 1 and len(k) > 80}
        print("   %-18s distinct=%d  values shared by >1 root=%d"
              % (f, len(vals), len(dup)))
        for k, v in sorted(dup.items(), key=lambda x: -x[1])[:3]:
            print("        x%d: %s" % (v, k[:130]))


if __name__ == "__main__":
    main()
