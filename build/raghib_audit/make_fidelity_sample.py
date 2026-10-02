#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deliverable 2 prep: stratified random sample + side-by-side fidelity bundles.

Strata (2 x 2), because two independent witnesses exist for a given root:
   M = al-Raghib's al-Mufradat has an entry for the root (yes/no)
   A = the Asas al-Balaghah corpus has an entry for the root (yes/no)
Seed is fixed and reported.
"""
import json
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, "/workspace/raghib_audit")
from extract_mufradat_index import extract_headers, strip_diac, norm

LEX = "/workspace/rootformer_v12/v18_next_root_morph/data/lisan_3pillar_master_roots.jsonl"
ZAM = ["/workspace/rootformer/data/rootformer_v7_6_zamakhshari_train.jsonl",
       "/workspace/rootformer/data/rootformer_v7_6_zamakhshari_val.jsonl"]
SEED = 20241017
TOTAL_TARGET = 260

WEAK = set("وي")


def plain(s):
    p = strip_diac(s)
    return (p.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
             .replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و").replace("ة", "ه"))


def clean_body(txt):
    out = []
    for l in txt.split("\n"):
        s = l.strip()
        if s.startswith("~~"):
            s = s[2:].strip()
        if not s:
            continue
        out.append(s)
    s = " ".join(out)
    s = s.replace("%~%", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def main():
    lines, headers = extract_headers()
    hdr_by_norm = {}
    for i, h in enumerate(headers):
        hdr_by_norm.setdefault(h["norm"], i)

    def muf_entry(root):
        nr = norm(root)
        i = hdr_by_norm.get(nr)
        if i is None:
            p = plain(root)
            if len(p) == 3 and p[2] in WEAK:
                i = hdr_by_norm.get(p[:2])
        if i is None:
            return None
        start = headers[i]["line"]                     # 1-based header line
        end = headers[i + 1]["line"] - 1 if i + 1 < len(headers) else len(lines)
        body = "\n".join(lines[start:end])             # text after the header
        return {"header": headers[i]["header"], "line": headers[i]["line"],
                "text": clean_body(body)}

    # Asas index
    asas = defaultdict(list)
    for p in ZAM:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tt = d.get("target_text") or ""
            m = re.search(r"\):\s*(.*)", tt, re.S)
            body = m.group(1) if m else tt
            for r in d.get("target_roots", []) or []:
                asas[r].append({"mode": d.get("mode"), "text": clean_body(body)})
    asas_norm = {}
    for r, v in asas.items():
        asas_norm.setdefault(norm(r), (r, v))

    recs = [json.loads(l) for l in open(LEX, encoding="utf-8")]

    strata = defaultdict(list)
    for r in recs:
        root = r["root"]
        M = muf_entry(root) is not None
        A = norm(root) in asas_norm or plain(root) in asas_norm
        strata[("M" if M else "m") + ("A" if A else "a")].append(root)

    print("=== STRATUM SIZES (full population) ===")
    for k in sorted(strata):
        print("   %s : %d" % (k, len(strata[k])))
    print("   total: %d" % sum(len(v) for v in strata.values()))

    rng = random.Random(SEED)
    # proportional allocation with a floor of 30 per non-empty stratum
    alloc = {}
    for k, v in strata.items():
        alloc[k] = max(30, round(TOTAL_TARGET * len(v) / len(recs)))
    # keep total near target by trimming the largest stratum
    while sum(alloc.values()) > TOTAL_TARGET:
        big = max(alloc, key=lambda k: alloc[k])
        if alloc[big] <= 30:
            break
        alloc[big] -= 1
    print("\n=== ALLOCATION (seed=%d) ===" % SEED)
    for k in sorted(alloc):
        print("   %s : %d of %d (%.1f%%)" % (k, alloc[k], len(strata[k]),
                                             100.0 * alloc[k] / len(strata[k])))

    sample = []
    for k in sorted(alloc):
        pool = sorted(strata[k])
        rng.shuffle(pool)
        for root in pool[:alloc[k]]:
            sample.append(root)
    sample = sorted(set(sample))
    print("\nsampled roots: %d" % len(sample))

    by_root = {r["root"]: r for r in recs}
    bundle = []
    for root in sample:
        rec = by_root[root]
        me = muf_entry(root)
        an = asas_norm.get(norm(root)) or asas_norm.get(plain(root))
        bundle.append({
            "root": root,
            "stratum": ("M" if me else "m") + ("A" if an else "a"),
            "record": {k: rec.get(k) for k in
                       ("ontological_core", "haqiqah_literal", "majaz_scholastic",
                        "canonical_definition", "derivations")},
            "mufradat": me,
            "asas": ({"root": an[0], "entries": an[1]} if an else None),
        })

    out = "/workspace/raghib_audit/fidelity_sample.json"
    json.dump({"seed": SEED, "n": len(bundle), "alloc": {k: v for k, v in alloc.items()},
               "strata_sizes": {k: len(v) for k, v in strata.items()},
               "bundle": bundle}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("wrote", out)

    # quick automated ground-truth metric: does the record's Arabic canonical
    # definition have a long verbatim run inside the Mufradat entry?
    print("\n=== quick metric: verbatim overlap of canonical_definition.arabic with Mufradat ===")
    import difflib
    hits = miss = none = 0
    for b in bundle:
        cd = (b["record"].get("canonical_definition") or {}).get("arabic") or ""
        if not b["mufradat"] or not cd:
            none += 1
            continue
        sm = difflib.SequenceMatcher(None, cd, b["mufradat"]["text"])
        m = sm.find_longest_match(0, len(cd), 0, len(b["mufradat"]["text"]))
        if m.size >= 25:
            hits += 1
        else:
            miss += 1
    print("   longest verbatim run >=25 chars: %d ; <25 chars: %d ; not-checkable: %d"
          % (hits, miss, none))


if __name__ == "__main__":
    main()
