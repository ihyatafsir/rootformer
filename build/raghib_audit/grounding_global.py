#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Global verbatim-grounding test over ALL 9,015 records (efficient).

Question: the record supplies Arabic in `canonical_definition.arabic`.  Is any
substantial run of that Arabic present VERBATIM in a source we actually hold?

Held sources searched, each normalised (diacritics/tatweel removed;
alef/ya/ta-marbuta unified; non-letters except space dropped):
  mufradat : Raghib_Al_Mufradat.txt
  dhariah  : Raghib_Al_Dhariah.txt
  tafsil   : Raghib_Tafsil_al_Nashatayn.txt
  asas     : the zamakhshari_asas_balaghah JSONL corpus

Method: word-anchored longest-common-substring.  For each record we anchor on
its 3 longest distinct words, look up their offsets in a prebuilt word index of
the haystack, and extend left/right while characters match.  This finds the
longest common run that contains one of those anchor words, which for any run
>=15 Arabic characters (i.e. spanning at least two words) is a safe detection.
The method is cross-validated on the 260-root sample against an exact
difflib-based longest-match.
"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/workspace/raghib_audit")
from extract_mufradat_index import strip_diac

LEX = "/workspace/rootformer_v12/v18_next_root_morph/data/lisan_3pillar_master_roots.jsonl"
TEXTS = {
    "mufradat": "/workspace/scholastic_masters/Raghib_Al_Mufradat.txt",
    "dhariah": "/workspace/scholastic_masters/Raghib_Al_Dhariah.txt",
    "tafsil": "/workspace/scholastic_masters/Raghib_Tafsil_al_Nashatayn.txt",
}
ZAM = ["/workspace/rootformer/data/rootformer_v7_6_zamakhshari_train.jsonl",
       "/workspace/rootformer/data/rootformer_v7_6_zamakhshari_val.jsonl"]
OUT = "/workspace/raghib_audit/grounding_global.json"
MAX_OCC = 30000


def prep(s, keep_space=True):
    s = strip_diac(s)
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
          .replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و").replace("ة", "ه"))
    if keep_space:
        s = re.sub(r"[^\u0621-\u064A ]", " ", s)
        return re.sub(r"\s+", " ", s).strip()
    return re.sub(r"[^\u0621-\u064A]", "", s)


def build_index(H):
    idx = {}
    for m in re.finditer(r"\S+", H):
        idx.setdefault(m.group(0), []).append(m.start())
    return idx


def longest_run(a, H, idx):
    best, bfrag = 0, ""
    words = sorted(set(re.findall(r"\S+", a)), key=len, reverse=True)[:3]
    for w in words:
        if len(w) < 3:
            continue
        pos_list = idx.get(w)
        if not pos_list or len(pos_list) > MAX_OCC:
            continue
        for mm in re.finditer(re.escape(w), a):
            ms, me = mm.start(), mm.end()
            for pos in pos_list:
                i, j = ms - 1, pos - 1
                while i >= 0 and j >= 0 and a[i] == H[j]:
                    i -= 1
                    j -= 1
                ls, lj = i + 1, j + 1
                i2, j2 = me, pos + len(w)
                while i2 < len(a) and j2 < len(H) and a[i2] == H[j2]:
                    i2 += 1
                    j2 += 1
                L = i2 - ls
                if L > best:
                    best, bfrag = L, a[ls:i2]
    return best, bfrag


def main():
    H = {}
    IDX = {}
    for k, p in TEXTS.items():
        H[k] = prep(open(p, encoding="utf-8").read())
        IDX[k] = build_index(H[k])
        print("%-9s normalised len=%9d  distinct words=%d" % (k, len(H[k]), len(IDX[k])))
    zam = []
    for p in ZAM:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                zam.append(prep(json.loads(line).get("target_text") or ""))
    H["asas"] = prep(" ".join(zam))
    IDX["asas"] = build_index(H["asas"])
    print("%-9s normalised len=%9d  distinct words=%d" % ("asas", len(H["asas"]), len(IDX["asas"])))

    recs = [json.loads(l) for l in open(LEX, encoding="utf-8")]
    hist = Counter()
    per_src = Counter()
    rows = []
    for r in recs:
        cd = (r.get("canonical_definition") or {}).get("arabic") or ""
        a = prep(cd)
        if len(a) < 12:
            rows.append({"root": r["root"], "alen": len(a), "best": 0, "src": None, "frag": ""})
            hist["too_short"] += 1
            continue
        best, bsrc, bfrag = 0, None, ""
        for k in ("mufradat", "dhariah", "tafsil", "asas"):
            L, frag = longest_run(a, H[k], IDX[k])
            if L > best:
                best, bsrc, bfrag = L, k, frag
        rows.append({"root": r["root"], "alen": len(a), "best": best, "src": bsrc,
                     "frag": bfrag, "ar": a})
        if best >= 40:
            hist["ge40"] += 1
        elif best >= 25:
            hist["25-39"] += 1
        elif best >= 15:
            hist["15-24"] += 1
        else:
            hist["lt15"] += 1
        if best >= 25:
            per_src[bsrc] += 1

    n = len(rows)
    print("\n=== VERBATIM GROUNDING of canonical_definition.arabic (all %d records) ===" % n)
    print("longest verbatim run found in ANY held source:")
    for k in ("ge40", "25-39", "15-24", "lt15", "too_short"):
        print("   %-9s : %5d  (%.2f%%)" % (k, hist[k], 100.0 * hist[k] / n))
    print("\nrecords with a >=25-char verbatim run, by source:", per_src.most_common())
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
