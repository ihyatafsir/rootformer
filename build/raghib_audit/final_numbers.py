#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final quantitative layer.

1. Verbatim grounding of canonical_definition.arabic inside al-Mufradat,
   split by whether the root has a Mufradat entry (all 9,015 records).
   Reported as: longest common run, and the fraction of the record's own Arabic
   that is covered by verbatim Mufradat material.
2. Same for the 260-root sample.
3. English provenance: longest verbatim WORD-SEQUENCE shared between
   canonical_definition.english and the synthesised `anchors` layer of the
   Raghib translation JSONs (a real test, unlike token-set overlap).
"""
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/workspace/raghib_audit")
from extract_mufradat_index import strip_diac, norm
sys.path.insert(0, "/workspace/raghib_audit")
import enrich_bundle as EB

LEX = "/workspace/rootformer_v12/v18_next_root_morph/data/lisan_3pillar_master_roots.jsonl"
RH = "/workspace/rootformer_v12/raw_translations/raghib__%s_translated.json"
WORKS = ["al_mufradat_fi_gharib_al_quran", "al_dhariah_ila_makarim_al_shariah",
         "muhadarat_al_udaba", "tafsil_al_nashatayn", "jami_al_tafsir",
         "adab_ikhtilat_al_nas"]


def prep_ar(s):
    s = strip_diac(s)
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
          .replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و").replace("ة", "ه"))
    s = re.sub(r"[^\u0621-\u064A ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def longest_run(a, H, idx, maxocc=30000):
    best, bfrag = 0, ""
    for w in sorted(set(re.findall(r"\S+", a)), key=len, reverse=True)[:3]:
        if len(w) < 3:
            continue
        pos = idx.get(w)
        if not pos or len(pos) > maxocc:
            continue
        for mm in re.finditer(re.escape(w), a):
            for p0 in pos:
                i, j = mm.start() - 1, p0 - 1
                while i >= 0 and j >= 0 and a[i] == H[j]:
                    i -= 1
                    j -= 1
                ls = i + 1
                i2, j2 = mm.end(), p0 + len(w)
                while i2 < len(a) and j2 < len(H) and a[i2] == H[j2]:
                    i2 += 1
                    j2 += 1
                if i2 - ls > best:
                    best, bfrag = i2 - ls, a[ls:i2]
    return best, bfrag


def main():
    lines, hdrs = EB.build(EB.LEXICONS["mufradat"])
    mmap = {}
    for i, h in enumerate(hdrs):
        mmap.setdefault(h["norm"], i)

    def muf_text(root):
        i = mmap.get(norm(root))
        if i is None:
            p = EB.plain(root)
            if len(p) == 3 and p[2] in "وي":
                i = mmap.get(p[:2])
        if i is None:
            return None
        start = hdrs[i]["line"]
        end = hdrs[i + 1]["line"] - 1 if i + 1 < len(hdrs) else len(lines)
        return EB.clean_body("\n".join(lines[start:end]))

    H = prep_ar(open(EB.LEXICONS["mufradat"], encoding="utf-8").read())
    idx = {}
    for m in re.finditer(r"\S+", H):
        idx.setdefault(m.group(0), []).append(m.start())

    recs = [json.loads(l) for l in open(LEX, encoding="utf-8")]
    groups = {"has_mufradat_entry": [], "no_mufradat_entry": []}
    for r in recs:
        mt = muf_text(r["root"])
        a = prep_ar((r.get("canonical_definition") or {}).get("arabic") or "")
        k = "has_mufradat_entry" if mt else "no_mufradat_entry"
        if len(a) < 12:
            groups[k].append((0, 0.0, r["root"]))
            continue
        best, frag = longest_run(a, H, idx)
        groups[k].append((best, best / len(a), r["root"]))

    print("=== verbatim grounding of canonical_definition.arabic in al-Mufradat ===")
    for k, v in groups.items():
        n = len(v)
        ge25 = sum(1 for b, _, _ in v if b >= 25)
        ge40 = sum(1 for b, _, _ in v if b >= 40)
        mean_cov = sum(c for _, c, _ in v) / n
        print("  %-22s n=%5d  >=25 chars verbatim: %5d (%.1f%%)  >=40: %5d (%.1f%%)  "
              "mean character coverage: %.3f"
              % (k, n, ge25, 100.0 * ge25 / n, ge40, 100.0 * ge40 / n, mean_cov))

    V = json.load(open("/workspace/raghib_audit/verdicts_final.json", encoding="utf-8"))
    sm = {c["root"] for c in V}
    sub = [(b, c, rt) for (b, c, rt) in groups["has_mufradat_entry"] if rt in sm]
    print("  sample roots with a Mufradat entry: n=%d ; of those >=25-char verbatim: %d (%.1f%%)"
          % (len(sub), sum(1 for b, _, _ in sub if b >= 25),
             100.0 * sum(1 for b, _, _ in sub if b >= 25) / max(1, len(sub))))

    # cross-tab: verbatim-grounded vs semantic verdict
    verd = {c["root"]: c for c in V}
    print("\n=== cross-tab (sample, roots with a Mufradat entry): semantic verdict x verbatim ===")
    tab = Counter()
    for b, c, rt in sub:
        lab = "verbatim>=25" if b >= 25 else ("verbatim15-24" if b >= 15 else "verbatim<15")
        tab[(verd[rt]["mufradat_verdict"], lab)] += 1
    for k in sorted(tab, key=lambda x: (x[0], x[1])):
        print("   %-11s %-14s %d" % (k[0], k[1], tab[k]))

    # English provenance
    print("\n=== English provenance: longest shared word-sequence with the anchors layer ===")
    anchor_words = []
    for w in WORKS:
        d = json.load(open(RH % w, encoding="utf-8"))
        for it in d:
            anchor_words.extend(re.findall(r"[a-z]+", (it.get("anchors") or "").lower()))
    grams = {}
    N = 8
    for i in range(len(anchor_words) - N + 1):
        grams.setdefault(tuple(anchor_words[i:i + N]), 0)
    print("  anchors word count: %d ; distinct %d-grams: %d" % (len(anchor_words), N, len(grams)))
    hits = tot = 0
    samp = recs[::30][:300]
    for r in samp:
        ws = re.findall(r"[a-z]+", ((r.get("canonical_definition") or {}).get("english") or "").lower())
        tot += 1
        found = False
        for i in range(len(ws) - N + 1):
            if tuple(ws[i:i + N]) in grams:
                found = True
                break
        if found:
            hits += 1
    print("  sampled records (%d) sharing an %d-word verbatim run with anchors: %d (%.1f%%)"
          % (tot, N, hits, 100.0 * hits / tot))


if __name__ == "__main__":
    main()
