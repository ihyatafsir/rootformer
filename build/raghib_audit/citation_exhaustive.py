#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exhaustive checks that explain WHERE the lexicon's content comes from.

A. Citation falsity over ALL 9,015 records:
     - how many records claim Asas al-Balaghah in haqiqah_literal while the
       Asas corpus has NO entry for that root?
     - how many use the "Asl al-Wad'" (Ibn Faris) framing while Maqayis has no
       entry for that root?
     - how many name al-Mufradat while Mufradat has no entry?
B. Provenance: is the artifact's prose derived from the synthesised `anchors`
   layer of the *_translated.json files?  Measured by longest verbatim run of
   the record's canonical_definition.english against the concatenated anchors,
   and of canonical_definition.arabic against the concatenated arabic_text.
"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/workspace/raghib_audit")
from extract_mufradat_index import strip_diac, norm

LEX = "/workspace/rootformer_v12/v18_next_root_morph/data/lisan_3pillar_master_roots.jsonl"
RH = "/workspace/rootformer_v12/raw_translations/raghib__%s_translated.json"
WORKS = ["al_mufradat_fi_gharib_al_quran", "al_dhariah_ila_makarim_al_shariah",
         "muhadarat_al_udaba", "tafsil_al_nashatayn", "jami_al_tafsir",
         "adab_ikhtilat_al_nas"]
WEAK = set("وي")


def prep_ar(s):
    s = strip_diac(s)
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
          .replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و").replace("ة", "ه"))
    s = re.sub(r"[^\u0621-\u064A ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def prep_en(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()


def main():
    cov = json.load(open("/workspace/raghib_audit/coverage_census.json", encoding="utf-8"))
    have_muf = set(r["root"] for r in cov["results"] if r["match"] != "none")
    enr = json.load(open("/workspace/raghib_audit/fidelity_sample_enriched.json", encoding="utf-8"))
    # maqayis coverage for all roots
    import subprocess
    print("loading maqayis index ...")
    sys.path.insert(0, "/workspace/raghib_audit")
    import enrich_bundle as EB
    lines, hdrs = EB.build(EB.LEXICONS["maqayis"])
    mmap = {}
    for i, h in enumerate(hdrs):
        mmap.setdefault(h["norm"], i)
    have_maq = set()
    recs = [json.loads(l) for l in open(LEX, encoding="utf-8")]
    for r in recs:
        p = EB.plain(r["root"])
        if r["root"] in mmap or norm(r["root"]) in mmap:
            have_maq.add(r["root"])
        elif len(p) == 3 and p[2] in WEAK and p[:2] in mmap:
            have_maq.add(r["root"])
    print("maqayis covers %d roots" % len(have_maq))

    ZAM = ["/workspace/rootformer/data/rootformer_v7_6_zamakhshari_train.jsonl",
           "/workspace/rootformer/data/rootformer_v7_6_zamakhshari_val.jsonl"]
    have_asas = set()
    for p in ZAM:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            for rr in json.loads(line).get("target_roots", []) or []:
                have_asas.add(norm(rr))
    print("asas covers %d distinct normalised roots" % len(have_asas))
    # map lexicon roots into asas normalised space
    def in_asas(root):
        p = EB.plain(root)
        return norm(root) in have_asas or p in have_asas or (
            len(p) == 3 and p[2] in WEAK and p[:2] in have_asas)

    n = len(recs)
    claim_asas = claim_maq = claim_muf = 0
    fals_asas = fals_maq = fals_muf = 0
    for r in recs:
        hl = str(r.get("haqiqah_literal") or "")
        oc = str(r.get("ontological_core") or "")
        mj = str(r.get("majaz_scholastic") or "")
        if re.search(r"As[āa]s al-Bal", hl):
            claim_asas += 1
            if not in_asas(r["root"]):
                fals_asas += 1
        if "Aṣl al-Wadʿ" in oc or "Asl al-Wad" in oc:
            claim_maq += 1
            if r["root"] not in have_maq:
                fals_maq += 1
        if re.search(r"Mufrad[āa]t", oc + mj):
            claim_muf += 1
            if r["root"] not in have_muf:
                fals_muf += 1

    print("\n=== A. CITATION FALSITY (all %d records) ===" % n)
    for name, cl, fa, src in (("haqiqah_literal -> Asas al-Balaghah", claim_asas, fals_asas, "Asas corpus"),
                              ("ontological_core -> 'Asl al-Wad' (Ibn Faris)", claim_maq, fals_maq, "Maqayis"),
                              ("ontological_core/majaz -> al-Mufradat", claim_muf, fals_muf, "al-Mufradat")):
        print("  %-46s claims=%5d  source has NO entry=%5d (%.1f%% of claims)"
              % (name, cl, fa, 100.0 * fa / cl if cl else 0))

    # B. provenance vs translated JSONs
    print("\n=== B. provenance: overlap with the *_translated.json layers ===")
    anchors, artxt = [], []
    for w in WORKS:
        d = json.load(open(RH % w, encoding="utf-8"))
        for it in d:
            if it.get("anchors"):
                anchors.append(it["anchors"])
            if it.get("arabic_text"):
                artxt.append(it["arabic_text"])
    anchor_tok = set(prep_en(" ".join(anchors)))
    print("distinct English tokens in all anchors: %d" % len(anchor_tok))
    ar_pos = {}
    blob = prep_ar(" ".join(artxt))
    for m in re.finditer(r"\S+", blob):
        ar_pos.setdefault(m.group(0), []).append(m.start())
    print("arabic_text normalised length: %d" % len(blob))

    def lcs_ar(a):
        best, frag = 0, ""
        for w in sorted(set(re.findall(r"\S+", a)), key=len, reverse=True)[:3]:
            if len(w) < 3:
                continue
            pos = ar_pos.get(w)
            if not pos or len(pos) > 30000:
                continue
            for mm in re.finditer(re.escape(w), a):
                for p0 in pos:
                    i, j = mm.start() - 1, p0 - 1
                    while i >= 0 and j >= 0 and a[i] == blob[j]:
                        i -= 1
                        j -= 1
                    ls = i + 1
                    i2, j2 = mm.end(), p0 + len(w)
                    while i2 < len(a) and j2 < len(blob) and a[i2] == blob[j2]:
                        i2 += 1
                        j2 += 1
                    if i2 - ls > best:
                        best, frag = i2 - ls, a[ls:i2]
        return best, frag

    hist = Counter()
    for r in recs:
        cd = (r.get("canonical_definition") or {})
        a = prep_ar(cd.get("arabic") or "")
        if len(a) < 12:
            hist["too_short"] += 1
            continue
        best, frag = lcs_ar(a)
        hist["ge30" if best >= 30 else ("15-29" if best >= 15 else "lt15")] += 1
    print("canonical_definition.arabic longest verbatim run inside the Raghib")
    print("  translation JSONs' arabic_text:", hist.most_common())

    # English side: share of records whose canonical_definition.english is
    # largely covered by the anchors vocabulary
    covs = []
    for r in recs:
        e = ((r.get("canonical_definition") or {}).get("english") or "")
        t = prep_en(e)
        if not t:
            continue
        covs.append(sum(1 for x in set(t) if x in anchor_tok) / len(set(t)))
    if covs:
        covs.sort()
        print("canonical_definition.english: mean share of its distinct tokens that")
        print("  occur anywhere in the anchors = %.3f ; median = %.3f"
              % (sum(covs) / len(covs), covs[len(covs) // 2]))


if __name__ == "__main__":
    main()
