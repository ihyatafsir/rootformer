#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deliverable 5: are the *_translated.json files faithful TRANSLATIONS?

Each record has: arabic_text, translation, anchors, title_ar, title_en.
We print paired arabic_text / translation / anchors for inspection, and measure
how much of the record is translation vs synthesised 'anchors'.
"""
import json
import glob
import os
import sys

FILES = [
    "/workspace/rootformer_v12/raw_translations/raghib__al_mufradat_fi_gharib_al_quran_translated.json",
    "/workspace/rootformer_v12/raw_translations/raghib__al_dhariah_ila_makarim_al_shariah_translated.json",
    "/workspace/rootformer_v12/raw_translations/raghib__tafsil_al_nashatayn_translated.json",
    "/workspace/rootformer_v12/raw_translations/raghib__muhadarat_al_udaba_translated.json",
    "/workspace/rootformer_v12/raw_translations/raghib__jami_al_tafsir_translated.json",
    "/workspace/rootformer_v12/raw_translations/raghib__adab_ikhtilat_al_nas_translated.json",
]
K = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def main():
    for p in FILES:
        d = json.load(open(p, encoding="utf-8"))
        print("=" * 100)
        print(os.path.basename(p), "items:", len(d))
        tot_ar = sum(len(it.get("arabic_text") or "") for it in d)
        tot_tr = sum(len(it.get("translation") or "") for it in d)
        tot_an = sum(len(it.get("anchors") or "") for it in d)
        print("  chars: arabic_text=%d translation=%d anchors=%d"
              % (tot_ar, tot_tr, tot_an))
        idxs = [0, len(d) // 2, len(d) - 1][:K]
        for i in idxs:
            it = d[i]
            print("  ---- item %d : chapter_index=%s" % (i, it.get("chapter_index")))
            print("   title_ar : %s" % (it.get("title_ar") or "")[:120])
            print("   title_en : %s" % (it.get("title_en") or "")[:160])
            print("   ARABIC   : %s" % (it.get("arabic_text") or "")[:600])
            print("   TRANSL   : %s" % (it.get("translation") or "")[:700])
            print("   ANCHORS  : %s" % (it.get("anchors") or "")[:400])


if __name__ == "__main__":
    main()
