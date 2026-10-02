#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Print 10 (arabic_text, translation) pairs, spread across the six Raghib
works, for faithful-translation adjudication.  Deliberately long excerpts so
omissions and added framing are visible.
"""
import json
import re

RH = "/workspace/rootformer_v12/raw_translations/raghib__%s_translated.json"
PICKS = [
    ("al_mufradat_fi_gharib_al_quran", 20),
    ("al_mufradat_fi_gharib_al_quran", 300),
    ("al_dhariah_ila_makarim_al_shariah", 30),
    ("al_dhariah_ila_makarim_al_shariah", 100),
    ("tafsil_al_nashatayn", 10),
    ("jami_al_tafsir", 100),
    ("jami_al_tafsir", 400),
    ("muhadarat_al_udaba", 200),
    ("adab_ikhtilat_al_nas", 20),
    ("muhadarat_al_udaba", 600),
]


def clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


def main():
    for i, (w, k) in enumerate(PICKS, 1):
        d = json.load(open(RH % w, encoding="utf-8"))
        it = d[min(k, len(d) - 1)]
        print("=" * 100)
        print("[%d] %s  item=%d chapter_index=%s" % (i, w, k, it.get("chapter_index")))
        print("TITLE_EN: %s" % clean(it.get("title_en"))[:150])
        print("-- ARABIC  : %s" % clean(it.get("arabic_text"))[:1100])
        print("-- TRANSL  : %s" % clean(it.get("translation"))[:1400])
        print("-- AR chars=%d  EN chars=%d" % (len(clean(it.get("arabic_text"))),
                                              len(clean(it.get("translation")))))


if __name__ == "__main__":
    main()
