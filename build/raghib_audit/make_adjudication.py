#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Emit compact per-case adjudication text and a checkability census.

For every sampled root we report which of the three NAMED held sources can
actually check which field:
   al-Mufradat      -> ontological_core, canonical_definition
   Asas al-Balaghah -> haqiqah_literal
   Maqayis al-Lugha -> ontological_core ("Asl al-Wad'" framing)
"""
import json
import os

S = json.load(open("/workspace/raghib_audit/fidelity_sample_enriched.json", encoding="utf-8"))
OUT = "/workspace/raghib_audit/adjudication"
os.makedirs(OUT, exist_ok=True)


def tr(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n] + " […]"


def main():
    b = S["bundle"]
    check = {}
    groups = {"MA_a": [], "Ma": [], "mA": [], "ma_none": []}
    n_muf = n_maq = n_asas = n_any = 0
    for e in b:
        src = e["sources"]
        has = {"mufradat": bool(src.get("mufradat")), "maqayis": bool(src.get("maqayis")),
               "asas": bool(src.get("asas")), "ayn": bool(src.get("ayn")),
               "tahdhib": bool(src.get("tahdhib"))}
        if has["mufradat"]:
            n_muf += 1
        if has["maqayis"]:
            n_maq += 1
        if has["asas"]:
            n_asas += 1
        if any(has.values()):
            n_any += 1
        key = ("mufradat" if has["mufradat"] else "") + ("maqayis" if has["maqayis"] else "") + \
              ("asas" if has["asas"] else "")
        check[e["root"]] = has
        groups.setdefault(key, []).append(e)

    N = len(b)
    print("=== CHECKABILITY of the %d-root sample ===" % N)
    print("  checkable vs al-Mufradat      : %d (%.1f%%)" % (n_muf, 100.0 * n_muf / N))
    print("  checkable vs Maqayis al-Lugha : %d (%.1f%%)" % (n_maq, 100.0 * n_maq / N))
    print("  checkable vs Asas al-Balaghah : %d (%.1f%%)" % (n_asas, 100.0 * n_asas / N))
    print("  checkable vs AT LEAST ONE     : %d (%.1f%%)" % (n_any, 100.0 * n_any / N))
    print("  checkable vs NONE             : %d (%.1f%%)" % (N - n_any, 100.0 * (N - n_any) / N))
    print("\n  source-combination groups:")
    for k in sorted(groups):
        print("     %-14s : %d" % (k or "(none)", len(groups[k])))

    # emit everything, but split by whether a Mufradat entry exists
    order = {"MA_a": 0}
    allc = sorted(b, key=lambda e: (not e["sources"].get("mufradat"),
                                    not e["sources"].get("maqayis"),
                                    not e["sources"].get("asas"),
                                    e["root"]))
    for i, e in enumerate(allc):
        src = e["sources"]
        r = e["record"]
        L = []
        L.append("=" * 96)
        L.append("ROOT %s   stratum=%s" % (e["root"], e["stratum"]))
        L.append("REC ontological_core : %s" % tr(r.get("ontological_core"), 430))
        L.append("REC haqiqah_literal  : %s" % tr(r.get("haqiqah_literal"), 380))
        L.append("REC canon_def.ar     : %s" % tr((r.get("canonical_definition") or {}).get("arabic"), 300))
        L.append("REC canon_def.en     : %s" % tr((r.get("canonical_definition") or {}).get("english"), 240))
        m = src.get("mufradat")
        L.append("SRC al-Mufradat      : %s" % (("[line %s hdr=%s] " % (m["line"], m["header"]) + tr(m["text"], 620)) if m else "(NO ENTRY)"))
        q = src.get("maqayis")
        L.append("SRC Maqayis (IbnFaris): %s" % (("[line %s hdr=%s] " % (q["line"], q["header"]) + tr(q["text"], 620)) if q else "(NO ENTRY)"))
        a = src.get("asas")
        if a:
            for ent in a["entries"][:3]:
                L.append("SRC Asas [%s]        : %s" % (ent["mode"], tr(ent["text"], 480)))
        else:
            L.append("SRC Asas             : (NO ENTRY)")
        an = src.get("ayn")
        L.append("SRC Ayn (al-Khalil)  : %s" % (("[line %s hdr=%s] " % (an["line"], an["header"]) + tr(an["text"], 420)) if an else "(NO ENTRY)"))
        L.append("")
        open(os.path.join(OUT, "case_%03d.txt" % i), "w", encoding="utf-8").write("\n".join(L))

    # one combined file in chunks of 21 cases for reading
    txt = []
    for i, e in enumerate(allc):
        txt.append(open(os.path.join(OUT, "case_%03d.txt" % i), encoding="utf-8").read())
        if (i + 1) % 21 == 0 or i == len(allc) - 1:
            k = i // 21
            open(os.path.join(OUT, "read_%02d.txt" % k), "w", encoding="utf-8").write(
                "\n".join(txt))
            txt = []
    print("\nwrote per-case files and read_*.txt to", OUT)
    json.dump(check, open("/workspace/raghib_audit/checkability.json", "w", encoding="utf-8"),
              ensure_ascii=False)


if __name__ == "__main__":
    main()
