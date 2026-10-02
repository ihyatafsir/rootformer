#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a generic root-entry index for every held root-keyed lexicon, then
enrich the 260-root fidelity sample with the actual entry text from each.
Also dumps readable per-batch text files for adjudication.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/workspace/raghib_audit")
from extract_mufradat_index import extract_headers, strip_diac, norm

LEX = "/workspace/rootformer_v12/v18_next_root_morph/data/lisan_3pillar_master_roots.jsonl"
SAMPLE = "/workspace/raghib_audit/fidelity_sample.json"
OUTJ = "/workspace/raghib_audit/fidelity_sample_enriched.json"
OUTDIR = "/workspace/raghib_audit/batches"

LEXICONS = {
    "mufradat": "/workspace/scholastic_masters/Raghib_Al_Mufradat.txt",
    "maqayis": "/workspace/heritage_foundations/Ibn_Faris_Mujam_Maqayis_Al_Lughah.txt",
}
# Works whose entries are INLINE, i.e. a line beginning "<entry>: <body>"
INLINE = {
    "ayn": "/workspace/heritage_foundations/Al_Khalil_Al_Ayn.txt",
    "sihah": "/workspace/heritage_foundations/Al_Jawhari_Al_Sihah.txt",
    "tahdhib": "/workspace/heritage_foundations/Al_Azhari_Tahdhib_Al_Lughah.txt",
    "jamharat": "/workspace/heritage_foundations/Ibn_Durayd_Jamharat_Al_Lughah.txt",
}
ZAM = ["/workspace/rootformer/data/rootformer_v7_6_zamakhshari_train.jsonl",
       "/workspace/rootformer/data/rootformer_v7_6_zamakhshari_val.jsonl"]
DIAC = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
ARABIC_ALL = re.compile(r"^[\u0621-\u064A\u0670-\u06D3\u064B-\u0652\u0640]+$")
CHAPTER_WORDS = ("كتاب", "باب", "فصل", "مقدمة", "ترجمة", "خاتمة", "مقدمة", "التعريف")
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
        if s:
            out.append(s)
    s = " ".join(out).replace("%~%", " ")
    return re.sub(r"\s+", " ", s).strip()


def build(path, maxlen=12):
    lines = open(path, encoding="utf-8").read().split("\n")
    hdrs = []
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("|||"):
            s = s[3:].strip()
        if not s or " " in s:
            continue
        if not ARABIC_ALL.match(s):
            continue
        p = strip_diac(s)
        if not (2 <= len(p) <= maxlen):
            continue
        if any(p.startswith(w) for w in CHAPTER_WORDS):
            continue
        hdrs.append({"line": i + 1, "header": s, "plain": p, "norm": norm(s)})
    return lines, hdrs


def build_inline(path):
    """Entries of the form '<root>: <body>' at start of a line.

    Returns (entries, line_of).  Body runs until the next such entry line.
    """
    lines = open(path, encoding="utf-8").read().split("\n")
    pat = re.compile(r"^([\u0621-\u064A\u0670-\u06D3]{2,6})\s*:")
    hits = []
    for i, raw in enumerate(lines):
        m = pat.match(raw.strip())
        if m:
            hits.append((i, m.group(1)))
    ent = {}
    for j, (i, word) in enumerate(hits):
        end = hits[j + 1][0] if j + 1 < len(hits) else len(lines)
        first = lines[i].strip()
        body = first.split(":", 1)[1] if ":" in first else ""
        rest = clean_body("\n".join(lines[i + 1:end]))
        ent.setdefault(norm(word), []).append(
            {"line": i + 1, "header": word, "text": clean_body(body + " " + rest)[:4000]})
    return ent, len(hits)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    idx = {}
    stats = {}
    for name, path in LEXICONS.items():
        lines, hdrs = build(path)
        m = {}
        for i, h in enumerate(hdrs):
            m.setdefault(h["norm"], i)

        def get(root, m=m, hdrs=hdrs, lines=lines):
            i = m.get(norm(root))
            if i is None:
                p = plain(root)
                if len(p) == 3 and p[2] in WEAK:
                    i = m.get(p[:2])
            if i is None:
                return None
            start = hdrs[i]["line"]
            end = hdrs[i + 1]["line"] - 1 if i + 1 < len(hdrs) else len(lines)
            return {"header": hdrs[i]["header"], "line": hdrs[i]["line"],
                    "text": clean_body("\n".join(lines[start:end]))}
        idx[name] = get
        stats[name] = len(hdrs)
        print("%-9s headers=%d" % (name, len(hdrs)))

    inline_entries = {}
    for name, path in INLINE.items():
        ent, n = build_inline(path)
        inline_entries[name] = ent
        print("%-9s inline entries=%d (distinct roots=%d)" % (name, n, len(ent)))

        def getin(root, ent=ent):
            v = ent.get(norm(root))
            if v is None:
                p = plain(root)
                if len(p) == 3 and p[2] in WEAK:
                    v = ent.get(p[:2])
            if not v:
                return None
            return {"header": v[0]["header"], "line": v[0]["line"], "text": v[0]["text"]}
        idx[name] = getin

    # Asas
    asas = defaultdict(list)
    for p in ZAM:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tt = d.get("target_text") or ""
            mm = re.search(r"\):\s*(.*)", tt, re.S)
            body = clean_body(mm.group(1) if mm else tt)
            for r in d.get("target_roots", []) or []:
                asas[norm(r)].append({"mode": d.get("mode"), "text": body})

    recs = {json.loads(l)["root"]: json.loads(l) for l in open(LEX, encoding="utf-8")}
    S = json.load(open(SAMPLE, encoding="utf-8"))

    # lexicon coverage for maqayis
    allroots = list(recs)
    for name in ("maqayis", "ayn", "sihah", "tahdhib", "jamharat"):
        c = sum(1 for r in allroots if idx[name](r))
        print("%s covers %d of 9015 lexicon roots (%.2f%%)" % (name, c, 100.0 * c / len(allroots)))
    both = sum(1 for r in allroots if idx["mufradat"](r) or idx["maqayis"](r))
    allunion = sum(1 for r in allroots
                   if idx["mufradat"](r) or idx["maqayis"](r)
                   or norm(r) in asas or idx["ayn"](r) or idx["sihah"](r)
                   or idx["tahdhib"](r) or idx["jamharat"](r))
    print("mufradat|maqayis union: %d (%.2f%%)" % (both, 100.0 * both / len(allroots)))
    print("union of ALL held lexicons: %d (%.2f%%)" % (allunion, 100.0 * allunion / len(allroots)))

    out = []
    for b in S["bundle"]:
        root = b["root"]
        e = dict(b)
        e["sources"] = {n: idx[n](root) for n in list(LEXICONS) + list(INLINE)}
        e["sources"]["asas"] = ({"entries": asas.get(norm(root), [])}
                                if norm(root) in asas else None)
        out.append(e)
    json.dump({"seed": S["seed"], "n": len(out), "alloc": S["alloc"],
               "strata_sizes": S["strata_sizes"], "lexicon_headers": stats,
               "bundle": out}, open(OUTJ, "w", encoding="utf-8"), ensure_ascii=False)
    print("wrote", OUTJ)

    # readable batches
    K = 20
    for bi in range(0, len(out), K):
        chunk = out[bi:bi + K]
        lines = []
        for e in chunk:
            r = e["record"]
            lines.append("=" * 100)
            lines.append("ROOT %s   stratum=%s" % (e["root"], e["stratum"]))
            lines.append("-- RECORD.ontological_core: %s" % r.get("ontological_core"))
            lines.append("-- RECORD.haqiqah_literal: %s" % r.get("haqiqah_literal"))
            lines.append("-- RECORD.majaz_scholastic: %s" % r.get("majaz_scholastic"))
            cd = r.get("canonical_definition") or {}
            lines.append("-- RECORD.canonical_definition.ar: %s" % cd.get("arabic"))
            lines.append("-- RECORD.canonical_definition.en: %s" % cd.get("english"))
            for n in ("mufradat", "maqayis", "asas", "ayn", "sihah", "tahdhib", "jamharat"):
                s = e["sources"].get(n)
                if not s:
                    lines.append("-- SOURCE.%s: (none)" % n)
                elif n == "asas":
                    for ent in s["entries"][:4]:
                        lines.append("-- SOURCE.asas[%s]: %s" % (ent["mode"], ent["text"][:700]))
                else:
                    lines.append("-- SOURCE.%s @line %s hdr=%s: %s"
                                 % (n, s["line"], s["header"], s["text"][:700]))
            lines.append("")
        open(os.path.join(OUTDIR, "batch_%02d.txt" % (bi // K)), "w",
             encoding="utf-8").write("\n".join(lines))
    print("wrote %d batches to %s (%d roots each)" % ((len(out) + K - 1) // K, OUTDIR, K))


if __name__ == "__main__":
    main()
