#!/usr/bin/env python3
"""Extractor v4 -- generic citation-record detection.

v3 assumed the text key was `arabic`.  In fact the modules use at least two
schemas:  {book, file, line, arabic}  (andalusian_realizer, khalil_orbits)
and      {book, file, ar, note}      (ibn_malik_automaton, sibawayh_governor).
This version keys on the *shape* of the dict, not on one spelling, and parses
`file:NNN` / `file:NNN-MMM` locators out of the file value.
"""
import ast
import json
import os
import re
import sys

AR = re.compile(r'[\u0600-\u06FF]')
AR_WORD = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+')
DIAC = re.compile(r'[\u064B-\u0652\u0670\u0640\u06D6-\u06ED]')

MODULES = [
    "andalusian_realizer.py", "ibn_malik_automaton.py", "sibawayh_governor.py",
    "khalil_orbits.py", "models/validated_segmentation.py", "basran_syntactic_engine.py",
    "andalusian_grammatical_algorithms.py", "khalil_students_andalusian_master_engine.py",
    "farahidian_syntactic_realizer.py", "models/ibn_jinni_ishtiqaq.py",
    "classical_governance_v2.py", "iktifa_apocope.py", "irab_realizer.py",
    "constituent_stack.py", "tasrif_engine.py",
]

TEXT_KEYS = {"arabic", "ar", "text", "quote", "quotation", "matn", "arabic_text", "arabictext"}
BOOK_KEYS = {"book", "work", "author", "source_book", "kitab", "source", "ref", "reference"}
FILE_KEYS = {"file", "corpus", "path", "corpus_file", "src", "corpus_path"}
LINE_KEYS = {"line", "lineno", "line_no", "lines", "loc"}
LOC_RE = re.compile(r'^\s*(?P<path>[^\s:]+?\.txt)\s*(?::\s*(?P<a>\d+)\s*(?:[-–]\s*(?P<b>\d+))?)?\s*$')


def ar_words(s):
    return AR_WORD.findall(DIAC.sub('', s))


def line_of(src, off):
    return src.count('\n', 0, off) + 1


def parse_loc(v):
    """Return (path, line) from a value like 'corpus/x.txt:3562-3565'."""
    if not isinstance(v, str):
        return None, None
    m = LOC_RE.match(v.strip())
    if m:
        a = m.group('a')
        return m.group('path'), (int(a) if a else None)
    return v.strip(), None


def main():
    root, outpath = sys.argv[1], sys.argv[2]
    struct, inline = [], []
    struct_strings = set()
    schema_counter = {}

    for rel in MODULES:
        src = open(os.path.join(root, rel), encoding='utf-8').read()
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            d, dlines = {}, {}
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                        and isinstance(v, ast.Constant):
                    d[k.value.strip().lower()] = v.value
                    dlines[k.value.strip().lower()] = v.lineno
            if not d:
                continue
            tk = sorted(set(d) & TEXT_KEYS)
            if not tk:
                continue
            arabic = next((d[k] for k in tk if isinstance(d[k], str) and AR.search(d[k])), None)
            if arabic is None or len(ar_words(arabic)) < 3:
                continue
            bk = sorted(set(d) & BOOK_KEYS)
            fk = sorted(set(d) & FILE_KEYS)
            lk = sorted(set(d) & LINE_KEYS)
            if not bk and not fk:
                continue
            book = d[bk[0]] if bk else None
            rawf = d[fk[0]] if fk else None
            path, fline = parse_loc(rawf)
            sline = fline
            if sline is None and lk:
                try:
                    sline = int(str(d[lk[0]]).strip())
                except (TypeError, ValueError):
                    sline = None
            schema = tuple(sorted(tk) + sorted(bk) + sorted(fk) + sorted(lk))
            schema_counter[schema] = schema_counter.get(schema, 0) + 1
            for k in list(d):
                if isinstance(d[k], str):
                    struct_strings.add(d[k])
            struct.append({
                "module": rel, "module_line": node.lineno, "kind": "structured",
                "book": book, "src_file": path, "src_line": sline,
                "src_raw": rawf, "arabic": arabic})

        doc_ranges = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                b = getattr(node, 'body', None)
                if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                        and isinstance(b[0].value.value, str):
                    doc_ranges.append((b[0].lineno, b[0].end_lineno))

        def in_docstring(ln):
            return any(a <= ln <= b for a, b in doc_ranges)

        cand = []
        for m in re.finditer(r'«([^«»]{2,2000})»', src, re.S):
            q = m.group(1)
            if not AR.search(q):
                continue
            if len(ar_words(q)) < 2 and len(DIAC.sub('', q)) < 12:
                continue
            cand.append((m.start(), m.end(), q, 'guillemet'))
        for m in re.finditer(r'"([^"\n]{2,2000})"', src):
            q = m.group(1)
            if not AR.search(q) or q in struct_strings:
                continue
            if len(ar_words(q)) < 3:
                continue
            cand.append((m.start(), m.end(), q, 'dquote'))

        for a, b, q, kind in cand:
            ln = line_of(src, a)
            inline.append({
                "module": rel, "module_line": ln, "kind": kind,
                "arabic": q, "n_words": len(ar_words(q)),
                "pre": src[max(0, a - 500):a], "post": src[b:b + 200],
                "in_docstring": in_docstring(ln)})

    seen, ded = set(), []
    for r in inline:
        k = (r["module"], r["arabic"])
        if k in seen:
            continue
        seen.add(k)
        ded.append(r)
    ded.sort(key=lambda r: (r["module"], r["module_line"]))

    json.dump({"structured": struct, "inline": ded},
              open(outpath, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    from collections import Counter
    print("structured:", len(struct))
    for k, v in Counter(r['module'] for r in struct).most_common():
        print(f"   {k:62s} {v}")
    print("inline:", len(ded))
    for k, v in Counter(r['module'] for r in ded).most_common():
        print(f"   {k:62s} {v}")
    print("\nSCHEMAS:")
    for k, v in schema_counter.items():
        print(f"  {v:4d}  {k}")


if __name__ == '__main__':
    main()
