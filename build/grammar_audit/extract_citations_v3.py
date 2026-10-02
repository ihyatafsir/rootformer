#!/usr/bin/env python3
"""Extractor v3: same as v2 but each inline span also carries `pre`/`post`
(module source immediately around the quotation) and `in_docstring`, so the
verifier can attribute each quote to the *nearest* named source instead of
any name within a 700-char window."""
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

BOOK_KEYS = {"book", "work", "author", "source_book", "kitab"}
FILE_KEYS = {"file", "corpus", "path", "corpus_file", "src"}
LINE_KEYS = {"line", "lineno", "line_no"}
TEXT_KEYS = {"arabic", "text", "quote", "quotation", "matn", "arabic_text"}


def ar_words(s):
    return AR_WORD.findall(DIAC.sub('', s))


def line_of(src, off):
    return src.count('\n', 0, off) + 1


def main():
    root = sys.argv[1]
    outpath = sys.argv[2]
    struct, inline = [], []
    struct_strings = set()

    for rel in MODULES:
        src = open(os.path.join(root, rel), encoding='utf-8').read()
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            d = {}
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                        and isinstance(v, ast.Constant):
                    d[k.value.strip().lower()] = v.value
            if not (set(d) & TEXT_KEYS):
                continue
            arabic = next((d[k] for k in TEXT_KEYS if k in d), None)
            if not isinstance(arabic, str) or not AR.search(arabic):
                continue
            for k in (BOOK_KEYS | FILE_KEYS | TEXT_KEYS):
                if k in d and isinstance(d[k], str):
                    struct_strings.add(d[k])
            struct.append({
                "module": rel, "module_line": node.lineno, "kind": "structured",
                "book": next((d[k] for k in BOOK_KEYS if k in d), None),
                "src_file": next((d[k] for k in FILE_KEYS if k in d), None),
                "src_line": next((d[k] for k in LINE_KEYS if k in d), None),
                "arabic": arabic})

        # docstring line ranges
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
                "pre": src[max(0, a - 500):a],
                "post": src[b:b + 200],
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
    print("structured:", len(struct), "inline:", len(ded))


if __name__ == '__main__':
    main()
