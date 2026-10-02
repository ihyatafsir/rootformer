#!/usr/bin/env python3
"""Extract every citation from the 15 grammar algorithm modules.

A "citation" is either:
  (A) a structured record: a dict literal carrying keys among
      book/work/author + file/corpus + line + arabic/text/quote   -> machine checkable
  (B) an inline quotation: an Arabic-bearing string constant (docstring / literal)
      or a '#' comment that also names an author or work           -> needs manual triage

Outputs citations.json in the audit dir.
"""
import ast
import json
import os
import re
import sys

AR = re.compile(r'[\u0600-\u06FF]')
AR_RUN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s\u060C\u061B\u061F\u0640\u064B-\u0652\u0670\u06D6-\u06ED]+')

MODULES = [
    "andalusian_realizer.py",
    "ibn_malik_automaton.py",
    "sibawayh_governor.py",
    "khalil_orbits.py",
    "models/validated_segmentation.py",
    "basran_syntactic_engine.py",
    "andalusian_grammatical_algorithms.py",
    "khalil_students_andalusian_master_engine.py",
    "farahidian_syntactic_realizer.py",
    "models/ibn_jinni_ishtiqaq.py",
    "classical_governance_v2.py",
    "iktifa_apocope.py",
    "irab_realizer.py",
    "constituent_stack.py",
    "tasrif_engine.py",
]

BOOK_KEYS = {"book", "work", "author", "source_book", "kitab"}
FILE_KEYS = {"file", "corpus", "path", "corpus_file", "src"}
LINE_KEYS = {"line", "lineno", "line_no", "lines"}
TEXT_KEYS = {"arabic", "text", "quote", "quotation", "matn", "arabic_text"}

ATTRIB_AR = [
    "الخليل", "سيبويه", "ابن جني", "ابن جنى", "الزجاجي", "الزجاج", "ابن مالك",
    "أبو حيان", "ابو حيان", "الشاطبي", "الشنتمري", "ابن عصفور", "المبرد",
    "ابن فارس", "الجوهري", "الأزهري", "ابن دريد", "ابن السكيت", "الأخفش",
    "ابن منظور", "الزمخشري", "الراغب", "الكتاب", "العين", "الخصائص",
    "سر صناعة", "المقتضب", "الصاحبي", "المقاييس", "ارتشاف", "التذييل",
    "الموافقات", "الألفية", "التسهيل", "شرح الكافية", "لامية", "الإيضاح",
    "معاني القرآن", "الجمهرة", "الاشتقاق", "المفردات", "الذريعة", "أساس البلاغة",
    "الصحاح", "الصحاح", "التهذيب", "إصلاح المنطق", "المحكم", "المخصص",
    "الممتع", "المقاصد", "نتائج الفكر", "الرد على النحاة", "جمل الزجاجي",
]
ATTRIB_LAT = [
    "al-Khalil", "al-Khalīl", "Khalil", "Sibawayh", "Sībawayh", "Ibn Jinni",
    "Ibn Jinnī", "Ibn Malik", "Ibn Mālik", "Abu Hayyan", "Abū Ḥayyān",
    "Shatibi", "Shāṭibī", "Zajjaji", "Zajjājī", "Usfur", "Usfūr", "Mubarrad",
    "Ibn Faris", "Jawhari", "Azhari", "Ibn Durayd", "Akhfash", "Zamakhshari",
    "Raghib", "al-Ayn", "al-Kitab", "al-Kitāb", "al-Khasais", "Sirr Sinat",
    "al-Muqtadab", "al-Sahibi", "Maqayis", "Irtishaf", "Tadhyil", "Muwafaqat",
    "Alfiyyah", "Tashil", "Lamiyyat", "al-Idah", "al-Mumti", "Sharh Jumal",
]


def has_attrib(s):
    for t in ATTRIB_AR:
        if t in s:
            return t
    for t in ATTRIB_LAT:
        if t in s:
            return t
    return None


def norm_light(s):
    """Remove diacritics/tatweel; collapse whitespace. Keeps letters + punctuation."""
    s = re.sub(r'[\u064B-\u0652\u0670\u0640\u06D6-\u06ED]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def clean_py_string(s):
    return s


def walk_dicts(tree, src_lines):
    """Yield (node, dict-value) for every ast.Dict that looks like a citation record."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        d = {}
        for k, v in zip(node.keys, node.values):
            if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                continue
            key = k.value.strip().lower()
            if isinstance(v, ast.Constant):
                d[key] = v.value
                d.setdefault('__lines__', {})[key] = v.lineno
            elif isinstance(v, (ast.List, ast.Tuple)) and v.elts and all(
                    isinstance(e, ast.Constant) for e in v.elts):
                d[key] = [e.value for e in v.elts]
                d.setdefault('__lines__', {})[key] = v.lineno
        if len(d) <= 1:
            continue
        keys = set(d)
        has_book = bool(keys & BOOK_KEYS)
        has_file = bool(keys & FILE_KEYS)
        has_text = bool(keys & TEXT_KEYS)
        if has_text and (has_book or has_file):
            out.append((node.lineno, d))
    return out


def find_assign_name(tree, target_node):
    """Find the enclosing assignment target name for a dict node (best effort)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    for sub in ast.walk(node.value):
                        if sub is target_node:
                            return t.id
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            for sub in ast.walk(node.value):
                if sub is target_node:
                    return node.target.id
    return None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/hf_v19_2_release"
    outpath = sys.argv[2] if len(sys.argv) > 2 else "/workspace/grammar_audit/citations.json"

    records = []
    stats = {}

    for rel in MODULES:
        path = os.path.join(root, rel)
        with open(path, encoding='utf-8') as fh:
            src = fh.read()
        lines = src.split('\n')
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            print("PARSE FAIL", rel, e)
            continue

        seen_nodes = set()
        n_struct = 0
        n_inline = 0

        # ---- (A) structured citation dicts
        for lineno, d in walk_dicts(tree, lines):
            book = next((d[k] for k in BOOK_KEYS if k in d), None)
            sfile = next((d[k] for k in FILE_KEYS if k in d), None)
            sline = next((d[k] for k in LINE_KEYS if k in d), None)
            arabic = next((d[k] for k in TEXT_KEYS if k in d), None)
            if arabic is None:
                continue
            if isinstance(arabic, list):
                arabic = " ".join(str(x) for x in arabic)
            if not isinstance(arabic, str) or not AR.search(arabic):
                continue
            key = (rel, lineno, arabic[:40])
            if key in seen_nodes:
                continue
            seen_nodes.add(key)
            n_struct += 1
            records.append({
                "module": rel,
                "module_line": lineno,
                "kind": "structured",
                "book": book,
                "src_file": sfile,
                "src_line": sline,
                "arabic": arabic,
            })

        # ---- (B) inline quotations: string constants + comments
        # string constants (docstrings and literals)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value
                if not AR.search(s):
                    continue
                at = has_attrib(s)
                if not at:
                    continue
                # skip ones already captured as structured arabic
                for r in records:
                    if r["module"] == rel and r["arabic"] == s:
                        break
                else:
                    n_inline += 1
                    records.append({
                        "module": rel,
                        "module_line": node.lineno,
                        "kind": "string",
                        "book": at,
                        "src_file": None,
                        "src_line": None,
                        "arabic": s,
                    })
        # comments
        for i, ln in enumerate(lines, 1):
            st = ln.strip()
            if not st.startswith('#'):
                continue
            if not AR.search(st):
                continue
            body = st.lstrip('#').strip()
            at = has_attrib(body)
            if not at:
                continue
            n_inline += 1
            records.append({
                "module": rel,
                "module_line": i,
                "kind": "comment",
                "book": at,
                "src_file": None,
                "src_line": None,
                "arabic": body,
            })

        stats[rel] = {"structured": n_struct, "inline": n_inline,
                      "total": n_struct + n_inline}

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w', encoding='utf-8') as fh:
        json.dump({"records": records, "stats": stats}, fh,
                  ensure_ascii=False, indent=1)
    tot = 0
    for rel in MODULES:
        s = stats.get(rel, {})
        print(f"{rel:62s} struct={s.get('structured',0):3d} inline={s.get('inline',0):3d} tot={s.get('total',0):3d}")
        tot += s.get('total', 0)
    print(f"{'TOTAL':62s} {tot:3d}")
    print("categories:", {
        "structured": sum(s['structured'] for s in stats.values()),
        "string": sum(1 for r in records if r['kind'] == 'string'),
        "comment": sum(1 for r in records if r['kind'] == 'comment'),
    })


if __name__ == '__main__':
    main()
