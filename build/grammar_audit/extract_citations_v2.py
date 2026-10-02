#!/usr/bin/env python3
"""Citation extractor v2 for the grammar algorithm modules.

Two families:
  A. STRUCTURED records -- dict literals with (book|work|author) + (file|corpus) + line
     + (arabic|text|quote). These are the ones the project itself calls "VERBATIM CITATIONS".
  B. INLINE quotations -- Arabic spans inside «...» (the project's own quotation marker),
     plus Arabic spans in double quotes "..." that contain Arabic script.
     Each is attached to its enclosing function/docstring/comment context for attribution.

Writes citations2.json
"""
import ast
import json
import os
import re
import sys

AR = re.compile(r'[\u0600-\u06FF]')
AR_WORD = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+')
# strip diacritics / tatweel for "word" counting
DIAC = re.compile(r'[\u064B-\u0652\u0670\u0640\u06D6-\u06ED]')

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
LINE_KEYS = {"line", "lineno", "line_no"}
TEXT_KEYS = {"arabic", "text", "quote", "quotation", "matn", "arabic_text"}

# author / work tokens used both for attribution and for locating the named work
ATTRIB = [
    "الخليل", "سيبويه", "ابن جني", "ابن جنى", "الزجاجي", "الزجاج", "ابن مالك",
    "أبو حيان", "ابو حيان", "الشاطبي", "الشنتمري", "ابن عصفور", "المبرد",
    "ابن فارس", "الجوهري", "الأزهري", "ابن دريد", "ابن السكيت", "الأخفش",
    "ابن منظور", "الزمخشري", "الراغب", "ابن مضاء", "ابن سيده", "السخاوي",
    "الكتاب", "العين", "الخصائص", "سر صناعة", "المقتضب", "الصاحبي", "المقاييس",
    "ارتشاف", "التذييل", "الموافقات", "الألفية", "التسهيل", "شرح الكافية",
    "لامية", "الإيضاح", "معاني القرآن", "الجمهرة", "الاشتقاق", "المفردات",
    "الذريعة", "أساس البلاغة", "الصحاح", "التهذيب", "إصلاح المنطق", "المحكم",
    "المخصص", "الممتع", "المقاصد", "نتائج الفكر", "الرد على النحاة", "جمل الزجاجي",
    "al-Khalil", "al-Khalīl", "Khalil", "Sibawayh", "Sībawayh", "Ibn Jinni",
    "Ibn Jinnī", "Ibn Jinni's", "Ibn Malik", "Ibn Mālik", "Abu Hayyan",
    "Abū Ḥayyān", "Shatibi", "Shāṭibī", "Zajjaji", "Zajjājī", "Usfur", "Usfūr",
    "Ibn 'Usfur", "Ibn Usfur", "Mubarrad", "Ibn Faris", "Jawhari", "Azhari",
    "Ibn Durayd", "Akhfash", "Zamakhshari", "Raghib", "al-Ayn", "al-Kitab",
    "al-Kitāb", "al-Khasais", "al-Khasa'is", "Sirr Sinat", "Sirr Sina'at",
    "al-Muqtadab", "al-Sahibi", "Maqayis", "Irtishaf", "Tadhyil", "Tadhyīl",
    "Muwafaqat", "Alfiyyah", "al-Alfiyyah", "Tashil", "Tashīl", "Lamiyyat",
    "al-Idah", "al-Mumti", "Sharh Jumal", "al-Muhkam", "al-Mukhassas",
    "Nata'ij", "al-Suhayli", "Suhayli", "Ibn Mada", "Ibn Mada'",
]


def ar_words(s):
    s = DIAC.sub('', s)
    return AR_WORD.findall(s)


def has_attrib(s):
    for t in ATTRIB:
        if t in s:
            return t
    return None


def line_of_offset(src, off):
    return src.count('\n', 0, off) + 1


def find_assign_name(tree, target):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    for sub in ast.walk(node.value):
                        if sub is target:
                            return t.id
    return None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/hf_v19_2_release"
    outpath = sys.argv[2] if len(sys.argv) > 2 else "/workspace/grammar_audit/citations2.json"

    struct, inline = [], []
    struct_strings = set()

    for rel in MODULES:
        path = os.path.join(root, rel)
        with open(path, encoding='utf-8') as fh:
            src = fh.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        # ---------- A: structured records ----------
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            d = {}
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                        and isinstance(v, ast.Constant):
                    d[k.value.strip().lower()] = v.value
            keys = set(d)
            if not (keys & TEXT_KEYS):
                continue
            arabic = next((d[k] for k in TEXT_KEYS if k in d), None)
            if not isinstance(arabic, str) or not AR.search(arabic):
                continue
            book = next((d[k] for k in BOOK_KEYS if k in d), None)
            sfile = next((d[k] for k in FILE_KEYS if k in d), None)
            sline = next((d[k] for k in LINE_KEYS if k in d), None)
            container = find_assign_name(tree, node)
            for k in (BOOK_KEYS | FILE_KEYS | TEXT_KEYS):
                if k in d and isinstance(d[k], str):
                    struct_strings.add(d[k])
            struct.append({
                "module": rel, "module_line": node.lineno, "kind": "structured",
                "container": container, "book": book, "src_file": sfile,
                "src_line": sline, "arabic": arabic,
            })

        # ---------- B: inline «...» spans ----------
        for m in re.finditer(r'«([^«»]{2,2000})»', src, re.S):
            q = m.group(1)
            if not AR.search(q):
                continue
            w = ar_words(q)
            if len(w) < 2 and len(DIAC.sub('', q)) < 12:
                continue
            ln = line_of_offset(src, m.start())
            # context: the enclosing docstring/comment block, up to 600 chars before
            ctx = src[max(0, m.start() - 700):m.end() + 200]
            inline.append({
                "module": rel, "module_line": ln, "kind": "guillemet",
                "book": has_attrib(ctx), "src_file": None, "src_line": None,
                "arabic": q, "n_words": len(w), "ctx": ctx,
            })

        # ---------- B2: Arabic inside "..." double quotes ----------
        for m in re.finditer(r'"([^"\n]{2,2000})"', src):
            q = m.group(1)
            if not AR.search(q):
                continue
            if q in struct_strings:
                continue
            w = ar_words(q)
            if len(w) < 3:
                continue
            ln = line_of_offset(src, m.start())
            ctx = src[max(0, m.start() - 700):m.end() + 200]
            inline.append({
                "module": rel, "module_line": ln, "kind": "dquote",
                "book": has_attrib(ctx), "src_file": None, "src_line": None,
                "arabic": q, "n_words": len(w), "ctx": ctx,
            })

    # de-duplicate inline on (module, arabic)
    seen = set()
    ded = []
    for r in inline:
        k = (r["module"], r["arabic"])
        if k in seen:
            continue
        seen.add(k)
        ded.append(r)
    inline = sorted(ded, key=lambda r: (r["module"], r["module_line"]))

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w', encoding='utf-8') as fh:
        json.dump({"structured": struct, "inline": inline}, fh,
                  ensure_ascii=False, indent=1)

    from collections import Counter
    print("STRUCTURED:", len(struct))
    for k, v in Counter(r["module"] for r in struct).most_common():
        print(f"   {k:62s} {v}")
    print("INLINE:", len(inline))
    for k, v in Counter(r["module"] for r in inline).most_common():
        print(f"   {k:62s} {v}")
    print("inline without attribution token:", sum(1 for r in inline if not r["book"]))
    print("TOTAL citations:", len(struct) + len(inline))


if __name__ == '__main__':
    main()
