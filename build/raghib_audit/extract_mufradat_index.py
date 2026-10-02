#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract a candidate entry index from Raghib_Al_Mufradat.txt.

Header convention observed in the file:
  - Most entries begin with a bare line containing only the entry word
    (e.g. line 4972 == 'بغت').
  - A handful of entries begin with '||| <word>' (e.g. '||| وآناء').
  - '||| <long text>' lines are chapter headers ('كتاب الباء') or section
    headers, and are excluded by the shape test below.
  - '~~' is a line-continuation marker; '%~%' is an inline verse/poetry
    separator.  Neither starts a header.
"""
import json
import re
import sys

SRC = "/workspace/scholastic_masters/Raghib_Al_Mufradat.txt"

DIAC = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
ARABIC_ALL = re.compile(r"^[\u0621-\u064A\u0670-\u06D3\u064B-\u0652\u0640]+$")

CHAPTER_WORDS = ("كتاب", "باب", "فصل", "مقدمة", "ترجمة", "خاتمة")


def strip_diac(s: str) -> str:
    return DIAC.sub("", s)


def norm(s: str) -> str:
    """Normalisation used for matching a lexicon root against a header."""
    s = strip_diac(s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    s = s.replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و")
    s = s.replace("ة", "ه")
    return s


def extract_headers(path=SRC):
    lines = open(path, encoding="utf-8").read().split("\n")
    headers = []
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        marked = False
        if s.startswith("|||"):
            s = s[3:].strip()
            marked = True
        if not s or " " in s:
            continue
        if not ARABIC_ALL.match(s):
            continue
        plain = strip_diac(s)
        if not (2 <= len(plain) <= 12):
            continue
        if any(plain.startswith(w) for w in CHAPTER_WORDS):
            continue
        headers.append({
            "line": i + 1,
            "header": s,
            "plain": plain,
            "norm": norm(s),
            "marked": marked,
        })
    return lines, headers


def entry_body(lines, idx, headers, max_lines=400):
    """Return the text of entry idx (0-based into headers) up to the next header."""
    start = headers[idx]["line"]          # 1-based line of header
    if idx + 1 < len(headers):
        end = headers[idx + 1]["line"] - 1
    else:
        end = len(lines)
    end = min(end, start + max_lines)
    body = "\n".join(lines[start:end])    # lines after header (start is 0-based next)
    return body.strip()


def main():
    lines, headers = extract_headers()
    print("lines=%d headers=%d marked=%d" % (
        len(lines), len(headers), sum(1 for h in headers if h["marked"])))
    from collections import Counter
    lc = Counter(len(h["plain"]) for h in headers)
    print("header plain-length histogram:", sorted(lc.items()))
    out = "/workspace/raghib_audit/mufradat_index.json"
    json.dump({"headers": headers, "n_lines": len(lines)}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False)
    print("wrote", out)
    # probe a few roots
    idx = {}
    for i, h in enumerate(headers):
        idx.setdefault(h["norm"], []).append(i)
    for probe in ["بغت", "قول", "قال", "علم", "كتب", "ضرب", "رحم", "اله", "الهه"]:
        print("probe", probe, "->", [headers[i]["header"] + "@" + str(headers[i]["line"])
                                     for i in idx.get(norm(probe), [])])


if __name__ == "__main__":
    main()
