#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostics: is the extracted Mufradat header list complete?

al-Mufradat is arranged by root in Arabic alphabetical order, in 28 chapters
('||| كتاب X').  If header extraction misses entries, we should see gaps or
order violations.  We also look for header-shaped lines containing spaces
(possible multi-word entries) and for entries whose body looks like it starts
a new root without a header line.
"""
import json
import re
import sys
sys.path.insert(0, "/workspace/raghib_audit")
from extract_mufradat_index import extract_headers, strip_diac, norm, SRC

ALPHA = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"


def rootkey(h):
    """Approximate sort key: index of letters, so we can detect order breaks."""
    p = strip_diac(h).replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    return tuple(ALPHA.find(c) if c in ALPHA else 99 for c in p)


def main():
    lines, headers = extract_headers()
    # chapter lines
    chapters = [(i + 1, l.strip()) for i, l in enumerate(lines) if l.strip().startswith("||| كتاب")]
    print("chapters:", len(chapters))
    for c in chapters:
        print("  ", c[1], "@", c[0])

    # Multi-word Arabic-only short lines -> possible missed headers
    print("\n--- Arabic-only lines containing a space, length<=16, not ~~/||| ---")
    n = 0
    ARABIC_SP = re.compile(r"^[\u0621-\u064A\u0670-\u06D3\u064B-\u0652\u0640 ]+$")
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s or s.startswith("~~") or s.startswith("|||"):
            continue
        if ARABIC_SP.match(s) and " " in s and len(s) <= 16:
            print("  ", i + 1, repr(s))
            n += 1
            if n > 60:
                print("   ...")
                break

    # ordering violations between consecutive headers, per chapter
    print("\n--- order violations (consecutive headers in same chapter) ---")
    viol = 0
    for a, b in zip(headers, headers[1:]):
        ka, kb = rootkey(a["header"]), rootkey(b["header"])
        if ka > kb:
            viol += 1
            if viol <= 40:
                print("   %s@%d  ->  %s@%d" % (a["header"], a["line"], b["header"], b["line"]))
    print("total order violations:", viol, "of", len(headers) - 1, "adjacent pairs")

    # header line-length: any header preceded by a line that is not blank/~~?
    print("\n--- headers NOT preceded by a blank line (possible merged/missed) ---")
    c = 0
    for h in headers:
        ln = h["line"] - 1  # 1-based line before header
        prev = lines[ln - 2].strip() if ln - 2 >= 0 else ""
        if prev and not prev.startswith("~~"):
            c += 1
            if c <= 25:
                print("   line %d prev=%r  hdr=%r" % (h["line"], prev[:70], h["header"]))
    print("count:", c)


if __name__ == "__main__":
    main()
