#!/usr/bin/env python3
"""Targeted diagnostics:
 (1) verify `PageVxxPyyy` page locators: does the marker exist, and does the quote
     actually sit near it?
 (2) for any quote that only reaches PARAPHRASE, print the character-level first
     divergence against the corpus window, so the difference is visible.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_citations_v3 as V  # noqa: E402

PAGE_RE = re.compile(r'PageV(\d+)P(\d+)')


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, a[max(0, i - 40):i + 60], b[max(0, i - 40):i + 60]
    return n, a[max(0, n - 40):], b[max(0, n - 40):]


def main():
    ext = sys.argv[1]
    files, by_base = V.load_corpus()
    data = json.load(open(ext, encoding='utf-8'))

    print("=" * 100)
    print("(1) PAGE-LOCATOR CHECK  (Sibawayh al-Kitab 'PageVxxPyyy')")
    print("=" * 100)
    n_ok = n_miss = n_far = 0
    for r in data['structured']:
        loc = (r.get('book') or '') + ' ' + (r.get('src_raw') or '')
        pm = PAGE_RE.search(loc)
        if not pm:
            continue
        marker = pm.group(0)
        base = os.path.basename(r.get('src_file') or '') or 'Sibawayh_Al_Kitab.txt'
        f = next((x for x in files if x['base'] == base), None)
        if f is None:
            print(f"  NO FILE  {r['module']}:{r['module_line']} {marker}")
            continue
        raw = f['raw']
        ln = raw.find(marker)
        if ln < 0:
            n_miss += 1
            print(f"  MARKER ABSENT  {r['module']}:{r['module_line']}  {marker}  "
                  f"book={r.get('book')}")
            continue
        mline = raw.count('\n', 0, ln) + 1
        h = V.match_in(f, r['arabic'])
        if h is None:
            n_far += 1
            print(f"  QUOTE NOT FOUND in {base}; marker {marker} at line {mline}")
            continue
        delta = h['line'] - mline
        if 0 <= delta <= 300:
            n_ok += 1
            print(f"  OK    {r['module']}:{r['module_line']} {marker}@{mline} "
                  f"quote@{h['line']} (+{delta})  [{h['level']}]")
        else:
            n_far += 1
            print(f"  FAR   {r['module']}:{r['module_line']} {marker}@{mline} "
                  f"quote@{h['line']} ({delta:+d})  [{h['level']}]")
    print(f"  -> marker present & quote near: {n_ok}; marker absent: {n_miss}; quote far/not found: {n_far}")

    print()
    print("=" * 100)
    print("(2) PARAPHRASE-ONLY QUOTES: first divergence vs the corpus")
    print("=" * 100)
    for r in data['structured']:
        base = os.path.basename(r.get('src_file') or '')
        prefer = {base} if base in by_base else set()
        ph, oh = V.find(r['arabic'], files, prefer)
        if ph is None or ph['cls'] != 'PARAPHRASE':
            continue
        f = next(x for x in files if x['path'] == ph['file'])
        q = V.lvl_orth(r['arabic'])
        pos = f['m3'].find(q.split()[0]) if q.split() else -1
        win = f['m3'][max(0, pos - 50):pos + len(q) + 100] if pos >= 0 else ''
        i, A, B = first_divergence(q, win)
        print(f"\n{r['module']}:{r['module_line']}  file={os.path.basename(ph['file'])} "
              f"coverage={ph['score']}")
        print(f"  quote(norm): {q[:180]}")
        print(f"  corpus(win): {win[:180]}")
        print(f"  diverge@{i}: module={A[:70]!r}")
        print(f"              corpus={B[:70]!r}")


if __name__ == '__main__':
    main()
