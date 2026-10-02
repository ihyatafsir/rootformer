#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalized (diacritic-stripped) phrase search over corpus files.
Usage: python3 _search_norm.py <pattern> <file...> [--ctx N] [--max M]
Pattern is a Python regex matched against stripped text; offsets map back to raw.
"""
import re, sys, os

DIAC = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]')
TATWEEL = '\u0640'
ALEF = str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ى':'ي','ؤ':'و','ئ':'ي','ة':'ه'})


def strip_text(t):
    # remove diacritics, keep length? no -> we build map
    out = []
    idx = []
    for i, ch in enumerate(t):
        if DIAC.match(ch):
            continue
        out.append(ch)
        idx.append(i)
    return ''.join(out), idx


def norm(s):
    return DIAC.sub('', s).translate(ALEF)


def main():
    args = [a for a in sys.argv[1:]]
    ctx = 130
    mx = 40
    files = []
    pat = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--ctx':
            ctx = int(args[i+1]); i += 2; continue
        if a == '--max':
            mx = int(args[i+1]); i += 2; continue
        if pat is None:
            pat = a
        else:
            files.append(a)
        i += 1
    rx = re.compile(norm(pat))
    total = 0
    for f in files:
        t = open(f, encoding='utf-8', errors='replace').read()
        s, idx = strip_text(t)
        s = s.translate(ALEF)  # length-preserving, matches norm(pat)
        hits = list(rx.finditer(s))
        if hits:
            print(f"\n########## {f}  ({len(hits)} hits)")
        for k, m in enumerate(hits[:mx]):
            a = max(0, m.start()-ctx); b = min(len(s), m.end()+ctx)
            # map to raw offsets
            ra = idx[a] if a < len(idx) else len(t)
            rb = (idx[b-1]+1) if b-1 < len(idx) and b > 0 else len(t)
            snippet = t[ra:rb].replace('\n', ' ⏎ ')
            print(f"--[{f}:{k}]-- {snippet}")
            total += 1
        if len(hits) > mx:
            print(f"   ...{len(hits)-mx} more in {f}")
    print(f"\nTOTAL SHOWN {total}")


if __name__ == '__main__':
    main()
