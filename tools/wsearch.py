#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diacritic-insensitive Arabic corpus search.

Usage:
  wsearch.py PATTERN FILE [FILE...] [--ctx N] [--max M] [--files-only]
  wsearch.py --list PATTERN DIR...

Strips Arabic diacritics/tatweel from both the corpus lines and the pattern,
then searches. Prints original (undiacritized-copy) line text with line numbers.
"""
import sys, os, re, glob

DIAC = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]')

def norm(s):
    s = DIAC.sub('', s)
    s = s.replace('\u0622', '\u0627')  # alef madda -> alef
    s = s.replace('\u0623', '\u0627')  # alef hamza above -> alef
    s = s.replace('\u0625', '\u0627')  # alef hamza below -> alef
    s = s.replace('\u0649', '\u064A')  # alef maqsura -> ya
    s = s.replace('\u0629', '\u0647')  # ta marbuta -> ha
    s = s.replace('\u0624', '\u0648')  # waw hamza -> waw
    s = s.replace('\u0626', '\u064A')  # ya hamza -> ya
    s = re.sub(r'\s+', ' ', s)
    return s

def main():
    args = sys.argv[1:]
    ctx = 0
    maxhits = 40
    files_only = False
    out = []
    i = 0
    pat = None
    files = []
    while i < len(args):
        a = args[i]
        if a == '--ctx':
            ctx = int(args[i+1]); i += 2
        elif a == '--max':
            maxhits = int(args[i+1]); i += 2
        elif a == '--files-only':
            files_only = True; i += 1
        elif pat is None:
            pat = a; i += 1
        else:
            files.append(a); i += 1
    exp = []
    for f in files:
        if os.path.isdir(f):
            exp.extend(sorted(glob.glob(os.path.join(f, '*.txt'))))
        else:
            exp.extend(sorted(glob.glob(f)) or [f])
    npat = norm(pat)
    total = 0
    for fp in exp:
        try:
            with open(fp, encoding='utf-8', errors='replace') as fh:
                lines = fh.read().split('\n')
        except Exception as e:
            print(f'!! {fp}: {e}', file=sys.stderr); continue
        normlines = [norm(l) for l in lines]
        hits = [j for j, l in enumerate(normlines) if npat in l]
        if not hits:
            continue
        if files_only:
            print(f'{fp}\t{len(hits)}')
            total += len(hits)
            continue
        print(f'\n########## {fp}  ({len(hits)} hits) ##########')
        for h in hits[:maxhits]:
            total += 1
            lo = max(0, h-ctx); hi = min(len(lines), h+ctx+1)
            for j in range(lo, hi):
                mark = '>>' if j == h else '  '
                print(f'{mark} {fp}:{j+1}: {lines[j]}')
            if ctx:
                print('   ---')
    print(f'\n=== total shown: {total} ===', file=sys.stderr)

if __name__ == '__main__':
    main()
