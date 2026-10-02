#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""corpus_read.py -- read real Arabic sentences from the three prompt-source corpora.

Deliberately mirrors scratch_lm.sentences_by_file(): same Arabic-regex gate, same
`SPLIT` punctuation split, same "join the ARABIC tokens with single spaces" rule (so the
sentence we hand to the model is exactly the string the project's own encoder sees), and
same minimum-length gates.  Differences: the roots are the DIRECTORIES the task named, and
we keep the source path alongside every sentence for provenance.
"""
import glob
import json
import os
import re
from pathlib import Path

AR = re.compile(r'[\u0600-\u06FF]')
SPLIT = re.compile(r'[\n.!?؟؛:]+')

CORPORA = [
    '/workspace/scholastic_sunni_sanitized',
    '/workspace/scholastic_falsafa_sanitized',
    '/workspace/scholastic_masters',
]


def _chunks_of(p):
    if p.endswith('.jsonl'):
        out = []
        for line in open(p, encoding='utf-8', errors='ignore'):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            for k in ('arabic', 'text', 'ar'):
                if isinstance(d.get(k), str):
                    out.append(d[k])
                    break
        return out
    if p.endswith('.json'):
        d = json.load(open(p, encoding='utf-8', errors='ignore'))
        out = []

        def rec(x):
            if isinstance(x, str):
                if AR.search(x):
                    out.append(x)
            elif isinstance(x, dict):
                hit = False
                for k in ('arabic', 'text', 'ar', 'content'):
                    if isinstance(x.get(k), str):
                        out.append(x[k]); hit = True; break
                if not hit:
                    for v in x.values():
                        rec(v)
            elif isinstance(x, list):
                for v in x[:5000]:
                    rec(v)
        rec(d)
        return out
    return [Path(p).read_text(encoding='utf-8', errors='ignore')]


def sentences(limit_per_corpus=None):
    """Yield (corpus, relpath, sentence) for real Arabic sentences, corpus by corpus."""
    for root in CORPORA:
        if not os.path.isdir(root):
            print(f'[corpus_read] MISSING {root}', flush=True)
            continue
        n = 0
        done = False
        for ext in ('*.txt', '*.jsonl', '*.json'):
            if done:
                break
            for p in sorted(glob.glob(os.path.join(root, '**', ext), recursive=True)):
                if os.path.getsize(p) > 400 * 1024 * 1024:
                    continue
                try:
                    chunks = _chunks_of(p)
                except Exception as exc:
                    print(f'[corpus_read] skip {p!r}: {exc!r}', flush=True)
                    continue
                for ch in chunks:
                    for s in SPLIT.split(ch):
                        s = s.strip()
                        if len(s) < 12 or not AR.search(s):
                            continue
                        ws = [w for w in s.split() if AR.search(w)]
                        if len(ws) < 6:          # need a real context plus a target
                            continue
                        yield root, os.path.relpath(p, root), ' '.join(ws)
                        n += 1
                        if limit_per_corpus is not None and n >= limit_per_corpus:
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
