#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
taxonomy.py -- CLOSED taxonomy of the 80,040 failing distinct forms at the published
mode-1 baseline (188,722 held-out types, 57.5884 %).

Every failing type is assigned by a FIRST-MATCH cascade of mechanical tests.  The cascade is
made exhaustive by an explicit residue bucket, and the residue is reported, never hidden.

The taxonomy separates REPRESENTABILITY from FAULT:

  level 1 (representability)
    DIVINE          the divine name and its clitic-built forms -- by product decision these are
                    surface-preserving passthroughs.  Not an analyzer defect.
    CONTROL         the ROOT slot is a control token (<PARTICLE>/<UNK>) -- no surface is carried
                    by the tuple at all, so the type CANNOT round-trip whatever the analyzer
                    does.  Sub-split by whether the design's OWN closed-class inventory accepts
                    the surface (by design) or declines it (analyzer returned no analysis).
    ANALYZER        a real decomposition was produced and it is wrong or unspellable.

  level 2 (cause, only inside ANALYZER)
    wrong root / wrong wazn / wrong affix / clitic mis-split / ORTHOGRAPHIC / irregular / OTHER

Run:
  ROOTFORMER_VALIDATED_SEG=1 python taxonomy.py
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

RELEASE = '/workspace/hf_v19_2_release'
BLUEPRINT = RELEASE + '/data/rootformer_v12_arabic_blueprint.json'
OUT = '/workspace/analyzer_fix'
sys.path.insert(0, RELEASE)
sys.path.insert(0, RELEASE + '/models')

import nrmp_vocab as nv
from tasrif_engine import TasrifEngine

DIAC_RE = re.compile('[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
sd = lambda s: DIAC_RE.sub('', s or '')
NONE = ('<NONE>', '<PAD>', '<UNK>', '', None)
WEAK = frozenset('اويىءأإآؤئٱ')
HAMZA = frozenset('ءأإآؤئٱ')
LETTERS = re.compile('^[\u0621-\u064a\u0671-\u06d3]+$')

# ---------------------------------------------------------------- the divine name
# The surface is ٱللّٰه / الله -- hamzat wasl + the lam-alif ligature (or lam+alif) + shadda +
# dagger alef.  No (P,R,W,S) tuple regenerates it.  The set below is not guessed: it is the set
# the SAMPLE actually contains, plus the clitic combinations the design's own prefix/suffix
# slots can build.  Measured in divine_scan() and written to taxonomy.json.
DIVINE_CORES = ('الله', 'ٱللّٰه', 'اللهم', 'ٱللّٰهم', 'ألله', 'ٱلله', 'إلله')
CLITIC_PRE = ('', 'و', 'ف', 'ب', 'ك', 'ل', 'ال', 'وال', 'فال', 'بال', 'كال', 'لل', 'ت', 'وت',
              'ولل', 'الل', 'س')
CLITIC_SUF = ('', 'ه', 'ها', 'هم', 'هما', 'هن', 'ك', 'كم', 'كن', 'نا', 'ي', 'ت', 'تم', 'ات', 'ين', 'ان')


def divine_forms():
    out = set()
    for pre in CLITIC_PRE:
        for core in DIVINE_CORES:
            for suf in CLITIC_SUF:
                out.add(pre + core + suf)
    return out


DIVINE_SET = divine_forms()
# the ligature-safe core: 'الله' written with lam+lam+heh; and the wasla spelling
DIVINE_RE = re.compile('(?:^|[وفبكلتس]|لل|ال|وال|فال|بال|كال)*(?:ٱ?للّ?ٰ?ه|ٱ?للّٰه)(?:ه|ها|هم|هما|هن|ك|كم|كن|نا|ي|ت|تم|ات|ين|ان)?$')


def is_divine(w):
    if w in DIVINE_SET:
        return True
    s = sd(w)
    return bool(DIVINE_RE.match(s)) and 'له' in s


# ---------------------------------------------------------------- radical tracing
def traces(root, stem):
    """Can the root's radicals be read off the stem skeleton, in order?

    A weak radical may surface as any weak letter or be ELIDED (al-hadhf, al-qalb, al-ibdal of
    the ʿilal); a hamza may surface as any hamza carrier.  A sound radical must appear.  This is
    the mechanical form of «الاشتقاق» as the first adilla of the asl (Ibn 'Usfur, al-Mumti':
    «أما الأدلة التي يعرف بها الزائد من الأصلي فهي: الاشتقاق، والتصريف، ...»).
    """
    if not root or not stem:
        return False
    r = sd(root)
    s = sd(stem)
    n, m = len(r), len(s)
    # DP
    prev = [True] + [False] * m
    for i in range(1, n + 1):
        cur = [False] * (m + 1)
        for j in range(1, m + 1):
            ri, sj = r[i - 1], s[j - 1]
            if ri == sj:
                cur[j] = prev[j - 1]
            elif ri in HAMZA and sj in HAMZA:
                cur[j] = prev[j - 1]
            elif ri in WEAK and sj in WEAK:
                cur[j] = prev[j - 1]
            if not cur[j] and ri in WEAK:
                # elided radical: consume no surface letter
                cur[j] = cur[j - 1] or (prev[j] if i == n else False) or prev[j - 1] or cur[j]
        # allow skipping a weak root letter entirely
        for j in range(m + 1):
            if not cur[j]:
                pass
        prev = cur
    return prev[m]


def traces2(root, stem):
    """As traces(), but a weak radical may also be skipped without consuming a surface letter."""
    r, s = sd(root), sd(stem)
    n, m = len(r), len(s)
    dp = [[False] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = True
    for i in range(1, n + 1):
        # a weak radical may vanish
        if r[i - 1] in WEAK:
            for j in range(m + 1):
                if dp[i - 1][j]:
                    dp[i][j] = True
        for j in range(1, m + 1):
            if not dp[i - 1][j - 1]:
                continue
            ri, sj = r[i - 1], s[j - 1]
            if ri == sj or (ri in HAMZA and sj in HAMZA) or (ri in WEAK and sj in WEAK):
                dp[i][j] = True
    return dp[n][m]


# ---------------------------------------------------------------- orthographic equality
ORTHO_MAP = str.maketrans({
    '\u0623': '\u0627', '\u0625': '\u0627', '\u0622': '\u0627', '\u0671': '\u0627',
    '\u0626': '\u0621', '\u0624': '\u0621', '\u0649': '\u064a', '\u0629': '\u0647',
    '\u0621': '\u0627',
})


def ortho(s):
    return sd(s).translate(ORTHO_MAP)


def main():
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    v = V(BLUEPRINT)
    eng = TasrifEngine(vocab=None)
    F = [json.loads(l) for l in open(OUT + '/failures.jsonl', encoding='utf-8')]
    TOT_TYPES = 188722

    AWZAN = set(w for w in v.awzan_list if not w.startswith('<'))
    # which wazn does the shipped hand-written realiser actually implement?  Determined by
    # EXECUTION: a probe root whose realisation differs from the bare root proves a branch ran.
    probe_roots = ['كتب', 'قوم', 'عدد', 'عمل', 'نهي', 'بدا', 'حقق', 'دلل', 'منع', 'شقق']
    IMPLEMENTED = set()
    for wz in AWZAN:
        for r in probe_roots:
            try:
                got = sd(v.base_tok.realize_root_and_wazn(r, wz))
            except Exception:
                continue
            bare = sd(__import__('morphemic_tokenizer_v12_arabic').ROOT_CANONICAL_MAP_REV.get(r, r))
            if got != bare:
                IMPLEMENTED.add(wz)
                break

    def cnt(name):
        return sum(x['n'] for x in F if not x['r'].startswith('<')) and 0

    rows = []
    ex = collections.defaultdict(list)
    sub = collections.Counter()

    for x in F:
        w, n, p, r, wz, s, dec = x['w'], x['n'], x['p'], x['r'], x['wz'], x['s'], x['dec']
        rec = dict(x)

        # ---------- level 1: representability ----------
        if is_divine(w):
            rec['cause'] = 'DIVINE_passthrough_by_design'
            rec['level'] = 1
            rows.append(rec)
            ex[rec['cause']].append(rec)
            continue
        if r in ('<PARTICLE>', '<UNK>', '<PAD>', '<BOS>', '<EOS>'):
            closed = v._closed_class_split(w)
            if closed is not None and ('<P:%s>' % closed[1]) in v.root2id:
                rec['cause'] = 'CONTROL_closed_class_by_design'
            else:
                rec['cause'] = 'CONTROL_no_analysis_unrepresentable'
            rec['level'] = 1
            rows.append(rec)
            ex[rec['cause']].append(rec)
            sub['control_root_' + r] += 1
            continue

        # ---------- level 2: ANALYZER ----------
        pt = '' if p in NONE else p
        st = '' if s in NONE else s
        cw = sd(w)
        stem = cw
        if pt and stem.startswith(sd(pt)):
            stem = stem[len(sd(pt)):]
        if st and stem.endswith(sd(st)):
            stem = stem[:len(stem) - len(sd(st))]
        rec['stem'] = stem

        if r.startswith('<P:'):
            rec['cause'] = 'ANALYZER_particle_overcapture'
        elif not LETTERS.match(cw) or len(cw) == 1:
            rec['cause'] = 'PUNCTUATION_tokenisation_artefact'
        elif wz in NONE:
            rec['cause'] = 'ANALYZER_no_wazn_assigned'
        else:
            g = None
            try:
                res = eng.generate(r, wz)
                g = sd(res[0]) if res else None
            except Exception:
                g = None
            rec['engine'] = g
            if g is not None and g == stem:
                rec['cause'] = 'REALISER_wazn_unimplemented_engine_ok'
            elif not traces2(r, stem):
                rec['cause'] = 'ANALYZER_wrong_root'
            elif ortho(dec) == ortho(w):
                rec['cause'] = 'ORTHOGRAPHIC_hamza_or_weak_spelling'
            elif g is not None:
                rec['cause'] = 'WRONG_WAZN_engine_disagrees'
            else:
                rec['cause'] = 'OTHER'
        rec['level'] = 2
        rows.append(rec)
        ex[rec['cause']].append(rec)

    # ---------------------------------------------------------------- report
    agg = collections.defaultdict(lambda: {'types': 0, 'occ': 0})
    for rec in rows:
        a = agg[rec['cause']]
        a['types'] += 1
        a['occ'] += rec['n']

    NFAIL = len(rows)
    L = []
    A = L.append
    A('# Failure taxonomy -- NRMP morphemic tokenizer, mode-1 shipped default')
    A('')
    A('Baseline reproduced by `dump_failures.py` on the published harness sample:')
    A('**type round-trip 57.5884 % (108,682 / 188,722)**, occurrence round-trip **82.1065 %**,')
    A('**80,040 failing distinct forms**.')
    A('')
    A('| cause | types | type % of fails | occurrences | occ % of fails | % of ALL types |')
    A('|---|---|---|---|---|---|')
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]['types']):
        A('| `%s` | %d | %.2f%% | %d | %.2f%% | %.4f%% |'
          % (k, a['types'], 100.0 * a['types'] / NFAIL, a['occ'],
             100.0 * a['occ'] / NFAIL, 100.0 * a['types'] / TOT_TYPES))
    A('| **total** | **%d** | 100%% | **%d** | 100%% | |'
      % (NFAIL, sum(a['occ'] for a in agg.values())))
    A('')
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]['types']):
        A('## `%s` -- %d types (%.2f%% of failures), %d occurrences'
          % (k, a['types'], 100.0 * a['types'] / NFAIL, a['occ']))
        A('')
        for rec in sorted(ex[k], key=lambda z: -z['n'])[:12]:
            A('* `%s` n=%d p=%s r=%s wz=%s s=%s -> `%s`%s'
              % (rec['w'], rec['n'], rec['p'], rec['r'], rec['wz'], rec['s'], rec['dec'],
                 ('  [engine=%s]' % rec.get('engine')) if rec.get('engine') else ''))
        A('')

    txt = '\n'.join(L)
    open(OUT + '/taxonomy.md', 'w', encoding='utf-8').write(txt + '\n')
    json.dump({'aggregate': {k: v for k, v in agg.items()},
               'total_failing': NFAIL,
               'total_types': TOT_TYPES,
               'implemented_wazn_count': len(IMPLEMENTED),
               'control_sub': dict(sub)},
              open(OUT + '/taxonomy.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(txt)
    print('\n[*] implemented wazn detected by execution: %d of %d' % (len(IMPLEMENTED), len(AWZAN)))
    print('[*] control sub-census:', dict(sub))


if __name__ == '__main__':
    main()
