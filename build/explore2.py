#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replica of _decompose_greedy Layer 1, used to stratify the E1 (<PARTICLE>) bucket."""
import json, re, collections

R = json.load(open('taxonomy_out/resources.json', encoding='utf-8'))
ROOTS = set(R['roots']); TOKENS = set(R['tokens']); CLOSED = set(R['closed_particles'])
PROC = R['proclitics']; ENC = R['enclitics']
TPL = [(re.compile(p), w, sp) for p, w, sp in R['awzan_templates']]
QTPL = [(re.compile(p), w) for p, w in R['quad_templates']]
H = str.maketrans({'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', 'ى': 'ي', 'ة': 'ه'})
norm = lambda s: s.translate(H)
NROOT = {norm(x) for x in ROOTS}


def cand_roots(stem, grps, special):
    out = []
    if special == 'hollow_w' and len(grps) >= 2:
        out += [grps[0] + 'و' + grps[1], grps[0] + 'ي' + grps[1]]
    elif special == 'quad' and len(grps) >= 1:
        out.append(grps[0])
    elif special == 'fawa_il' and len(grps) >= 3:
        out.append(grps[0] + grps[1] + grps[2])
    elif special == 'faalil' and len(grps) >= 4:
        out.append(grps[0] + grps[1] + grps[2] + grps[3])
        if grps[1] == grps[2]:
            out.append(grps[0] + grps[2] + grps[3])
    elif grps:
        out.append(''.join(grps))
    return out


def analysis(C):
    """Return (kind, detail). kind in {exact_root, template_ok, template_missing,
    template_norm, nothing}."""
    article = [p for p in ['فال', 'وال', 'كال', 'بال', 'لل', 'ال'] if C.startswith(p)]
    single = [p for p in ['و', 'ف', 'ب', 'ل', 'ك'] if C.startswith(p)]
    if C.startswith('و') and len(C) >= 4 and not C.startswith('وال'):
        single = [p for p in single if p != 'و'] + ['و']
    prefs = article + [''] + single
    cs = sorted([s for s in ENC if s and C.endswith(s)], key=len, reverse=True)
    if C.endswith('ا') and len(C) >= 4 and not C.endswith(('ها', 'وا', 'يا')):
        if 'ا' not in cs:
            cs = ['ا'] + [s for s in cs if s != 'ا']
    sufs = [''] + cs
    best = ('nothing', None)
    for p in prefs:
        for s in sufs:
            if p and s and len(C) < len(p) + len(s) + 2:
                continue
            stem = C[len(p):len(C) - len(s)] if s else C[len(p):]
            if not stem:
                continue
            if len(stem) == 3 and stem in ROOTS:
                return ('exact_root', stem)
            if len(stem) == 4 and stem in ROOTS:
                return ('exact_root4', stem)
            if len(stem) == 3 and stem[1] == 'ا':
                d = stem[0] + stem[2] + stem[2]
                if d in ROOTS:
                    return ('exact_root', d)
            if len(stem) == 2:
                d = stem + stem[-1]
                if d in ROOTS:
                    return ('exact_root', d)
            for pat, wazn, sp in TPL:
                m = pat.match(stem)
                if not m:
                    continue
                for cand in cand_roots(stem, m.groups(), sp):
                    if cand in ROOTS:
                        return ('template_ok', (cand, wazn, stem))
                    if norm(cand) in NROOT:
                        best = ('template_norm', (cand, wazn, stem))
                    elif best[0] == 'nothing':
                        best = ('template_missing', (cand, wazn, stem))
    return best


if __name__ == '__main__':
    rows = [json.loads(l) for l in open('taxonomy_out/failures_m1.jsonl', encoding='utf-8')]
    E1 = [r for r in rows if r['cause'] == 'E1_particle_echo_no_analysis']
    c = collections.Counter(); o = collections.Counter()
    ex = collections.defaultdict(list)
    detail = collections.defaultdict(collections.Counter)
    for r in E1:
        k, d = analysis(r['w'])
        c[k] += 1; o[k] += r['count']
        if d is not None and isinstance(d, tuple) and len(d) == 2:
            detail[k][d[1] if k == 'template_missing' or k == 'template_norm' else ''] += 1
        if len(ex[k]) < 8:
            ex[k].append((r['w'], r['count'], d))
    for k, n in c.most_common():
        print('%-20s t=%-7d o=%-8d  %s' % (k, n, o[k], ex[k][:5]))
    print()
    for k in ('template_missing', 'template_norm'):
        print(k, 'top wazn:', detail[k].most_common(12))
    # details of template_missing roots
    print()
    for k in ('template_missing', 'template_norm'):
        roots = collections.Counter()
        for r in E1:
            kk, d = analysis(r['w'])
            if kk == k and d: roots[d[0]] += r['count']
        print(k, 'top candidate roots:', roots.most_common(15))
