#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local exploration: re-stratify the big buckets from taxonomy_out/failures_m1.jsonl."""
import json, collections, re, sys

R = json.load(open('taxonomy_out/resources.json', encoding='utf-8'))
ROOTS = set(R['roots']); TOKENS = set(R['tokens']); CLOSED = set(R['closed_particles'])
COMPOUND = set(R['compound_particles']); PROC = R['proclitics']; ENC = R['enclitics']
PREFIXES = R['prefixes']; SUFFIXES = R['suffixes']
HAMZA_NORM = str.maketrans({'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', 'ى': 'ي', 'ة': 'ه', 'ؤ': 'و', 'ئ': 'ي', 'ء': ''})
def norm(s): return s.translate(HAMZA_NORM)
NCLOSED = {norm(x) for x in CLOSED}; NTOK = {norm(x) for x in TOKENS}; NROOT = {norm(x) for x in ROOTS}

rows = [json.loads(l) for l in open('taxonomy_out/failures_m1.jsonl', encoding='utf-8')]
print('rows', len(rows), 'occ', sum(r['count'] for r in rows))


def longest_root_sub(C):
    best = ''
    for i in range(len(C)):
        for j in range(i + 3, min(len(C), i + 6) + 1):
            s = C[i:j]
            if s in ROOTS and len(s) > len(best):
                best = s
    return best


def clitic_closed(C):
    for pre in [''] + PROC:
        if pre and not C.startswith(pre):
            continue
        rest = C[len(pre):]
        if len(rest) >= 2 and (rest in CLOSED or norm(rest) in NCLOSED):
            return pre or None, rest
    return None, None


def clitic_token(C):
    for pre in [''] + PROC:
        if pre and not C.startswith(pre):
            continue
        rest = C[len(pre):]
        if len(rest) >= 2 and (rest in TOKENS or norm(rest) in NTOK):
            return pre or None, rest
    return None, None


def clitic_root_enclitic(C):
    for pre in [''] + PROC:
        if pre and not C.startswith(pre):
            continue
        rest = C[len(pre):]
        for suf in [''] + ENC:
            if suf and not rest.endswith(suf):
                continue
            stem = rest[:len(rest) - len(suf)] if suf else rest
            if len(stem) >= 2 and (stem in ROOTS or norm(stem) in NROOT):
                return pre or None, stem, suf or None
    return None, None, None


def e1_stratum(C):
    if norm(C) in NCLOSED:
        return 'closed_particle_orthographic_variant'
    if clitic_closed(C)[0] is not None or C in CLOSED:
        return 'clitic_plus_closed_particle'
    if C in COMPOUND or norm(C) in {norm(x) for x in COMPOUND}:
        return 'compound_particle_not_resolved'
    if clitic_token(C)[1]:
        return 'clitic_plus_known_token'
    if len(C) <= 2:
        return 'short_or_single_char'
    if C.startswith('ال') and (C[2:] in ROOTS or norm(C[2:]) in NROOT):
        return 'definite_article_plus_bare_root'
    cr = clitic_root_enclitic(C)
    if cr[1]:
        return 'clitic_root_enclitic_exact_match'
    if C.startswith('ال') and longest_root_sub(C[2:]):
        return 'definite_article_plus_root_inside'
    if longest_root_sub(C):
        return 'lexical_root_inside_no_exact_split'
    return 'no_root_evidence_oov'


def e2_stratum(C):
    if C in CLOSED or C in COMPOUND:
        return 'closed_particle_without_P_tag'
    if norm(C) in NCLOSED:
        return 'closed_particle_orthographic_variant'
    if clitic_closed(C)[0] is not None:
        return 'clitic_plus_closed_particle'
    if len(C) == 1:
        return 'single_char'
    if C in TOKENS:
        return 'vocab_token_not_a_root'
    if clitic_token(C)[1]:
        return 'clitic_plus_known_token'
    return 'oov_unanalysable'


def show(name, sel, strat):
    c = collections.Counter(); o = collections.Counter()
    ex = collections.defaultdict(list)
    for r in rows:
        if not sel(r): continue
        s = strat(r)
        c[s] += 1; o[s] += r['count']
        if len(ex[s]) < 5: ex[s].append((r['w'], r['count'], r['dec']))
    ctx = sum(c.values()); cot = sum(o.values())
    print('#' * 108); print('%s: types=%d occ=%d' % (name, ctx, cot))
    for s, n in c.most_common():
        print('  %-42s t=%-6d o=%-7d (%.1f%%/%.1f%%)  %s' % (
            s, n, o[s], 100.0 * n / ctx, 100.0 * o[s] / cot,
            ', '.join('%s(%d)->%s' % e for e in ex[s][:4])))


show('E1 <PARTICLE>', lambda r: r['cause'] == 'E1_particle_echo_no_analysis', lambda r: e1_stratum(r['w']))
show('E2 <UNK>', lambda r: r['cause'] == 'E2_unk_echo_unknown_root', lambda r: e2_stratum(r['w']))
