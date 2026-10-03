#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe1.py -- shape of the failing set, before any cascade is designed."""
import json, collections, sys

P = '/workspace/analyzer_fix/failures.jsonl'
F = [json.loads(l) for l in open(P, encoding='utf-8')]
print('fails', len(F))


def rclass(r):
    if r == '<PARTICLE>':
        return 'PARTICLE'
    if r == '<UNK>':
        return 'UNK'
    if r.startswith('<P:'):
        return 'P:' + r
    if r.startswith('<'):
        return 'SPECIAL:' + r
    return 'REAL'


c = collections.Counter(rclass(x['r']) for x in F)
print('--- root slot class ---')
for k, v in c.most_common(25):
    print('%-30s %7d' % (k, v))

print('--- decoded contains a tag ---')
print(sum(1 for x in F if '<' in x['dec']))

print('--- examples by root class ---')
for want in ('PARTICLE', 'UNK', 'REAL'):
    ex = [x for x in F if rclass(x['r']) == want][:6]
    for x in ex:
        print('%-8s %-14s p=%-8s r=%-10s wz=%-12s s=%-8s -> %s'
              % (want, x['w'], x['p'], x['r'], x['wz'], x['s'], x['dec']))

print('--- particle-token roots (<P:..>) count and top ---')
pc = collections.Counter(x['r'] for x in F if x['r'].startswith('<P:'))
print('distinct particle roots in failures:', len(pc), 'types:', sum(pc.values()))
for k, v in pc.most_common(15):
    print('   %-20s %d' % (k, v))

print('--- wazn values on REAL roots ---')
wc = collections.Counter(x['wz'] for x in F if rclass(x['r']) == 'REAL')
for k, v in wc.most_common(20):
    print('   %-14s %d' % (k, v))

print('--- prefix/suffix on REAL roots ---')
sp = collections.Counter((x['p'], x['s']) for x in F if rclass(x['r']) == 'REAL')
for k, v in sp.most_common(15):
    print('   %-28s %d' % (str(k), v))
