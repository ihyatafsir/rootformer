#!/usr/bin/env python3
"""split_fixed.py -- which taxonomy class does each FIXED type belong to?"""
import json, collections
base = [json.loads(l) for l in open('/workspace/analyzer_fix/base.jsonl', encoding='utf-8')]
on = [json.loads(l) for l in open('/workspace/analyzer_fix/on2.jsonl', encoding='utf-8')]
tax = {r['w']: r for r in (json.loads(l) for l in open('/workspace/analyzer_fix/taxonomy_rows.jsonl', encoding='utf-8'))} if False else {}
fixed = [i for i in range(len(base)) if (not base[i]['ok']) and on[i]['ok']]
print('FIXED total', len(fixed))
c = collections.Counter()
for i in fixed:
    a, b = base[i], on[i]
    if b['r'].startswith('<DIV:'):
        c['DIVINE_passthrough'] += 1
    elif a['r'] in ('<PARTICLE>', '<UNK>'):
        c['CONTROL (unexpected)'] += 1
    else:
        c['TASRIF_delegation'] += 1
for k, v in c.most_common():
    print('  %-24s %d' % (k, v))
# occurrences
for k in c:
    n = sum(base[i]['n'] for i in fixed
            if (on[i]['r'].startswith('<DIV:') and k == 'DIVINE_passthrough')
            or (not on[i]['r'].startswith('<DIV:') and k == 'TASRIF_delegation'))
    if n:
        print('  occ %-20s %d' % (k, n))
# how many changed-but-still-failing, by their base root class
neut = [i for i in range(len(base)) if base[i] != on[i] and not on[i]['ok']]
print('neutral changed, still failing:', len(neut))
print('  their base root class:',
      collections.Counter('CONTROL' if base[i]['r'] in ('<PARTICLE>', '<UNK>')
                          else ('REAL' if not base[i]['r'].startswith('<') else 'PTAG')
                          for i in neut))
# residual failing counts with fix on
res = collections.Counter()
for i, x in enumerate(on):
    if x['ok']:
        continue
    res['CONTROL' if x['r'] in ('<PARTICLE>', '<UNK>')
        else ('PTAG' if x['r'].startswith('<') else 'REAL')] += 1
print('residual failing under fix:', dict(res), 'total', sum(res.values()))
