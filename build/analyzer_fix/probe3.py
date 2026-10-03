#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe3.py -- WHAT IS ACTUALLY IN THE <PARTICLE> BUCKET?

Claim under test (from the brief): "<PARTICLE>/<UNK> targets ... unrepresentable BY DESIGN".
Test: for every failing type whose root slot is a control token, does the DESIGN'S OWN
closed-class route (_closed_class_split + a <P:..> id) accept the surface?

  * accepted  -> the control tag is a DESIGN artefact for closed-class material -> by design
  * declined  -> the design's own inventory says "not closed-class" -> ANALYZER failure

Also: the diagnostic case مكتوب -> كوب.
"""
import collections
import json
import sys

RELEASE = '/workspace/hf_v19_2_release'
BLUEPRINT = RELEASE + '/data/rootformer_v12_arabic_blueprint.json'
sys.path.insert(0, RELEASE)
sys.path.insert(0, RELEASE + '/models')

import nrmp_vocab as nv

V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
v = V(BLUEPRINT)
print('[*] roots=%d awzan=%d prefixes=%d suffixes=%d'
      % (v.num_roots, v.num_awzan, v.num_prefixes, v.num_suffixes))
print('[*] closed inventory size=%d  <P:..> ids=%d'
      % (len(v._closed_inventory), len(v.appended_particle_roots) + 66))

F = [json.loads(l) for l in open('/workspace/analyzer_fix/failures.jsonl', encoding='utf-8')]

buckets = collections.Counter()
examples = collections.defaultdict(list)
for x in F:
    r = x['r']
    if r == '<PARTICLE>':
        closed = v._closed_class_split(x['w'])
        if closed is not None and ('<P:%s>' % closed[1]) in v.root2id:
            buckets['CONTROL_accepted_by_design'] += 1
            examples['CONTROL_accepted_by_design'].append(x)
        else:
            buckets['NO_ANALYSIS_real_word'] += 1
            examples['NO_ANALYSIS_real_word'].append(x)
    elif r == '<UNK>':
        buckets['UNK_control'] += 1
        examples['UNK_control'].append(x)
    elif r.startswith('<P:'):
        buckets['PTAG_misroute'] += 1
        examples['PTAG_misroute'].append(x)
    elif r.startswith('<'):
        buckets['SPECIAL'] += 1
    else:
        buckets['REAL_root'] += 1
        examples['REAL_root'].append(x)

print('\n--- bucket census (types) ---')
tot = sum(buckets.values())
for k, c in buckets.most_common():
    print('  %-32s %7d  %6.2f%%' % (k, c, 100.0 * c / tot))

for k in ('NO_ANALYSIS_real_word', 'CONTROL_accepted_by_design', 'PTAG_misroute', 'UNK_control'):
    sub = sorted(examples[k], key=lambda z: -z['n'])[:8]
    print('\n--- %s: top by occurrences ---' % k)
    for x in sub:
        pid, rid, wid, sid = v.encode_word(x['w'])
        closed = v._closed_class_split(x['w'])
        print('  %-14s n=%-7d tuple=(%s,%s,%s,%s) closed=%s dec=%r'
              % (x['w'], x['n'], v.id2prefix[pid], v.id2root[rid], v.id2wazn[wid],
                 v.id2suffix[sid], closed, x['dec']))

print('\n--- DIAGNOSTIC CASE مكتوب ---')
for w in ('مكتوب', 'كتب', 'مكتوبة', 'المكتوب', 'مكتوبا'):
    pid, rid, wid, sid = v.encode_word(w)
    dec = v.decode_word(pid, rid, wid, sid)
    print('  %-12s -> p=%-6s r=%-8s wz=%-12s s=%-6s dec=%-12s ok=%s'
          % (w, v.id2prefix[pid], v.id2root[rid], v.id2wazn[wid], v.id2suffix[sid],
             dec, dec == w))
    print('        base_tok.decompose_arabic_word -> %r'
          % (v.base_tok.decompose_arabic_word(w),))

print('\n--- DIAGNOSTIC CASE قيل (passive of قال) ---')
for w in ('قيل', 'قال', 'بيع', 'خوف', 'مصون'):
    pid, rid, wid, sid = v.encode_word(w)
    dec = v.decode_word(pid, rid, wid, sid)
    print('  %-12s -> p=%-6s r=%-8s wz=%-12s s=%-6s dec=%-12s ok=%s'
          % (w, v.id2prefix[pid], v.id2root[rid], v.id2wazn[wid], v.id2suffix[sid],
             dec, dec == w))
    print('        base_tok -> %r' % (v.base_tok.decompose_arabic_word(w),))
