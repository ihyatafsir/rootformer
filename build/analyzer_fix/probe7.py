#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe7.py -- WHY does the analyzer return root هلل for أهل / بيي for أبي?"""
import json
import re
import sys
import collections

RELEASE = '/workspace/hf_v19_2_release'
sys.path.insert(0, RELEASE)
sys.path.insert(0, RELEASE + '/models')
import nrmp_vocab as nv
import morphemic_tokenizer_v12_arabic as MT

V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
v = V(RELEASE + '/data/rootformer_v12_arabic_blueprint.json')

for w in ('أهل', 'أبي', 'الأرض', 'أخذ', 'أقل', 'أشد', 'عدد', 'العدد', 'قائم', 'قائمة',
          'يستعمل', 'يستحيل', 'أمة', 'أكان', 'أول', 'الأول', 'أبو', 'أب'):
    print('%-10s greedy=%-42s validated=%s'
          % (w, v.base_tok.decompose_arabic_word(w),
             v.base_tok.decompose_arabic_word(w)))
    pid, rid, wid, sid = v.encode_word(w)
    print('           tuple=(%s,%s,%s,%s) dec=%r'
          % (v.id2prefix[pid], v.id2root[rid], v.id2wazn[wid], v.id2suffix[sid],
             v.decode_word(pid, rid, wid, sid)))

print()
print('--- how many ANALYZER_wrong_root have an initial hamza/alif? ---')
F = [json.loads(l) for l in open('/workspace/analyzer_fix/failures.jsonl', encoding='utf-8')]
import unicodedata
wr = [x for x in F if not x['r'].startswith('<')
      and x['wz'] not in ('<NONE>', '<UNK>', '<PAD>')]
print('real-root+wazn failures:', len(wr))
c = collections.Counter()
for x in wr:
    w = re.sub('[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]', '', x['w'])
    c['initial_alif_or_hamza' if w[:1] in 'اأإآٱ' else 'other'] += 1
print(c)
c2 = collections.Counter()
for x in wr:
    c2[x['wz']] += 1
print(c2.most_common(12))
c3 = collections.Counter()
for x in wr:
    w = re.sub('[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]', '', x['w'])
    if w[:1] in 'اأإآٱ':
        c3[x['r'][-1] == x['r'][-2]] += 1
print('initial-alif subgroup, doubled third radical:', c3)
