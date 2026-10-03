#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""divine_scan.py -- the exact divine-name surfaces that actually occur in the sample.

Printed with their occurrence counts, plus the clitic grid the design's own prefix/suffix
slots can build on top of them.  This is the hard-coded set the passthrough uses.
"""
import collections
import json
import re
import sys

SAMPLE = '/workspace/eval_heldout_sample.json'
DIAC_RE = re.compile('[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
sd = lambda s: DIAC_RE.sub('', s or '')

d = json.load(open(SAMPLE, encoding='utf-8'))
forms = collections.Counter()
for f in d['files']:
    for w, n in f['counts'].items():
        forms[w] += n
print('distinct forms in sample:', len(forms))

# any surface whose CONSONANT skeleton carries the lam-lam-heh of the divine name
hits = collections.Counter()
for w, n in forms.items():
    s = sd(w)
    if 'لل' in s and s.endswith('ه'):
        hits[w] += n
    elif 'الله' in s or 'ٱلل' in w or 'للّٰ' in w:
        hits[w] += n
print('\n--- surfaces containing the divine lam-lam-heh ---')
for w, n in sorted(hits.items(), key=lambda kv: -kv[1]):
    print('  %-16s n=%d  skeleton=%s' % (w, n, sd(w)))

# the exact codepoints of the common spellings, for byte-exactness
print('\n--- codepoints of the canonical spellings ---')
for w in ('الله', 'ٱللّٰه', 'ٱلله', 'ألله', 'اللهم', 'ٱللّٰهم', 'تالله'):
    print('  %-10s %s' % (w, ' '.join('U+%04X' % ord(c) for c in w)))
