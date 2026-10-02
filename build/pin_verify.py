#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pin_verify.py -- prove a release snapshot reproduces the mode-1 dump exactly."""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('--release', required=True)
ap.add_argument('--dump', default='/workspace/du_m1.json')
ap.add_argument('--sample', default='/workspace/eval_heldout_sample.json')
args = ap.parse_args()

os.environ.setdefault('ROOTFORMER_VALIDATED_SEG', '1')
R = args.release
for p in (R, R + '/models'):
    if p not in sys.path:
        sys.path.insert(0, p)

import nrmp_vocab as nv
import validated_segmentation as vs
cls = next(c for k, c in vars(nv).items() if isinstance(c, type) and 'MorphemicVocab' in k)
v = cls(R + '/data/rootformer_v12_arabic_blueprint.json')
print('[*] nrmp_vocab=%s' % nv.__file__)
print('[*] validated_segmentation=%s mode=%s' % (vs.__file__, vs.mode()))

du = json.load(open(args.dump, encoding='utf-8'))['words']
sample = json.load(open(args.sample, encoding='utf-8'))
cnt = {}
for f in sample['files']:
    for w, n in f['counts'].items():
        cnt[w] = cnt.get(w, 0) + n

t0 = time.time()
mism = 0
false_pos = 0
false_neg = 0
rt_occ = 0
tot_occ = 0
ex = []
for w, rec in du.items():
    p, r, wz, s = v.encode_word(w)
    dec = v.decode_word(p, r, wz, s)
    ok = (dec == w)
    if ok != bool(rec[0]):
        mism += 1
        if ok:
            false_pos += 1
        else:
            false_neg += 1
        if len(ex) < 12:
            ex.append((w, rec[0], ok, dec, rec[3]))
    if ok:
        rt_occ += cnt[w]
    tot_occ += cnt[w]
print('[*] words=%d mismatches=%d (dump-fail-but-live-pass=%d, dump-pass-but-live-fail=%d) %.0fs'
      % (len(du), mism, false_pos, false_neg, time.time() - t0))
print('[*] occurrence round-trip = %.4f%%  (dump: 73.8610%%)' % (100.0 * rt_occ / tot_occ))
print('[*] distinct-form round-trip = %.4f%% (dump: 55.9611%%)'
      % (100.0 * (len(du) - sum(1 for r in du.values() if not r[0])) / len(du)))
for e in ex:
    print('    ', e)
