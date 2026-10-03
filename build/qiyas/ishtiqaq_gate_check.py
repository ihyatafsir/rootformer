#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ishtiqaq_gate_check.py -- did the FIXED native root-score-bias path actually TRAIN?

The pre-fix finding was decisive and damning: `stream_mix[1]` and `ishtiqaq_gamma` sat
BIT-EXACTLY at their `__init__` values (0.25) in all 24 layers, which is only possible if those
terms never produced a gradient.  The fix added zero-init `root_gate`/`pillar_gate` and a swap
helper, and arm RCA_NATIVE_A2 trained it.

This reads a saved arm checkpoint and reports, per layer:
  * `root_gate`         -- zero-init; non-zero only if the branch received gradient
  * `stream_mix`        -- [0.75, 0.25] at init; element 1 non-0.25 only if it got gradient
  * `ishtiqaq_gamma`    -- 0.25 at init; non-0.25 only if it got gradient
  * `pillar_gate`       -- zero-init
so "did it train" is answered by the numbers, not by inference.

CPU only, read-only.
usage: python ishtiqaq_gate_check.py <arm.pt|arm.pt.trunk.pt> [...]
"""
from __future__ import annotations

import collections
import sys

import torch

INIT = {'stream_mix': (0.75, 0.25), 'ishtiqaq_gamma': 0.25,
        'root_gate': 0.0, 'pillar_gate': 0.0}


def scan(path):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    st = blob.get('state', blob)
    print('=' * 78)
    print('[*] %s' % path)
    print('    keys=%d  step=%s  live=%s  ishtiqaq_layers=%s'
          % (len(st), blob.get('step'), blob.get('live'), blob.get('ishtiqaq_layers')))
    print('    ishtiqaq_root_source=%s' % blob.get('ishtiqaq_root_source'))

    per = collections.defaultdict(dict)
    for k, v in st.items():
        for name in INIT:
            if k.endswith('.' + name):
                li = None
                for part in k.split('.'):
                    if part.isdigit():
                        li = int(part)
                        break
                per[name][li] = v.detach().to(torch.float32).flatten().tolist()

    for name in ('root_gate', 'pillar_gate', 'stream_mix', 'ishtiqaq_gamma'):
        if name not in per:
            print('    %-16s : NOT PRESENT' % name)
            continue
        rows = per[name]
        vals = [tuple(r) if len(r) > 1 else r[0] for r in rows.values()]
        distinct = sorted(set(vals), key=lambda x: (str(type(x)), str(x)))
        print('    %-16s : layers=%d  init=%s  distinct=%s'
              % (name, len(rows), INIT[name], distinct[:4]))
        if name == 'stream_mix':
            l0 = [r[0] for r in rows.values()]
            l1 = [r[1] for r in rows.values()]
            print('        element0 range %.6f..%.6f   element1 range %.6f..%.6f'
                  % (min(l0), max(l0), min(l1), max(l1)))
            moved = sum(1 for x in l1 if abs(x - 0.25) > 0)
            print('        element1 != 0.25 in %d/%d layers' % (moved, len(l1)))
        else:
            f = [r[0] if isinstance(r, list) else r for r in rows.values()]
            moved = sum(1 for x in f if abs(x - float(INIT[name])) > 0)
            print('        != init in %d/%d layers' % (moved, len(f)))
    return per


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    verdict = {}
    for p in sys.argv[1:]:
        per = scan(p)
        sm = per.get('stream_mix', {})
        g = per.get('root_gate', {})
        ig = per.get('ishtiqaq_gamma', {})
        moved_sm = sum(1 for r in sm.values() if abs(r[1] - 0.25) > 0)
        moved_g = sum(1 for r in g.values() if abs(r[0]) > 0)
        moved_ig = sum(1 for r in ig.values() if abs(r[0] - 0.25) > 0)
        verdict[p] = (moved_g, moved_sm, moved_ig, len(g))
    print()
    print('=' * 78)
    print('VERDICT')
    for p, (mg, msm, mig, ng) in verdict.items():
        print('  %s' % p)
        print('    root_gate non-zero      : %d/%d layers' % (mg, ng))
        print('    stream_mix[1] != 0.25   : %d/%d layers' % (msm, ng))
        print('    ishtiqaq_gamma != 0.25  : %d/%d layers' % (mig, ng))
        if ng == 0:
            print('    -> no ishtiqaq modules in this checkpoint (flag off, e.g. FLOOR_A)')
        elif mg == 0 and msm == 0 and mig == 0:
            print('    -> *** BIT-EXACTLY AT INIT: the fixed path received NO gradient ***')
        else:
            print('    -> the fixed path TRAINED (moved off init)')


if __name__ == '__main__':
    main()
