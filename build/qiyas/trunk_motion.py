#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trunk_motion.py -- is the TRUNK actually moving?  A head-independent check on a saved arm
checkpoint, independent of any self-reported metric.

WHY
---
The trainer claims "383.3M backbone + 4.29M embed trainable" under --unfreeze-trunk-all.  But the
project has already been burned by parameters that LOOK trainable and receive nothing:
`stream_mix[1].grad == 0.0` exactly in all 24 layers, 201 causally-inert parameters, and a naive
state-dict load that silently loaded nothing.  If FLOOR_A's trunk is not moving, then "matched
trunk training" is false and the C-vs-X comparison would be measuring a frozen trunk that nobody
noticed.

This loads the arm's `.trunk.pt` and the ORIGINAL release checkpoint, matches backbone tensors,
and reports the per-layer delta.  A trunk that is training shows non-zero drift concentrated in
whatever the LR schedule has reached; a frozen one shows 0.0 everywhere.

CPU only.  Reads only.  Writes nothing.

usage: python trunk_motion.py /tmp/root_arch_arms/head_FLOOR_A.pt.trunk.pt
"""
from __future__ import annotations

import os
import sys

import torch

ORIG = os.environ.get(
    'ORIG_CKPT',
    '/workspace/hf_v19_2_release/checkpoints/'
    'rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors')


def load_orig(path):
    """Load the release checkpoint as a flat name -> tensor dict.

    safetensors may carry a `backbone.` prefix, or the project's `backbone.model.` CausalLM
    layout.  Try the direct load first and report what was found rather than guessing silently.
    """
    from safetensors.torch import load_file
    sd = load_file(path)
    return sd


def norm_name(n):
    """Map both layouts onto a comparable suffix keyed on the layer index."""
    for pref in ('backbone.model.', 'backbone.', 'model.'):
        if n.startswith(pref):
            return 'backbone.' + n[len(pref):]
    return n


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    arm_path = sys.argv[1]
    if not os.path.exists(arm_path):
        raise SystemExit('missing arm checkpoint: %s' % arm_path)

    blob = torch.load(arm_path, map_location='cpu', weights_only=False)
    arm = blob.get('state', blob)
    print('[*] arm checkpoint : %s' % arm_path)
    print('    keys=%d  unfrozen_layers=%s' % (len(arm), blob.get('unfrozen_layers')))
    print('    step=%s  live=%s' % (blob.get('step'), blob.get('live')))

    print('[*] original       : %s' % ORIG)
    orig = load_orig(ORIG)
    print('    keys=%d' % len(orig))
    on = {norm_name(k): v for k, v in orig.items()}

    # compare every backbone tensor present in BOTH
    rows = []
    missing = 0
    for k, v in arm.items():
        bn = norm_name(k)
        if not bn.startswith('backbone.layers.'):
            continue
        o = on.get(bn)
        if o is None:
            missing += 1
            continue
        if o.shape != v.shape:
            rows.append((bn, float('nan'), 'SHAPE %s vs %s' % (tuple(o.shape), tuple(v.shape))))
            continue
        d = (v.to(torch.float32) - o.to(torch.float32))
        rows.append((bn, float(d.abs().max()), ''))

    print('    backbone tensors compared: %d   (not found in orig: %d)' % (len(rows), missing))

    # aggregate per layer
    import collections
    per = collections.defaultdict(lambda: [0, 0.0])   # n, maxabs
    shape_skips = 0
    for bn, mx, note in rows:
        if note:
            shape_skips += 1
            continue
        parts = bn.split('.')
        try:
            li = int(parts[2])
        except (IndexError, ValueError):
            continue
        per[li][0] += 1
        per[li][1] = max(per[li][1], mx)
    if shape_skips:
        print('    (%d tensors skipped on SHAPE mismatch -- expected for the stale 9015/128 '
              'root/wazn tables if the baseline is the pre-alignment release)' % shape_skips)

    print()
    print('    layer   tensors   max|delta| vs original')
    moved = 0
    frozen = 0
    for li in sorted(per):
        n, mx = per[li]
        tag = 'MOVED' if mx > 0 else 'FROZEN'
        if mx > 0:
            moved += 1
        else:
            frozen += 1
        print('    %5d   %7d   %.6e   %s' % (li, n, mx, tag))
    print()
    print('  VERDICT: %d/%d backbone tensors moved, %d frozen' % (moved, moved + frozen, frozen))
    if moved == 0:
        print('  *** TRUNK IS NOT MOVING -- "matched trunk training" would be FALSE ***')
    else:
        print('  trunk is moving; --unfreeze-trunk-all is doing what it claims')

    # also confirm the root pathway survived into the checkpoint (this arm may have none)
    rca = [k for k in arm if k.startswith('root_cross.')]
    ish = [k for k in arm if 'ishtiqaq' in k or 'stream_mix' in k]
    print('  root_cross.* tensors: %d   ishtiqaq/stream_mix tensors: %d' % (len(rca), len(ish)))


if __name__ == '__main__':
    main()
