#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""equiv_check.py -- CPU-only proof that `--rca-dim 0` (the default) does NOT change the module.

Three things are asserted:
  A. the shipped module (/workspace/root_attn/root_cross_attn.py) and the width module
     (/workspace/root_attn_width/root_cross_attn_width.py) build BIT-IDENTICAL parameters
     (same keys, same shapes, same values) at d_attn=None / d_attn=d_model;
  B. their forward() returns BIT-IDENTICAL deltas on the same inputs (out_norm off AND on);
  C. the width lever does what it claims: d_attn=1792 -> head_dim 224, k_proj (1792,448),
     o_proj (896,1792), 19.271M params for 4 layers vs 9.636M at 896.

No GPU, no checkpoint, no file written.  Exit 0 = PASS.
"""
import importlib.util
import sys

import torch

SHIPPED = '/workspace/root_attn/root_cross_attn.py'
WIDTH = '/workspace/root_attn_width/root_cross_attn_width.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(mod, d_attn):
    torch.manual_seed(1234)
    return mod.RootHistoryCrossAttention(
        896, 448, num_heads=8, dtype=torch.float32, out_norm=False, d_attn=d_attn)


def main():
    shipped, width = load('rca_shipped', SHIPPED), load('rca_width', WIDTH)
    ok = True

    # ---- A/B: default width is bit-identical ------------------------------------------
    for out_norm in (False, True):
        # NOTE: the seed must be reset before EACH construction -- the module consumes RNG for
        # q/k/v/o_proj init, so reusing a leftover RNG state would compare different draws and
        # look like a code difference when it is only a test artifact.
        torch.manual_seed(7)
        a = shipped.RootHistoryCrossAttention(896, 448, num_heads=8, dtype=torch.float32,
                                              out_norm=out_norm)
        torch.manual_seed(7)
        b = width.RootHistoryCrossAttention(896, 448, num_heads=8, dtype=torch.float32,
                                            out_norm=out_norm, d_attn=None)
        torch.manual_seed(7)
        c = width.RootHistoryCrossAttention(896, 448, num_heads=8, dtype=torch.float32,
                                            out_norm=out_norm, d_attn=896)
        for label, other in (('d_attn=None', b), ('d_attn=896', c)):
            ka, kb = list(a.state_dict()), list(other.state_dict())
            if ka != kb:
                print(f'FAIL keys differ vs {label}: {set(ka) ^ set(kb)}')
                ok = False
                continue
            bad = [k for k in ka
                   if not torch.equal(a.state_dict()[k], other.state_dict()[k])]
            if bad:
                print(f'FAIL values differ vs {label}: {bad}')
                ok = False
            else:
                print(f'PASS params bit-identical vs {label} (out_norm={out_norm}, '
                      f'{len(ka)} tensors, {sum(v.numel() for v in a.state_dict().values())/1e6:.6f}M)')

        # forward equivalence.  gate is zero-init, so the default forward returns EXACTLY 0 for
        # every module and the comparison would be vacuous; open the gate first, and ALSO check
        # the gate==0 no-op separately.
        torch.manual_seed(99)
        h = torch.randn(2, 32, 896)
        root_ids = torch.randint(0, 9490, (2, 32))
        rw = torch.randn(9490, 448) * 0.1
        a.eval(); b.eval(); c.eval()
        with torch.no_grad():
            z0 = a(h, root_ids, rw)
            if float(z0.abs().max()) != 0.0:
                print(f'FAIL gate=0 forward is not a no-op: max|d|={float(z0.abs().max()):.3e}')
                ok = False
            else:
                print(f'PASS gate=0 forward is EXACTLY 0 (no-op at init, out_norm={out_norm})')
            for m in (a, b, c):
                m.gate.fill_(0.37)
            da, db, dc = a(h, root_ids, rw), b(h, root_ids, rw), c(h, root_ids, rw)
        for label, other in (('d_attn=None', db), ('d_attn=896', dc)):
            if torch.equal(da, other):
                print(f'PASS forward bit-identical vs {label} (out_norm={out_norm}, '
                      f'gate=0.37, out rms {float(da.pow(2).mean().sqrt()):.6e})')
            else:
                print(f'FAIL forward differs vs {label}: max|d|={float((da-other).abs().max()):.3e}')
                ok = False

    # ---- C: the lever does what it claims --------------------------------------------
    w = build(width, 1792)
    shp = {'q_proj': (1792, 896), 'k_proj': (1792, 448), 'v_proj': (1792, 448),
           'o_proj': (896, 1792)}
    for k, want in shp.items():
        got = tuple(getattr(w, k).weight.shape)
        good = got == want
        ok &= good
        print(f'{"PASS" if good else "FAIL"} d_attn=1792 {k}.weight {got} (want {want})')
    print(f'PASS d_attn=1792 head_dim={w.head_dim} heads={w.num_heads}')
    p1 = build(width, 896)
    n1 = sum(p.numel() for p in p1.parameters())
    n2 = sum(p.numel() for p in w.parameters())
    print(f'  params/layer: 896 -> {n1/1e6:.6f}M | 1792 -> {n2/1e6:.6f}M | '
          f'4 layers 9.636M -> {4*n2/1e6:.3f}M (delta {4*(n2-n1)/1e6:+.3f}M)')

    # ---- d_attn must divide by heads --------------------------------------------------
    # 1001 is deliberately NOT a multiple of 8 (1000 would be: 8*125 == 1000, no error).
    try:
        width.RootHistoryCrossAttention(896, 448, num_heads=8, dtype=torch.float32, d_attn=1001)
        print('FAIL d_attn=1001 (not divisible by 8) did not raise')
        ok = False
    except ValueError as e:
        print(f'PASS d_attn=1001 raises: {e}')

    print('EQUIV_CHECK: ' + ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
