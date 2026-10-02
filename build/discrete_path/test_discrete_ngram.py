#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_discrete_ngram.py -- unit tests for the DISCRETE root-history pathway in nrmt_arch.py.

Run on the pod:  /workspace/venvs/rootformer/bin/python test_discrete_ngram.py
(needs `nrmt_arch` importable, i.e. run from /workspace/hf_v19_2_release or with it on sys.path)

Covers, per the brief: shapes, determinism, gradient flow, no NaN, correct dtypes -- plus the
two properties that make the arm safe to bolt onto a trained head:
  * ngram_mode='none' reproduces the historical branch BIT-EXACTLY
  * ngram_mode='vocab' with gate == 0 also reproduces it bit-exactly (no warm-up slam)
  * the OOV row is exactly 0 and receives exactly 0 gradient on train-like input (backoff)
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/workspace/hf_v19_2_release')
from nrmt_arch import (DiscreteNgramFeatures, NRMTHead, ngram_order_vocabs,  # noqa: E402
                       _NGRAM_HASH_A, _NGRAM_HASH_MOD)

C = int(os.environ.get('TEST_NUM_ROOTS', '64'))
D = int(os.environ.get('TEST_D_MODEL', '32'))
PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}",
          flush=True)


def make_roots(n, vocab_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, vocab_size, (n,), generator=g)


def build_vocab(stream, k, num_roots):
    return ngram_order_vocabs(stream.numpy(), num_roots, (k,))[0]


def np_hash_ref(codes):
    """Independent numpy reference for the module's int64 mixer (all products < 2**63)."""
    h = np.mod(np.asarray(codes, dtype=np.int64), _NGRAM_HASH_MOD)
    h = ((h ^ (h >> 21)) * _NGRAM_HASH_A) % _NGRAM_HASH_MOD
    h = ((h ^ (h >> 17)) * _NGRAM_HASH_A) % _NGRAM_HASH_MOD
    return h


def main():
    torch.manual_seed(0)
    B, T = 3, 12
    stream = make_roots(4000, C, seed=1)                       # a synthetic train root stream
    roots = make_roots(B * T, C, seed=2).reshape(B, T)
    roots[0, :4] = 0                                           # exercise the null context
    k = 3
    vocab = build_vocab(stream, k, C)
    print(f'[*] C={C} d_model={D} vocab(k={k})={len(vocab)} rows (+2 reserved)', flush=True)

    # ---------------------------------------------------------------- 1. shapes / dtype
    mod = DiscreteNgramFeatures(C, D, orders=(1, 2, 3), d_ng=4, mode='vocab',
                                order_vocabs=ngram_order_vocabs(stream.numpy(), C, (1, 2, 3)))
    mod.eval()
    with torch.no_grad():
        mod.gate.fill_(1.0)
    out = mod.build(roots)
    check('shape [B,T,d_model]', tuple(out.shape) == (B, T, D), str(tuple(out.shape)))
    check('dtype float32', out.dtype == torch.float32)
    check('no NaN/Inf', bool(torch.isfinite(out).all()))

    mod64 = DiscreteNgramFeatures(C, D, orders=(1, 2, 3), d_ng=4, mode='vocab',
                                  order_vocabs=ngram_order_vocabs(stream.numpy(), C, (1, 2, 3)))
    mod64 = mod64.double().eval()
    with torch.no_grad():
        mod64.gate.fill_(1.0)
        o64 = mod64.build(roots)
    check('float64 works', o64.dtype == torch.float64 and bool(torch.isfinite(o64).all()))

    # ---------------------------------------------------------------- 2. determinism
    with torch.no_grad():
        a, b = mod.build(roots), mod.build(roots)
    check('deterministic in eval mode', torch.equal(a, b))

    # ---------------------------------------------------------------- 3. id mapping
    with torch.no_grad():
        codes = mod._codes(roots, k)
        valid = torch.arange(T) >= (k - 1)
        ids = mod._ids(codes, k, valid)
    words = stream[:T + 200]
    win = words[:T].unsqueeze(0)
    with torch.no_grad():
        ids2 = mod._ids(mod._codes(win, k), k, torch.arange(T) >= (k - 1))
    check('null positions (t<k-1) -> row 1', bool((ids[0, :k - 1] == 1).all()))
    check('a train context maps to row >= 2', bool((ids2[0, k - 1:] >= 2).all()))
    unseen = torch.full((1, T), 0)
    unseen[0, k - 1:] = C - 1                                  # a root almost surely not in vocab
    with torch.no_grad():
        ids3 = mod._ids(mod._codes(unseen, k), k, torch.arange(T) >= (k - 1))
    check('unseen context -> row 0 (OOV)', bool((ids3[0, k - 1:] == 0).any()))
    # the module's code of a train context must equal the code the VOCAB was built with
    with torch.no_grad():
        c_mod = mod._codes(win, k)[0, k - 1:].numpy()
    c_ref = []
    for t in range(k - 1, T):
        code = 0
        for x in win[0, t - k + 1:t + 1].tolist():
            code = code * C + x
        c_ref.append(code)
    check('module code == forward block code used by the vocab',
          bool((c_mod == np.array(c_ref)).all()))

    # ---------------------------------------------------------------- 4. OOV row is 0
    check('OOV row exactly zero', bool((mod.embs[0].weight[0] == 0).all()))
    check('null row exactly zero', bool((mod.embs[0].weight[1] == 0).all()))

    # ---------------------------------------------------------------- 5. hash matches numpy
    hm = DiscreteNgramFeatures(C, D, orders=(3,), d_ng=4, mode='hash', bucket_bits=10)
    with torch.no_grad():
        hc = hm._codes(roots, 3).reshape(-1)
        hids = hm._ids(hm._codes(roots, 3), 3, torch.ones_like(roots, dtype=torch.bool)).reshape(-1)
    ref = (np_hash_ref(hc.numpy()) % (1 << 10)) + 1
    check('hash bucket == overflow-free numpy reference',
          bool((hids.numpy() == ref).all()))
    check('hash ids within [1, 2**bits]',
          bool((hids >= 1).all() and (hids <= (1 << 10)).all()))

    # ---------------------------------------------------------------- 6. gradients
    mod2 = DiscreteNgramFeatures(C, D, orders=(1, 2, 3), d_ng=4, mode='vocab',
                                 order_vocabs=ngram_order_vocabs(stream.numpy(), C, (1, 2, 3)))
    mod2.train()
    with torch.no_grad():
        mod2.gate.fill_(0.5)
    # use windows FROM the training stream: every full context is then in the vocab by
    # construction, which is the real training situation (and is what keeps the OOV row at 0)
    sw = torch.stack([stream[100 + 37 * i:100 + 37 * i + T] for i in range(B)])
    ro = mod2.build(sw)
    # NB: sum(LN(x)) == d_model * beta is CONSTANT in x, so it would give exactly zero gradient
    # to everything upstream.  Use a random projection as the loss.
    loss = (ro * torch.randn_like(ro)).sum()
    loss.backward()
    g_ok = all(p.grad is not None for p in mod2.parameters())
    g_fin = all(bool(torch.isfinite(p.grad).all()) for p in mod2.parameters())
    nz_proj = float(mod2.proj.weight.grad.abs().sum())
    nz_emb = float(sum(float(e.weight.grad.abs().sum()) for e in mod2.embs))
    check('every parameter gets a gradient', g_ok)
    check('all gradients finite', g_fin)
    check('proj gradient non-zero', nz_proj > 0, f'{nz_proj:.3e}')
    check('embedding gradients non-zero', nz_emb > 0, f'{nz_emb:.3e}')
    check('OOV row gradient exactly zero (all train contexts are in the vocab)',
          all(float(e.weight.grad[0].abs().sum()) == 0.0 for e in mod2.embs))

    # ---------------------------------------------------------------- 7. dead saddle probe
    mod3 = DiscreteNgramFeatures(C, D, orders=(1,), d_ng=4, mode='vocab',
                                 order_vocabs=ngram_order_vocabs(stream.numpy(), C, (1,)))
    mod3.train()
    _o3 = mod3.build(roots)
    (_o3 * torch.randn_like(_o3)).sum().backward()
    gn = float(mod3.gate.grad)
    check('gate has non-zero gradient at gate==0 (no dead saddle)', gn != 0.0, f'{gn:.3e}')

    # ---------------------------------------------------------------- 8. head integration
    head = NRMTHead(D, C, 5, 4, 4, d_root=6, hist=3, d_hist=4, d_op=2, d_morph=2,
                    dropout=0.0, use_features=True, feat_gate=True,
                    ngram_mode='vocab', ngram_orders=(1, 2, 3), d_ng=4,
                    ngram_vocabs=ngram_order_vocabs(stream.numpy(), C, (1, 2, 3)))
    head.eval()
    h = torch.randn(B, T, D)
    op = torch.zeros(B, T, dtype=torch.long)
    w = torch.randint(0, 5, (B, T))
    p = torch.randint(0, 4, (B, T))
    s = torch.randint(0, 4, (B, T))
    with torch.no_grad():
        f_ng = head.build_features(roots, op, w, p, s)
        head2 = NRMTHead(D, C, 5, 4, 4, d_root=6, hist=3, d_hist=4, d_op=2, d_morph=2,
                         dropout=0.0, use_features=True, feat_gate=True, ngram_mode='none')
        head2.load_state_dict({kk: vv for kk, vv in head.state_dict().items()
                               if not kk.startswith('ngram_feat.')}, strict=False)
        f_no = head2.build_features(roots, op, w, p, s)
    check('gate==0 => discrete arm is a no-op (bit-identical to ngram_mode=none)',
          torch.equal(f_ng, f_no),
          f'max|d|={float((f_ng-f_no).abs().max()):.3e}')
    with torch.no_grad():
        head.ngram_feat.gate.fill_(1.0)
        f_on = head.build_features(roots, op, w, p, s)
    check('gate==1 => discrete arm changes the output', not torch.equal(f_on, f_no),
          f'max|d|={float((f_on-f_no).abs().max()):.3e}')
    check('ngram_scale diagnostics', head.ngram_gate() == 1.0 and head.ngram_scale() > 0
          and head.ngram_params() > 0)

    # ---------------------------------------------------------------- 9. historical path
    head3 = NRMTHead(D, C, 5, 4, 4, d_root=6, hist=3, d_hist=4, d_op=2, d_morph=2,
                     dropout=0.0, use_features=False, ngram_mode='none')
    with torch.no_grad():
        z = head3.build_features(roots, op, w, p, s)
    check('use_features=False & mode=none -> exact zeros',
          bool((z == 0).all()) and tuple(z.shape) == (B, T, D))
    head4 = NRMTHead(D, C, 5, 4, 4, d_root=6, hist=3, d_hist=4, d_op=2, d_morph=2,
                     dropout=0.0, use_features=False, ngram_mode='vocab', ngram_orders=(1, 2, 3),
                     d_ng=4, ngram_vocabs=ngram_order_vocabs(stream.numpy(), C, (1, 2, 3)))
    head4.eval()
    with torch.no_grad():
        head4.ngram_feat.gate.fill_(1.0)
        z4 = head4.build_features(roots, op, w, p, s)
    check('discrete-only arm works with use_features=False (non-zero)',
          tuple(z4.shape) == (B, T, D) and float(z4.abs().sum()) > 0)

    # ---------------------------------------------------------------- 10. real config smoke
    if int(os.environ.get('TEST_FULL', '0')):
        d_model, num_roots = 896, 9490
        st = torch.randint(1, num_roots, (200000,), generator=torch.Generator().manual_seed(5))
        vos = ngram_order_vocabs(st.numpy(), num_roots, (1, 2, 3, 4))
        t0 = __import__('time').time()
        big = DiscreteNgramFeatures(num_roots, d_model, orders=(1, 2, 3, 4), d_ng=16,
                                    mode='vocab', order_vocabs=vos, gated=True)
        r = torch.randint(1, num_roots, (2, 128))
        with torch.no_grad():
            big.gate.fill_(1.0)
            o = big.build(r)
        check('real-config (C=9490,d=896,d_ng=16, orders 1-4) forward',
              tuple(o.shape) == (2, 128, d_model) and bool(torch.isfinite(o).all()),
              f'{__import__("time").time()-t0:.1f}s, params={big.table_stats()}')

    print(f'\n==== {len(PASS)} passed, {len(FAIL)} failed ====')
    if FAIL:
        print('FAILED: ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
