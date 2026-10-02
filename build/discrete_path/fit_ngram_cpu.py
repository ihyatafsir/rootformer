#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_ngram_cpu.py -- short CPU optimisation of the DISCRETE root-history pathway on the REAL
train root stream, to answer the one question the closed-form proof does not: can gradient
descent actually FIND a good table, at the table size the GPU arm uses?

This is NOT a substitute for the GPU run -- the GPU run gets ~81 M position-samples (11.5
epochs); this fits a bounded budget of steps and reports the whole learning curve so the
extrapolation is explicit.  It is the "short optimisation" arm of the CPU-first proof.

Faithful to the GPU arm in: vocab mode, orders 1-4, d_ng, the zero-init gate, the zero OOV row,
AdamW(lr=1e-3, wd=0.01), batch of real 128-token windows, and the masking/protocol of
nrmt_train.py.  NOT faithful in: h = 0 (no frozen backbone; this isolates the pathway), and the
number of steps.

CPU only.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path('/workspace/discrete_path')
WIN, STRIDE, CTX = 128, 64, 12
HASH_B = np.uint64(1000003)
ORDERS = (1, 2, 3, 4)


def rolling_hashes(stream, n, B=HASH_B):
    a = np.asarray(stream, dtype=np.uint64)
    L = len(a) - n + 1
    h = np.zeros(max(L, 0), dtype=np.uint64)
    for k in range(n):
        h = h * B + a[k:k + L]
    return h


def hash_ctx(ctx, B=1000003):
    h = 0
    for x in ctx:
        h = (h * B + int(x)) & ((1 << 64) - 1)
    return np.uint64(h)


def codes_for(stream, k, num_roots):
    a = np.asarray(stream, dtype=np.int64)
    L = len(a) - k + 1
    code = np.zeros(max(L, 0), dtype=np.int64)
    for j in range(k):
        code = code * num_roots + a[j:j + L]
    return code


def topk_map(keys, targets, k):
    order = np.lexsort((targets, keys))
    ks, ts = keys[order], targets[order]
    n = len(ks)
    newgrp = np.empty(n, dtype=bool)
    newgrp[0] = True
    newgrp[1:] = ks[1:] != ks[:-1]
    newpair = newgrp.copy()
    newpair[1:] |= (ts[1:] != ts[:-1])
    pid = np.cumsum(newpair) - 1
    npairs = int(pid[-1]) + 1
    pcount = np.bincount(pid).astype(np.int64)
    p_tgt = ts[newpair]
    p_grp = np.cumsum(newgrp)[newpair] - 1
    G = int(newgrp.sum())
    gh = ks[newgrp]
    o = np.lexsort((-pcount, p_grp))
    p_grp_s, p_tgt_s, pcount_s = p_grp[o], p_tgt[o], pcount[o]
    first = np.empty(npairs, dtype=bool)
    first[0] = True
    first[1:] = p_grp_s[1:] != p_grp_s[:-1]
    starts = np.flatnonzero(first)
    rank = np.arange(npairs) - np.repeat(starts, np.diff(np.append(starts, npairs)))
    gsize = np.minimum(np.diff(np.append(starts, npairs)), k)
    g_arr = np.repeat(np.arange(G), gsize)
    sel = rank < k
    top = np.zeros((G, k), dtype=np.int64)
    top[g_arr, rank[sel]] = p_tgt_s[sel]
    return gh, top


def starts_for(t, n, seed):
    s = list(range(0, len(t) - WIN - 1, STRIDE))
    np.random.default_rng(seed).shuffle(s)
    return sorted(s[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--d-ng', type=int, default=16)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--eval-windows', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--no-gate', action='store_true',
                    help='disable the fade-in gate (the branch starts at FULL scale) -- this '
                         'removes the gate-ramp confound and measures the TABLE learnability alone')
    ap.add_argument('--out', default=str(OUT / 'fit_ngram_cpu.json'))
    args = ap.parse_args()

    import sys
    sys.path.insert(0, '/workspace/hf_v19_2_release')
    import nrmp_vocab as nv
    from nrmt_arch import DiscreteNgramFeatures, ngram_order_vocabs

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t00 = time.time()
    tr = [t.long() for t in torch.load(Path(args.cache) / 'train.pt', map_location='cpu')]
    va = [t.long() for t in torch.load(Path(args.cache) / 'val.pt', map_location='cpu')]
    tr_r = tr[1].numpy().astype(np.int64)
    va_r = va[1].numpy().astype(np.int64)
    V = next(v for k, v in nv.__dict__.items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    C, d_model = int(vocab.num_roots), 896
    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec_t = torch.tensor(specials)
    torch.set_num_threads(int(__import__('os').environ.get('OMP_NUM_THREADS', '8')))

    t0 = time.time()
    vos = ngram_order_vocabs(tr_r, C, ORDERS)
    print('[*] vocab ' + ', '.join(f'k{k}={len(v)}' for k, v in zip(ORDERS, vos))
          + f' ({time.time()-t0:.0f}s)', flush=True)
    tab = sum(len(v) + 2 for v in vos) * args.d_ng
    print(f'[*] discrete table params {tab/1e6:.1f}M (d_ng={args.d_ng})', flush=True)

    # ---- val windows + masks, identical protocol to nrmt_train.py ------------------------
    va_st = starts_for(va_r, 300, 1)
    Tv = np.stack([va_r[j:j + WIN] for j in va_st])
    va_keep_st = sorted(np.random.default_rng(0).permutation(len(va_st))[:args.eval_windows])
    Rv = torch.from_numpy(Tv[:, :-1]).long()
    tv = Torch_t = torch.from_numpy(Tv[:, 1:].reshape(-1)).long()
    keep_v = ~torch.isin(tv, spec_t)
    tr12 = np.unique(rolling_hashes(tr_r, CTX))
    novel = np.zeros(Tv.shape[0] * (WIN - 1), dtype=bool)
    for w in range(Tv.shape[0]):
        seg = Tv[w]
        for t in range(CTX, WIN):
            h = hash_ctx(seg[t - CTX:t])
            i = np.searchsorted(tr12, h)
            novel[w * (WIN - 1) + (t - 1)] = not (i < len(tr12) and tr12[i] == h)
    keep_n = keep_v & torch.from_numpy(novel)
    print(f'[*] val positions all={int(keep_v.sum())} novel={int(keep_n.sum())}', flush=True)

    # ---- the module + a readout, exactly the head's root path with h = 0 ------------------
    mod = DiscreteNgramFeatures(C, d_model, orders=ORDERS, d_ng=args.d_ng, mode='vocab',
                                order_vocabs=vos, gated=not args.no_gate, emb_init_std=0.02)
    W = nn.Linear(d_model, C, bias=False)
    params = list(mod.parameters()) + list(W.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    lookup = {}
    for k in ORDERS:
        codes = codes_for(tr_r, k, C)
        tgt = tr_r[k:]
        L = min(len(codes), len(tgt))
        gh, tops = topk_map(codes[:L], tgt[:L], 5)
        lookup[k] = (gh, tops)
    print(f'[*] lookup reference order<=4 = 53.13% ALL / 49.51% NOVEL', flush=True)

    @torch.no_grad()
    def evaluate():
        mod.eval()
        lg, idx = [], va_keep_st
        for i in range(0, len(idx), 16):
            sl = torch.tensor(idx[i:i + 16])
            f = mod.build(Rv[sl])
            lg.append(f.reshape(-1, d_model) @ W.weight.t())
        LG = torch.cat(lg)
        # masks for the subset, in the same window-major order as LG
        sel = torch.from_numpy(np.concatenate(
            [np.arange(w * (WIN - 1), (w + 1) * (WIN - 1)) for w in idx]))
        tv_sel, kv_sel, kn_sel = tv[sel], keep_v[sel], keep_n[sel]
        out = {}
        for tag, m in (('all', kv_sel), ('novel', kn_sel)):
            l, t = LG[m], tv_sel[m]
            out[tag] = {'n': int(m.sum()),
                        'acc@1_pct': 100 * float((l.argmax(-1) == t).float().mean())}
        mod.train()
        return out

    rows = []
    n_win = len(tr_r) - WIN - 1
    rng = np.random.default_rng(args.seed)
    t1 = time.time()
    mod.train()
    for step in range(1, args.steps + 1):
        j = rng.integers(0, n_win, size=args.batch_size)
        R = torch.stack([torch.from_numpy(tr_r[a:a + WIN - 1]) for a in j]).long()
        T = torch.stack([torch.from_numpy(tr_r[a + 1:a + WIN]) for a in j]).long()
        f = mod.build(R)
        logits = f.reshape(-1, d_model) @ W.weight.t()
        tt = T.reshape(-1)
        keep = ~torch.isin(tt, spec_t)
        loss = F.cross_entropy(logits[keep], tt[keep])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        gv = float(mod.gate.detach()) if getattr(mod, 'gated', False) else 1.0
        if step % 50 == 0:
            print(f'  step {step}/{args.steps} loss {float(loss):.4f} gnorm {gn:.3f} '
                  f'gate {gv:.5f} {step/(time.time()-t1):.2f} it/s', flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            e = evaluate()
            rows.append({'step': step, 'loss': float(loss), 'gate': gv, **e})
            print(f'  [eval @{step}] ALL {e["all"]["acc@1_pct"]:6.2f}%  '
                  f'NOVEL {e["novel"]["acc@1_pct"]:6.2f}%  gate {gv:.5f}', flush=True)

    res = {'args': vars(args), 'vocab_sizes': {k: len(v) for k, v in zip(ORDERS, vos)},
           'table_params_M': tab / 1e6, 'lookup_reference': {'all': 53.13, 'novel': 49.51},
           'curve': rows, 'wall_s': time.time() - t00}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f'[*] wrote {args.out} in {res["wall_s"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
