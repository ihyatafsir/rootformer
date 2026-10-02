#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expressibility_proof.py -- prove, on CPU, that the DISCRETE root-history pathway can represent
the order-k root n-gram lookup that scores 53.13 % acc@1 on val -- BEFORE any GPU time.

WHAT IS PROVED, AND HOW
-----------------------
The module is `nrmt_arch.DiscreteNgramFeatures`: for every distinct order-k context it owns an
embedding row e, and the branch output is

    f = LayerNorm( sum_{k in orders} proj_k(e^k_{ctx_k}) )        (gate = 1)

added to h_t, after which the head's own `root_head` W maps to root logits.

Claim: for any lookup table ctx -> class (the majority next root of that ctx) there is a setting
of (e, proj, LayerNorm, W) that realises it on every context the table covers.

CONSTRUCTION (closed form, no gradient descent):
  1. draw x_c ~ N(0, I) in R^{d_ng}, one prototype per root class c.
  2. draw an orthonormal partial isometry P: R^{d_ng} -> R^{d_model} (proj row block).
  3. set z_c = LayerNorm(P x_c).  Then ||z_c|| = 1 exactly (LayerNorm output is unit RMS).
  4. set the readout W := Z^T, i.e. row/column c of W is z_c.
  5. set the order-k embedding row of context b to x_{class_k(b)}, where class_k(b) is the
     order-k lookup's top-1 target.  Rows 0 (OOV) and 1 (null) stay exactly ZERO.
  6. scale the order blocks geometrically: proj_k = eps^(K-k) * P.

WHY IT WORKS
  * logits for a context whose order-K row is e = x_c are (W f)_j = <z_j, LN(...)>; with only
    order K present this is <z_j, z_c>, whose maximum is at j = c because <z_c,z_c> = 1 and
    |<z_j,z_c>| <= ||z_j|| ||z_c|| = 1, with equality only for z_j == z_c (probability 0).
    So the argmax is EXACT, with no margin requirement.
  * a val context whose order-K n-gram was never seen in train has embedding row 0 = 0, so the
    order-K term VANISHES and the next-shorter seen order decides.  That is the backoff, exactly
    as in the count lookup: the eps-geometric scaling makes the shorter-order terms perturbations
    of relative size eps, so they cannot flip the argmax.
  * the same argument applies with h_t != 0: scaling the module's LayerNorm gain gamma by rho
    scales f by rho, so W h_t becomes an arbitrarily small *relative* perturbation.

The script measures, on the SAME 300 val windows / 18,869 radical positions / 15,046 NOVEL
positions as every 20k run:
  * lookup acc@1 (reference) at order <= K, K = 1..4
  * constructed-module acc@1 at the same K
  * the prototype margin 1 - max_{j != c} <z_j, z_c>  (the robustness budget)
  * the accuracy when a unit-RMS h_t is added, as a function of the LayerNorm gain rho.

CPU only. Read-only on the caches.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = Path('/workspace/head_fix/nrmp_cache_9490_aligned')
OUT = Path('/workspace/discrete_path')
WIN, STRIDE, CTX = 128, 64, 12
HASH_B = np.uint64(1000003)
ORDERS = (1, 2, 3, 4)


# --------------------------------------------------------------------------- shared helpers
def rolling_hashes(stream, n, B=HASH_B):
    a = np.asarray(stream, dtype=np.uint64)
    L = len(a) - n + 1
    if L <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(L, dtype=np.uint64)
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
    cts = np.zeros((G, k), dtype=np.int64)
    top[g_arr, rank[sel]] = p_tgt_s[sel]
    cts[g_arr, rank[sel]] = pcount_s[sel]
    gcount = np.bincount(p_grp).astype(np.int64)
    return gh, top, cts, gcount


def starts_for(t, n, seed):
    s = list(range(0, len(t) - WIN - 1, STRIDE))
    np.random.default_rng(seed).shuffle(s)
    return sorted(s[:n])


def val_keys(W, k, num_roots):
    nwin = W.shape[0]
    hs = np.zeros((nwin, WIN - 1), dtype=np.int64)
    for w in range(nwin):
        rh = codes_for(W[w], k, num_roots)
        hs[w, k - 1:] = rh[:WIN - k]
    return hs.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=str(CACHE))
    ap.add_argument('--min-count', type=int, default=1)
    ap.add_argument('--d-ngs', default='16,32,64')
    ap.add_argument('--eps', type=float, default=0.02, help='geometric decay of the order blocks')
    ap.add_argument('--seed', type=int, default=20261002)
    ap.add_argument('--out', default=str(OUT / 'expressibility_proof.json'))
    args = ap.parse_args()

    import sys
    sys.path.insert(0, '/workspace/hf_v19_2_release')
    import nrmp_vocab as nv
    from nrmt_arch import DiscreteNgramFeatures

    t00 = time.time()
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(__import__('os').environ.get('OMP_NUM_THREADS', '8')))

    tr = [t.long() for t in torch.load(Path(args.cache) / 'train.pt', map_location='cpu')]
    va = [t.long() for t in torch.load(Path(args.cache) / 'val.pt', map_location='cpu')]
    tr_r = tr[1].numpy().astype(np.int64)
    va_r = va[1].numpy().astype(np.int64)

    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    C = int(vocab.num_roots)
    d_model = 896
    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec = np.array(specials)

    va_st = starts_for(va_r, 300, 1)
    Tv = np.stack([va_r[j:j + WIN] for j in va_st])
    tv = Tv[:, 1:].reshape(-1)
    keep_v = ~np.isin(tv, spec)
    tr12 = np.unique(rolling_hashes(tr_r, CTX))
    novel = np.zeros(len(tv), dtype=bool)
    for w in range(Tv.shape[0]):
        seg = Tv[w]
        for t in range(CTX, WIN):
            h = hash_ctx(seg[t - CTX:t])
            i = np.searchsorted(tr12, h)
            novel[w * (WIN - 1) + (t - 1)] = not (i < len(tr12) and tr12[i] == h)
    keep_novel = keep_v & novel
    n_all, n_nov = int(keep_v.sum()), int(keep_novel.sum())
    print(f'[*] val radical all={n_all} novel={n_nov}', flush=True)

    tv_t = torch.from_numpy(tv)
    Rv = torch.from_numpy(Tv[:, :-1]).long()          # the head's root input, exactly as trained
    keep_v_t = torch.from_numpy(keep_v)
    keep_n_t = torch.from_numpy(keep_novel)

    # ---- per-order lookup tables from the FULL train stream ------------------------------
    look = {}
    for k in ORDERS:
        codes = codes_for(tr_r, k, C)
        tgt = tr_r[k:]
        L = min(len(codes), len(tgt))
        gh, top, cts, gcount = topk_map(codes[:L], tgt[:L], 5)
        look[k] = {'vocab': gh, 'tops': top, 'count': gcount}
        print(f'[*] order {k}: {len(gh)} distinct train contexts', flush=True)
    keys_val = {k: val_keys(Tv, k, C) for k in ORDERS}
    uni = np.bincount(tr_r, minlength=C)
    uni_top5 = np.argsort(-uni)[:5]

    def lookup_eval(kmax, min_count):
        pred = np.tile(uni_top5, (len(tv), 1))
        resolved = np.zeros(len(tv), dtype=bool)
        used = np.zeros(len(tv), dtype=np.int64)
        for k in range(kmax, 0, -1):
            gh, tops, gcount = look[k]['vocab'], look[k]['tops'], look[k]['count']
            hs = keys_val[k]
            idx = np.clip(np.searchsorted(gh, hs), 0, len(gh) - 1)
            found = (gh[idx] == hs) & (gcount[idx] >= min_count)
            if k == 1:
                found = found | True
            valid = np.tile(np.arange(WIN - 1) >= k - 1, Tv.shape[0])
            take = found & valid & ~resolved
            pred[take] = tops[idx[take]]
            used[take] = k
            resolved |= take
        out = {}
        for tag, keep in (('all', keep_v), ('novel', keep_novel)):
            p, t = pred[keep], tv[keep]
            out[tag] = {'n': int(keep.sum()), 'acc@1_pct': 100 * float((p[:, 0] == t).mean()),
                        'acc@5_pct': 100 * float((p == t[:, None]).any(-1).mean())}
        out['backoff_hist_all'] = {int(i): int(v) for i, v in
                                   enumerate(np.bincount(used[keep_v], minlength=5)) if v}
        return out

    # ---- closed-form construction --------------------------------------------------------
    def build_constructed(K, d_ng, min_count, eps, seed):
        g = torch.Generator().manual_seed(seed)
        # orthonormal partial isometry P: R^{d_ng} -> R^{d_model}
        A = torch.randn(d_model, d_ng, generator=g)
        Q, _ = torch.linalg.qr(A)                       # Q: [d_model, d_ng], orthonormal cols
        P = Q.t().contiguous()                          # [d_ng, d_model]
        X = torch.randn(C, d_ng, generator=g)           # one prototype per root class
        mod = DiscreteNgramFeatures(num_roots=C, d_model=d_model, orders=tuple(range(1, K + 1)),
                                    d_ng=d_ng, mode='vocab',
                                    order_vocabs=[look[k]['vocab'] for k in range(1, K + 1)],
                                    gated=True, norm=True, emb_init_std=1.0)
        # (3) z_c = LayerNorm(P x_c) with gamma=1, beta=0  == exactly what the module computes
        with torch.no_grad():
            mod.norm.weight.fill_(1.0)
            mod.norm.bias.zero_()
            mod.norm.eps = 1e-5
            Y = X @ P                                   # [C, d_model]
            Z = mod.norm(Y)                             # ||Z_c|| = sqrt(d_model)
            mod.gate.fill_(1.0)
            # (5) tables: row 0 = OOV = 0, row 1 = null = 0, row pos+2 = x_{class_k(ctx)}
            # Contexts whose train count < min_count are PRUNED: their row stays 0, exactly like
            # OOV, so the shorter orders decide -- i.e. the module implements the pruned lookup.
            sizes = []
            for i, k in enumerate(range(1, K + 1)):
                gh, tops, gc = look[k]['vocab'], look[k]['tops'], look[k]['count']
                keepc = gc >= min_count
                cells = torch.from_numpy(tops[keepc, 0].astype(np.int64))
                idx_full = torch.from_numpy(np.flatnonzero(keepc).astype(np.int64))
                w = mod.embs[i].weight
                w.zero_()
                w.data[2 + idx_full] = X[cells]
                sizes.append(int(keepc.sum()))
            # (6) geometric block scaling proj_k = eps^(K-k) * P
            Wp = mod.proj.weight.data
            Wp.zero_()
            for i, k in enumerate(range(1, K + 1)):
                Wp[:, i * d_ng:(i + 1) * d_ng] = (eps ** (K - k)) * P.t()
            W = Z.t().contiguous()                      # (4) readout W := Z^T
        # margin: 1 - max_{j != c} cos(z_j, z_c)   (chunked, no C x C allocation)
        Zn = F.normalize(Z, dim=-1)
        marg = torch.empty(C)
        for i0 in range(0, C, 512):
            blk = Zn[i0:i0 + 512] @ Zn.t()
            blk[torch.arange(blk.shape[0]), torch.arange(i0, i0 + blk.shape[0])] = -2.0
            marg[i0:i0 + blk.shape[0]] = 1.0 - blk.max(dim=1).values
        return mod, W, X, Z, marg, sizes

    def eval_constructed(mod, W, rho=1.0, h_pert=None):
        mod = mod.eval()
        if rho != 1.0:
            mod.norm.weight.data.fill_(rho)
        with torch.no_grad():
            logits = []
            for i in range(0, Rv.shape[0], 50):
                f = mod.build(Rv[i:i + 50])
                if h_pert is not None:
                    f = f + h_pert
                logits.append(f.reshape(-1, mod.proj.out_features) @ W)
            LG = torch.cat(logits)
        out = {}
        for tag, keep in (('all', keep_v_t), ('novel', keep_n_t)):
            l, t = LG[keep], tv_t[keep]
            out[tag] = {'n': int(keep.sum()),
                        'acc@1_pct': 100 * float((l.argmax(-1) == t).float().mean()),
                        'acc@5_pct': 100 * float((l.topk(5, -1).indices == t[:, None])
                                                 .any(-1).float().mean())}
        return out

    report = {'cache': args.cache, 'n_all': n_all, 'n_novel': n_nov, 'eps': args.eps,
              'protocol': 'closed-form construction: x_c random, P orthonormal, W = LN(P X)^T',
              'lookup': {}, 'constructed': {}}

    print('[*] lookup references', flush=True)
    for K in ORDERS:
        r = lookup_eval(K, 1)
        report['lookup'][K] = r
        print(f'    lookup order<={K}: ALL {r["all"]["acc@1_pct"]:6.2f}%  NOVEL '
              f'{r["novel"]["acc@1_pct"]:6.2f}%', flush=True)

    d_ngs = [int(x) for x in args.d_ngs.split(',')]
    for d_ng in d_ngs:
        for K in ORDERS:
            t1 = time.time()
            mod, W, X, Z, marg, sizes = build_constructed(K, d_ng, args.min_count, args.eps,
                                                          args.seed)
            acc = eval_constructed(mod, W)
            fam = float((marg < 0).float().mean())         # fraction of classes whose self-max fails
            rep = {'acc': acc, 'margin_min': float(marg.min()), 'margin_p01': float(marg.quantile(0.01)),
                   'margin_median': float(marg.median()), 'frac_classes_failing_selfmax': fam,
                   'table_sizes': sizes, 'wall_s': time.time() - t1}
            report['constructed'][f'K{K}_d{d_ng}_mc{args.min_count}'] = rep
            print(f'    constructed K<={K} d_ng={d_ng}: ALL {acc["all"]["acc@1_pct"]:6.2f}%  '
                  f'NOVEL {acc["novel"]["acc@1_pct"]:6.2f}%  margin(min/p01/med) '
                  f'{marg.min():.3f}/{marg.quantile(0.01):.3f}/{marg.median():.3f}  '
                  f'fail={100*fam:.3f}%  sizes={sizes} ({time.time()-t1:.0f}s)', flush=True)

    # ---- h_t dominance: a unit-RMS hidden state added to the module output ---------------
    print('[*] h_t dominance test (K=4, d_ng=%d): accuracy vs the module gain rho' % d_ngs[-1],
          flush=True)
    mod, W, X, Z, marg, sizes = build_constructed(4, d_ngs[-1], args.min_count, args.eps, args.seed)
    g = torch.Generator().manual_seed(7)
    hp = torch.randn(1, 1, d_model, generator=g)
    hp = hp / hp.norm() * np.sqrt(d_model)                 # unit RMS per element
    report['h_t_dominance'] = {}
    for rho in (1.0, 3.0, 10.0, 30.0):
        acc = eval_constructed(mod, W, rho=rho, h_pert=hp)
        report['h_t_dominance'][rho] = acc
        print(f'    rho={rho:5.1f}: ALL {acc["all"]["acc@1_pct"]:6.2f}%  NOVEL '
              f'{acc["novel"]["acc@1_pct"]:6.2f}%', flush=True)
    mod.norm.weight.data.fill_(1.0)

    # ---- min-count pruning at K=4, d_ng=64 ----------------------------------------------
    report['pruning'] = {}
    for mc in (2, 3, 5):
        t1 = time.time()
        mod, W, X, Z, marg, sizes = build_constructed(4, d_ngs[-1], mc, args.eps, args.seed)
        acc = eval_constructed(mod, W)
        ref = lookup_eval(4, mc)
        report['pruning'][mc] = {'constructed': acc, 'lookup': ref, 'table_sizes': sizes,
                                 'wall_s': time.time() - t1}
        print(f'    min_count>={mc}: constructed ALL {acc["all"]["acc@1_pct"]:6.2f}% / lookup '
              f'{ref["all"]["acc@1_pct"]:6.2f}% (NOVEL {acc["novel"]["acc@1_pct"]:6.2f}% / '
              f'{ref["novel"]["acc@1_pct"]:6.2f}%)  sizes={sizes}', flush=True)

    report['wall_s'] = time.time() - t00
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f'[*] wrote {args.out} in {report["wall_s"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
