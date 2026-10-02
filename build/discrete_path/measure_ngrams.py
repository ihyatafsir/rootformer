#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_ngrams.py -- size the discrete root-history pathway from the data, and measure the
EXPRESSIBILITY CEILING of each candidate table form directly.

For every order k = 1..4 it reports
  * the number of DISTINCT order-k root contexts in the train stream
  * the order-k lookup accuracy on the exact 300 val windows / 18,869 radical positions /
    15,046 NOVEL positions used by every 20k run (backoff to shorter orders, unigram fallback)
  * the same under a HASHED table with B = 2**18 .. 2**24 buckets (contexts that collide are
    merged and vote) -- this is the ceiling of the `--ngram-features hash` arm, because a hashed
    table cannot separate two contexts that share a bucket.

The collision-merged number is the point of this script: it says, before any GPU time, what
accuracy a hash table of a given size can possibly express on this data.

CPU only.  Read-only on the caches.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CACHE = Path(sys.argv[1] if len(sys.argv) > 1 else '/workspace/head_fix/nrmp_cache_9490_aligned')
WIN, STRIDE, CTX = 128, 64, 12
HASH_B = np.uint64(1000003)          # the TRAINER's own novelty hash (nrmt_train.py)
OUT = Path('/workspace/discrete_path')
OUT.mkdir(parents=True, exist_ok=True)
KMAX = 4
HASH_BS = (1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24)
_HASH_MOD = (1 << 40) - 1
_HASH_A = 1000003


def rolling_hashes(stream, n, B=HASH_B):
    """Verbatim from nrmt_train.py: the novelty hash of every n-block, uint64 wraparound."""
    a = np.asarray(stream, dtype=np.uint64)
    L = len(a) - n + 1
    if L <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(L, dtype=np.uint64)
    for k in range(n):
        h = h * B + a[k:k + L]
    return h


def hash_ctx(ctx, B=1000003):
    """Verbatim from nrmt_train.py: the same arithmetic without numpy overflow warnings."""
    h = 0
    for x in ctx:
        h = (h * B + int(x)) & ((1 << 64) - 1)
    return np.uint64(h)


def codes_for(stream, k, num_roots):
    """Exact order-k code of the block STARTING at i (stream[i..i+k-1]).

    9490**4 = 8.11e15 < 2**63, so k <= 4 is exact for this vocabulary.
    """
    a = np.asarray(stream, dtype=np.int64)
    L = len(a) - k + 1
    code = np.zeros(max(L, 0), dtype=np.int64)
    for j in range(k):
        code = code * num_roots + a[j:j + L]
    return code


def hash64(codes):
    h = np.mod(codes, _HASH_MOD).astype(np.int64)
    h = ((h ^ (h >> 21)) * _HASH_A) % _HASH_MOD
    h = ((h ^ (h >> 17)) * _HASH_A) % _HASH_MOD
    return h


def topk_map(keys, targets, k):
    """Group (key, target); return (sorted unique keys, top-k targets, top-k counts, group size)."""
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
    gcount = np.bincount(p_grp, weights=pcount).astype(np.int64)   # TOTAL occurrences
    ndist = np.bincount(p_grp).astype(np.int64)                    # DISTINCT targets seen
    return gh, top, cts, gcount, ndist


def starts_for(t, n, seed):
    s = list(range(0, len(t) - WIN - 1, STRIDE))
    np.random.default_rng(seed).shuffle(s)
    return sorted(s[:n])


def val_keys(W, k, num_roots):
    """[nwin*(WIN-1)] code of the k roots ENDING at input position i (target i+1).

    A block ending at i starts at i-k+1, so hs[i] = rolling[i-k+1]; i < k-1 has no context and
    is marked invalid by the caller (exactly as nrmt_train.py's window layout).
    """
    nwin = W.shape[0]
    hs = np.zeros((nwin, WIN - 1), dtype=np.int64)
    for w in range(nwin):
        rh = codes_for(W[w], k, num_roots)
        hs[w, k - 1:] = rh[:WIN - k]
    return hs.reshape(-1)


def main():
    t0 = time.time()
    tr = [t.long() for t in torch.load(CACHE / 'train.pt', map_location='cpu')]
    va = [t.long() for t in torch.load(CACHE / 'val.pt', map_location='cpu')]
    tr_r = tr[1].numpy().astype(np.int64)
    va_r = va[1].numpy().astype(np.int64)
    print(f'[*] cache={CACHE} train_root={len(tr_r)} val_root={len(va_r)} t={time.time()-t0:.0f}s',
          flush=True)

    sys.path.insert(0, '/workspace/hf_v19_2_release')
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    C = int(vocab.num_roots)
    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec = np.array(specials)
    print(f'[*] num_roots={C} n_specials={len(specials)} distinct_train_ids={len(np.unique(tr_r))}',
          flush=True)

    va_st = starts_for(va_r, 300, 1)
    tr_st = starts_for(tr_r, 3000, 0)
    Tv = np.stack([va_r[j:j + WIN] for j in va_st])
    Tr = np.stack([tr_r[j:j + WIN] for j in tr_st])
    tv = Tv[:, 1:].reshape(-1)
    keep_v = ~np.isin(tv, spec)

    # NOVEL: the 12-root context never occurs in train (EXACTLY nrmt_train.py's Novelty: the
    # trainer's own base-1000003 uint64 rolling hash, so the 15,046 figure is reproduced)
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
    print(f'[*] val radical positions all={n_all} novel={n_nov} '
          f'({100*n_nov/max(n_all,1):.1f}%) | majority={100*np.bincount(tv[keep_v], minlength=C).max()/n_all:.3f}%',
          flush=True)

    report = {'cache': str(CACHE), 'train_root_tokens': int(len(tr_r)), 'num_roots': C,
              'val_positions_all': n_all, 'val_positions_novel': n_nov, 'orders': {}}

    uni = np.bincount(tr_r, minlength=C)
    uni_top5 = np.argsort(-uni)[:5]
    state = {'keep_v': keep_v, 'keep_novel': keep_novel, 'Tv': Tv, 'tv': tv, 'uni_top5': uni_top5}

    def evaluate(maps, kmax, keys_val, min_count=1):
        tgt = tv
        N = len(tgt)
        pred = np.full((N, 5), -1, dtype=np.int64)
        resolved = np.zeros(N, dtype=bool)
        used = np.zeros(N, dtype=np.int64)
        for k in sorted(maps, reverse=True):
            if k > kmax or k not in maps:
                continue
            hs = keys_val[k]
            gh, top, gcount = maps[k]['vocab'], maps[k]['tops'], maps[k]['count']
            idx = np.clip(np.searchsorted(gh, hs), 0, len(gh) - 1)
            found = (gh[idx] == hs) & (gcount[idx] >= min_count)
            if k == 1:
                found = found | True
            valid = np.tile(np.arange(WIN - 1) >= k - 1, Tv.shape[0])
            take = found & valid & ~resolved
            pred[take] = top[idx[take]]
            used[take] = k
            resolved |= take
        if (~resolved).any():
            pred[~resolved] = uni_top5
            used[~resolved] = 0
        out = {}
        for tag, keep in (('all', keep_v), ('novel', keep_novel)):
            p, t = pred[keep], tgt[keep]
            out[tag] = {'n': int(keep.sum()),
                        'acc@1_pct': 100 * float((p[:, 0] == t).mean()),
                        'acc@5_pct': 100 * float((p == t[:, None]).any(-1).mean())}
        h = np.bincount(used[keep_v], minlength=kmax + 1)
        out['backoff_hist_all'] = {int(i): int(v) for i, v in enumerate(h) if v}
        return out

    for k in range(1, KMAX + 1):
        t1 = time.time()
        tr_codes = codes_for(tr_r, k, C)
        tgt = tr_r[k:]
        L = min(len(tr_codes), len(tgt))
        uniq = np.unique(tr_codes[:L])
        row = {'distinct_train': int(len(uniq)), 'valid_frac_train': float(L / len(tr_r))}
        print(f'  k={k}: distinct_train={len(uniq):>9d}  ({time.time()-t1:.0f}s)', flush=True)
        report['orders'][k] = row

    # every candidate table, fitted on the FULL train stream
    variants = {'exact': None}
    for B in HASH_BS:
        variants[f'hash2^{B.bit_length()-1}'] = B
    for name, B in variants.items():
        t1 = time.time()
        maps = {}
        distinct = {}
        for k in range(1, KMAX + 1):
            tr_codes = codes_for(tr_r, k, C)
            tgt = tr_r[k:]
            L = min(len(tr_codes), len(tgt))
            keys = tr_codes[:L] if B is None else (hash64(tr_codes[:L]) % B)
            gh, top, cts, gcount, ndist = topk_map(keys, tgt[:L], 5)
            maps[k] = {'vocab': gh, 'tops': top, 'cts': cts, 'count': gcount, 'ndist': ndist}
            distinct[k] = int(len(gh))
        keys_val = {}
        for k in range(1, KMAX + 1):
            kv = val_keys(Tv, k, C)
            keys_val[k] = kv if B is None else (hash64(kv) % B)
        # per-order (no backoff) and backoff curves
        res = {'distinct': distinct, 'per_order': {}, 'backoff': {}}
        for K in range(1, KMAX + 1):
            res['backoff'][K] = evaluate({k: maps[k] for k in range(1, K + 1)}, K, keys_val)
        for k in range(1, KMAX + 1):
            res['per_order'][k] = evaluate({k: maps[k]}, k, keys_val)
        report.setdefault('variants', {})[name] = res
        print(f'  [{name}] distinct={distinct} ({time.time()-t1:.0f}s)', flush=True)
        for K in range(1, KMAX + 1):
            b = res['backoff'][K]
            print(f'      order<={K}: ALL {b["all"]["acc@1_pct"]:6.2f}%  NOVEL '
                  f'{b["novel"]["acc@1_pct"]:6.2f}%  (hist {b["backoff_hist_all"]})', flush=True)

    # ---- pruning sweeps (table size vs accuracy): the vocab design knobs ------------------
    maps4 = {}
    for k in range(1, KMAX + 1):
        tr_codes = codes_for(tr_r, k, C)
        tgt = tr_r[k:]
        L = min(len(tr_codes), len(tgt))
        gh, top, cts, gcount, ndist = topk_map(tr_codes[:L], tgt[:L], 5)
        maps4[k] = {'vocab': gh, 'tops': top, 'cts': cts, 'count': gcount, 'ndist': ndist}
    keys_val4 = {k: val_keys(Tv, k, C) for k in range(1, KMAX + 1)}

    # how much of the signal is DETERMINISTIC (exactly one target ever observed)?
    report['purity'] = {}
    for k in range(1, KMAX + 1):
        nd, gc = maps4[k]['ndist'], maps4[k]['count']
        det = nd == 1
        report['purity'][k] = {
            'contexts': int(len(nd)), 'deterministic_contexts': int(det.sum()),
            'deterministic_frac': float(det.mean()),
            'positions_in_deterministic': int(gc[det].sum()),
            'positions_total': int(gc.sum())}
        print(f"  purity k={k}: {det.sum()}/{len(nd)} contexts deterministic "
              f"({100*det.mean():.2f}%), covering {100*gc[det].sum()/gc.sum():.2f}% of train "
              f"positions", flush=True)

    report['min_count_sweep'] = {}
    for mc in (1, 2, 3, 5, 10):
        r = evaluate(maps4, KMAX, keys_val4, min_count=mc)
        sizes = {k: int((maps4[k]['count'] >= mc).sum()) for k in range(1, KMAX + 1)}
        r['table_sizes'] = sizes
        r['params_k4'] = {f'd{d}': sizes[4] * d for d in (8, 16, 32)}
        report['min_count_sweep'][mc] = r
        print(f'  [occurrences>={mc}] order<=4: ALL {r["all"]["acc@1_pct"]:6.2f}%  '
              f'NOVEL {r["novel"]["acc@1_pct"]:6.2f}%  sizes={sizes} '
              f'order4 params@d16={sizes[4]*16/1e6:.1f}M', flush=True)

    report['deterministic_only_sweep'] = {}
    for tag, sel in (('ndist==1', lambda m: m['ndist'] == 1),
                     ('ndist<=2', lambda m: m['ndist'] <= 2),
                     ('ndist<=4', lambda m: m['ndist'] <= 4)):
        maps5 = {}
        for k in range(1, KMAX + 1):
            m = maps4[k]
            keepsel = sel(m)
            maps5[k] = {'vocab': m['vocab'][keepsel], 'tops': m['tops'][keepsel],
                        'count': m['count'][keepsel]}
        r = evaluate(maps5, KMAX, keys_val4)
        sizes = {k: int(len(maps5[k]['vocab'])) for k in range(1, KMAX + 1)}
        r['table_sizes'] = sizes
        report['deterministic_only_sweep'][tag] = r
        print(f'  [{tag}] order<=4: ALL {r["all"]["acc@1_pct"]:6.2f}%  NOVEL '
              f'{r["novel"]["acc@1_pct"]:6.2f}%  sizes={sizes}', flush=True)

    report['wall_s'] = time.time() - t0
    (OUT / 'measure_ngrams.json').write_text(json.dumps(report, indent=2))
    print(f'[*] wrote {OUT/"measure_ngrams.json"} in {report["wall_s"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
