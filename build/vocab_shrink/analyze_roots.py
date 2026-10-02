#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_roots.py -- CPU-only frequency analysis of the ALIGNED cache root stream, and the
construction of the SHRUNK root label map (top-K roots by TRAIN frequency).

READ-ONLY on nrmp_vocab.py / the blueprint / the checkpoints / the caches.
Writes ONLY under --out (default /workspace/vocab_shrink/).

The map is built from the TRAIN stream only (no val leakage).  Class index c in [0,K) is the
c-th most frequent real (non-special) root in train.  Nothing else changes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

RELEASE = Path('/workspace/hf_v19_2_release')
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'models'))

# MUST match nrmt_train.py exactly
WIN, STRIDE, CTX = 128, 64, 12
VAL_WINDOWS = 300
SPEC_CTX = CTX


def starts_for(t, n, seed):
    s = list(range(0, t.numel() - WIN - 1, STRIDE))
    np.random.default_rng(seed).shuffle(s)
    return sorted(s[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--out', default='/workspace/vocab_shrink')
    ap.add_argument('--topk', type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = RELEASE / 'data/rootformer_v12_arabic_blueprint.json'
    vocab = V(str(bp))
    N = vocab.num_roots

    special = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                      vocab.root2id['<PARTICLE>']} |
                     {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec = np.zeros(N, dtype=bool)
    spec[np.array(special, dtype=np.int64)] = True

    tr4 = [t.long() for t in torch.load(Path(args.cache) / 'train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(Path(args.cache) / 'val.pt', map_location='cpu')]
    tr_root = tr4[1].numpy().astype(np.int64)
    va_root = va4[1].numpy().astype(np.int64)

    # ---- the EXACT eval positions of every 20k arm -------------------------------------
    va_st = starts_for(va4[1], VAL_WINDOWS, 1)
    Rt_va = np.stack([va_root[j + 1:j + WIN] for j in va_st])          # [300,127]
    Rt_flat = Rt_va.reshape(-1)
    keep_all = ~spec[Rt_flat]
    n_eval, n_keep = int(Rt_flat.size), int(keep_all.sum())

    # ---- frequency over real-root train positions --------------------------------------
    tr_real = tr_root[~spec[tr_root]]
    cnt = np.bincount(tr_real, minlength=N).astype(np.int64)
    order = np.argsort(-cnt, kind='stable')                            # most frequent first
    topk = order[:args.topk]

    # ---- full-space marginal on the identical eval positions ----------------------------
    cnt_eval = np.bincount(Rt_flat[keep_all], minlength=N).astype(np.int64)
    full_marg_id = int(np.argmax(cnt_eval))
    full_marg = float(cnt_eval[full_marg_id] / n_keep)

    # ---- shrunk (DROP) space: keep only eval positions whose root is in top-k ------------
    in_topk = np.zeros(N, dtype=bool)
    in_topk[topk] = True
    keep_shrunk = keep_all & in_topk[Rt_flat]
    n_shrunk = int(keep_shrunk.sum())
    cnt_shr = np.bincount(Rt_flat[keep_shrunk], minlength=N).astype(np.int64)
    shr_marg_id = int(np.argmax(cnt_shr))
    shr_marg = float(cnt_shr[shr_marg_id] / max(n_shrunk, 1))
    coverage_eval = n_shrunk / max(n_keep, 1)

    # ---- shrunk (OTHER) space: every non-top-k real root -> one OTHER class ---------------
    other_cnt = int(n_keep - n_shrunk)
    other_marg = float(max(cnt_shr.max(), other_cnt) / max(n_keep, 1))

    # data-per-class, train side
    tr_real_cnt = np.bincount(tr_real, minlength=N).astype(np.int64)
    d_per_cls_full = tr_real.size / max(int((tr_real_cnt > 0).sum()), 1)
    d_per_cls_drop = int(tr_real_cnt[topk].sum()) / args.topk
    d_per_cls_other = tr_real.size / (args.topk + 1)

    # ---- forbidden set (Al-Khalil) intersected with the top-k ---------------------------
    forb_topk, forb_n = [], None
    try:
        from classical_governance_v2 import AlKhalilV2
        forb = sorted(AlKhalilV2(vocab).mask())
        forb_n = len(forb)
        fset = set(forb)
        forb_topk = [c for c, o in enumerate(topk.tolist()) if o in fset]
    except Exception as e:                                            # pragma: no cover
        print(f'[warn] forbidden-set unavailable: {e!r}')

    orig_to_class = np.full(N, -1, dtype=np.int64)
    orig_to_class[topk] = np.arange(args.topk, dtype=np.int64)
    topk_str = [vocab.roots_list[int(o)] for o in topk]
    topk_cnt = [int(cnt[int(o)]) for o in topk]
    topk_eval_cnt = [int(cnt_eval[int(o)]) for o in topk]

    meta = {
        'cache': str(args.cache),
        'checkpoint_md5_expected': '3335a3d39091535d0fdd047dca842715',
        'built_from': 'TRAIN stream only',
        'topk': args.topk,
        'num_roots_full': N,
        'n_train_real_root_tokens': int(tr_real.size),
        'n_val_real_root_tokens': int((~spec[va_root]).sum()),
        'eval_positions_total': n_eval,
        'eval_positions_kept_nonspecial': n_keep,
        'full_space_marginal': full_marg,
        'full_space_marginal_root': vocab.roots_list[full_marg_id],
        'full_space_marginal_id': full_marg_id,
        'topk_train_coverage': float(cnt[topk].sum() / tr_real.size),
        'eval_positions_in_topk': n_shrunk,
        'eval_coverage_in_topk': float(coverage_eval),
        'shrunk_drop_marginal': shr_marg,
        'shrunk_drop_marginal_root': vocab.roots_list[shr_marg_id],
        'shrunk_other_marginal': other_marg,
        'shrunk_other_class_share': float(other_cnt / max(n_keep, 1)),
        'data_per_class_full': float(d_per_cls_full),
        'data_per_class_drop': float(d_per_cls_drop),
        'data_per_class_other': float(d_per_cls_other),
        'forbidden_roots_full': forb_n,
        'forbidden_classes_in_topk': forb_topk,
        'topk_roots': topk_str,
        'topk_train_counts': topk_cnt,
        'topk_eval_counts': topk_eval_cnt,
        'class_to_orig': [int(o) for o in topk.tolist()],
        'specials': special,
    }
    (out / ('root_map_top%d.json' % args.topk)).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    torch.save({'class_to_orig': torch.tensor(topk.tolist(), dtype=torch.long),
                'orig_to_class': torch.tensor(orig_to_class, dtype=torch.long),
                'topk': args.topk},
               out / ('root_map_top%d.pt' % args.topk))

    print(json.dumps({k: v for k, v in meta.items()
                      if k not in ('topk_roots', 'topk_train_counts', 'topk_eval_counts',
                                   'class_to_orig', 'specials')}, indent=2, ensure_ascii=False))
    print('\ntop-15 shrunk classes (rank, root, train count, eval count):')
    for c in range(15):
        print(f'  {c:4d}  {topk_str[c]:>10s}  {topk_cnt[c]:9d}  {topk_eval_cnt[c]:7d}')
    print('\nfull-space top-10 eval classes:')
    for i in np.argsort(-cnt_eval)[:10]:
        print(f'  id {i:5d}  {vocab.roots_list[int(i)]:>10s}  {int(cnt_eval[i]):7d} '
              f'({100*cnt_eval[i]/n_keep:.2f}%)' + ('  [SPECIAL]' if spec[i] else ''))
    print(f'\nwrote {out}/root_map_top{args.topk}.json in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
