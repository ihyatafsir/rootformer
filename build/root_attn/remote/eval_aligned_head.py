#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_aligned_head.py -- like-for-like comparison of the CONTROL head and the ALIGNED head on the
SAME frozen hidden states, both label arms, plus the free closed-form readout ceilings.

Evaluated on exactly the 300 val windows used by every 20k run (starts_for(va4[1], 300, 1)):
    next-word arm : logits at t -> root_{t+1}   (the task the head is trained on)
    identity arm  : logits at t -> root_t       (the same head outputs, label shifted back)
    + length-1 echo on unique held-out val words
and, on the SAME tensors, ridge probes fitted on the 3000 train windows:
    probe next-word  -> the single-position LINEAR CEILING for the trained head's own task
    probe identity   -> the "is the root linearly present" number

Reuses /workspace/echo_test/echo_identity.py's own extraction/metric code so nothing is
re-implemented.  One GPU process.  Writes /workspace/head_fix/eval_aligned_head.json.

!!! CACHE MATTERS -- READ BEFORE QUOTING A NUMBER !!!
echo_identity.CACHE is /workspace/nrmp_cache_9490 (the MISALIGNED cache).  A head trained on the
ALIGNED cache must be evaluated on the ALIGNED cache; feeding it the misaligned prefix stream is
off-distribution and costs it ~0.70pp of next-word acc@1 (MEASURED: head_ALIGNED.pt scores
6.15% on the aligned cache but 5.448% on the legacy cache).  Pass --cache to select, e.g.

    python eval_aligned_head.py --cache /workspace/head_fix/nrmp_cache_9490_aligned
    python eval_aligned_head.py --cache /workspace/nrmp_cache_9490

and quote each head against its own cache.  The default is the LEGACY cache so that the CONTROL
head's published 5.718% is reproduced exactly.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

RELEASE = Path('/workspace/hf_v19_2_release')
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'models'))
sys.path.insert(0, '/workspace/echo_test')

HEADS = {
    'CONTROL': Path('/workspace/v18fix/head_CONTROL.pt'),
    'ALIGNED': Path('/workspace/head_fix/head_ALIGNED.pt'),
}
OUT = Path('/workspace/head_fix/eval_aligned_head.json')  # overridden by --out
T0 = time.time()


def log(*a):
    print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/workspace/nrmp_cache_9490',
                    help='cache to feed BOTH the backbone input and the head features')
    ap.add_argument('--out', default='/workspace/head_fix/eval_aligned_head.json')
    args = ap.parse_args()
    import echo_identity as E
    E.CACHE = Path(args.cache)
    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from safetensors.torch import load_file
    from nrmt_arch import RootformerNRMT, build_operator_table

    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json')
    vocab = V(bp)
    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16, hist=3, dropout=0.1,
                           use_features=True, feat_gate=False).to(device)
    model.load_state_dict(load_file(str(E.CKPT)), strict=False)
    for p in model.parameters():
        p.requires_grad = False
    model.nrmt_head.to(torch.float32)
    head = model.nrmt_head
    head.eval()
    model.eval()
    log('model ready')

    tr4 = [t.long() for t in torch.load(E.CACHE / 'train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(E.CACHE / 'val.pt', map_location='cpu')]

    def starts_for(t, n, seed):
        s = list(range(0, t.numel() - E.WIN - 1, E.STRIDE))
        np.random.default_rng(seed).shuffle(s)
        return sorted(s[:n])

    va_st, tr_st = starts_for(va4[1], 300, 1), starts_for(tr4[1], 3000, 0)

    def win(t, st):
        return torch.stack([t[j:j + E.WIN] for j in st])

    Tv, Wv, Pv, Sv = (win(va4[i], va_st) for i in (1, 2, 0, 3))
    Tr, Wr, Pr, Sr = (win(tr4[i], tr_st) for i in (1, 2, 0, 3))
    op_table = build_operator_table(vocab)
    try:
        from sibawayh_governor import SibawayhGovernor
        gov = SibawayhGovernor(vocab)
        op_ids_of = lambda P, R, W: torch.tensor(
            [gov.op_ids_for_ids(rr, ww, pp) for pp, rr, ww in zip(P.tolist(), R.tolist(),
                                                                 W.tolist())], dtype=torch.long)
        op_src = 'SibawayhGovernor'
    except Exception as e:                                          # pragma: no cover
        log(f'[warn] governor unavailable: {e!r}')
        op_ids_of = lambda P, R, W: op_table[R.clamp(min=0)]
        op_src = f'fallback: {e!r}'
    Ova, Otr = op_ids_of(Pv, Tv, Wv), op_ids_of(Pr, Tr, Wr)

    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec_t = torch.tensor(specials)

    Hva = E.extract_h(model, va4, va_st, device, 'val')
    Htr = E.extract_h(model, tr4, tr_st, device, 'train')
    Hva_w, Htr_w = Hva[:, :-1, :].contiguous(), Htr[:, :-1, :].contiguous()
    Rt_id, Rt_nw = Tv[:, :-1].contiguous(), Tv[:, 1:].contiguous()
    keep_id = ~torch.isin(Rt_id.reshape(-1), spec_t)
    keep_nw = ~torch.isin(Rt_nw.reshape(-1), spec_t)
    log(f'val radical positions identity n={int(keep_id.sum())} next-word n={int(keep_nw.sum())}')

    RES = {'headdir': str(Path('/workspace/head_fix')), 'operator_stream': op_src,
           'n_identity': int(keep_id.sum()), 'n_next_word': int(keep_nw.sum()),
           'marginals': {
               'identity': E.majority_rate(Rt_id.reshape(-1)[keep_id], vocab.num_roots),
               'next_word': E.majority_rate(Rt_nw.reshape(-1)[keep_nw], vocab.num_roots)},
           'heads': {}}

    # ---- ridge ceilings on the SAME tensors ------------------------------------------------
    Xtr = Htr_w.reshape(-1, Htr_w.shape[-1])
    Xv = Hva_w.reshape(-1, Hva_w.shape[-1])
    yv_id, yv_nw = Rt_id.reshape(-1), Rt_nw.reshape(-1)
    ytr_id, ytr_nw = Tr[:, :-1].reshape(-1), Tr[:, 1:].reshape(-1)
    ktr_id, ktr_nw = ~torch.isin(ytr_id, spec_t), ~torch.isin(ytr_nw, spec_t)
    W_id, lam_id = E.fit_ridge(Xtr[ktr_id], ytr_id[ktr_id], vocab.num_roots, device)
    W_nw, lam_nw = E.fit_ridge(Xtr[ktr_nw], ytr_nw[ktr_nw], vocab.num_roots, device)
    one = lambda n: torch.ones(n, dtype=torch.bool)
    RES['ridge_probe'] = {
        'identity': E.eval_ridge(W_id, Xv, yv_id, keep_id, vocab.num_roots, device,
                                 vocab.num_roots)['all'],
        'next_word': E.eval_ridge(W_nw, Xv, yv_nw, keep_nw, vocab.num_roots, device,
                                  vocab.num_roots)['all'],
        'lam_id': lam_id, 'lam_nw': lam_nw}
    log(f"  probe identity : {RES['ridge_probe']['identity']}")
    log(f"  probe next-word: {RES['ridge_probe']['next_word']}")

    # ---- both trained heads, both arms -----------------------------------------------------
    for tag, path in HEADS.items():
        if not path.exists():
            log(f'[skip] {tag}: {path} missing')
            continue
        sd = torch.load(str(path), map_location='cpu')
        r = head.load_state_dict(sd, strict=True)
        log(f'head {tag}: {path.name} ({len(sd)} tensors) {r}')
        wm = E.window_metrics(head, Hva_w, Tv[:, :-1], Ova[:, :-1], Wv[:, :-1], Pv[:, :-1],
                              Sv[:, :-1], Rt_id, Rt_nw, keep_id, keep_nw, device,
                              vocab.num_roots)
        entry = {'head_file': str(path),
                 'next_word': {k: v.out() for k, v in wm['next_word'].items()},
                 'identity': {k: v.out() for k, v in wm['identity'].items()}}
        # length-1 echo on unique held-out val words (the "echo" arm)
        vp, vr, vw, vs = E.unique_words(va4, vocab.num_roots, vocab.num_awzan,
                                        vocab.num_suffixes)
        kw = ~torch.isin(vr, spec_t)
        vp, vr, vw, vs = vp[kw], vr[kw], vw[kw], vs[kw]
        entry['echo_len1'] = E.identify_word_level(model, head, vp, vr, vw, vs, device,
                                                   vocab.num_roots, tag=f'{tag}/echo')
        RES['heads'][tag] = entry
        log(f"  [{tag}] next-word {entry['next_word']['all']}")
        log(f"  [{tag}] identity  {entry['identity']['all']}")
        log(f"  [{tag}] echo      {entry['echo_len1']['all']}")

    RES['peak_vram_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    RES['wall_s'] = round(time.time() - T0, 1)
    RES['cache_used'] = str(E.CACHE)
    Path(args.out).write_text(json.dumps(RES, indent=2))
    log(f'wrote {args.out}')


if __name__ == '__main__':
    main()
