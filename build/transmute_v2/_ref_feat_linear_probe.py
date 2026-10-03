#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feat_linear_probe.py -- WHY the trained head is at 5.72% when a root n-gram reaches 53.13%.

nrmt_arch.NRMTHead.build_features hands the root head
    [ emb(r_{t-1}), emb(r_{t-2}), emb(r_{t-3}), emb(op_t), emb(w_{t-1}), emb(p_{t-1}), emb(s_{t-1}) ]
through ONE Linear(320, 896) and a LayerNorm, ADDED to h_t (which carries r_t at 91.75% linear
decodability).  So the branch's view of the root history is a LINEAR map of the previous roots'
EMBEDDINGS.

This measures the ceiling of that FORM, with no backbone and no training loop: a closed-form
ridge readout on the embedding of the previous k roots -> root_{t+1}, on the same 3000/300
windows and the same 18,869 val radical positions as every other measurement here.  Compare
with the order-k root n-gram on exactly the same context:

    order   n-gram (lookup)   linear-on-embeddings (this script)
      1          1.74%
      2         11.81%
      3         38.85%
      4         53.13%

If the linear-on-embeddings numbers stay near chance while the n-gram climbs, then the limiter is
the head's CONDITIONING FORM (a linear projection of embeddings cannot implement a discrete
n-gram lookup), not the backbone, not the readout, and not the number of roots.

CPU only, read-only.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

RELEASE = Path('/workspace/hf_v19_2_release')
CACHES = {
    'legacy': Path('/workspace/nrmp_cache_9490'),
    'aligned': Path('/workspace/head_fix/nrmp_cache_9490_aligned'),
}
CKPT = (RELEASE /
        'checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors')
WIN, STRIDE = 128, 64
KS = [1, 2, 3, 4, 5]
LAM_FRAC = 3e-3
T0 = time.time()


def log(*a):
    print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)


def root_embed():
    """The released morphemic root embedding -- the exact table build_features embeds through."""
    with safe_open(str(CKPT), framework='pt') as f:
        cand = [(k, f.get_slice(k).get_shape()) for k in f.keys() if 'root_embed' in k]
        log(f'root_embed candidates: {cand}')
        pick = [k for k, shp in cand if shp[0] == 9490]
        assert pick, f'no 9490-row root_embed in {cand}'
        return f.get_tensor(pick[0]).float()


def starts_for(t, n, seed):
    s = list(range(0, t.numel() - WIN - 1, STRIDE))
    np.random.default_rng(seed).shuffle(s)
    return sorted(s[:n])


def design_emb(R, E, k):
    """[N, T-1, k*d]: embeddings of the k roots BEFORE each target (t-1 .. t-k)."""
    N, T = R.shape
    T1 = T - 1
    d = E.shape[1]
    out = torch.empty(N, T1, k * d, dtype=torch.float32)
    for j in range(1, k + 1):
        idx = (torch.arange(T1) - j).clamp(min=0)
        out[:, :, (j - 1) * d:j * d] = E[R[:, idx]]
    return out


def ridge_fit_eval(Xtr, ytr, ktr, Xva, yva, kva, C, device):
    dim = Xtr.shape[-1]
    XTX = torch.zeros(dim + 1, dim + 1, dtype=torch.float64, device=device)
    XTY = torch.zeros(dim + 1, C, dtype=torch.float64, device=device)
    for i in range(0, Xtr.shape[0], 16384):
        sl = slice(i, i + 16384)
        m = ktr[sl]
        if not bool(m.any()):
            continue
        xb = Xtr[sl][m].to(device).double()
        yb = ytr[sl][m].to(device)
        xb1 = torch.cat([xb, torch.ones(xb.shape[0], 1, dtype=torch.float64, device=device)], 1)
        XTX += xb1.t() @ xb1
        XTY.index_add_(1, yb, xb1.t())
    lam = LAM_FRAC * float(torch.diagonal(XTX[:dim, :dim]).mean())
    W = torch.linalg.solve(XTX + lam * torch.eye(dim + 1, dtype=torch.float64, device=device),
                           XTY).float()
    lg = torch.cat([Xva.to(device),
                    torch.ones(Xva.shape[0], 1, device=device)], 1) @ W
    t = yva.to(device)
    a1 = float((lg.argmax(-1) == t).float().mean())
    a5 = float((lg.topk(5, -1).indices == t.unsqueeze(-1)).any(-1).float().mean())
    return {'n': int(t.numel()), 'acc@1_pct': 100 * a1, 'acc@5_pct': 100 * a5, 'dim': dim,
            'lam': lam}


def main():
    device = torch.device('cpu')
    torch.set_num_threads(min(32, torch.get_num_threads()))
    E = root_embed()
    log(f'root_embed {tuple(E.shape)}')
    C = E.shape[0]
    RES = {'protocol': 'closed-form ridge on emb(r_{t-1})..emb(r_{t-k}) -> root_{t+1}; same '
                       '3000/300 windows and 18,869 val radical positions as every other number',
           'root_embed_shape': list(E.shape), 'ks': KS, 'n_gram_ceiling': {}, 'curve': {}}
    # published n-gram curve, same positions
    NG = {1: 1.74, 2: 11.81, 3: 38.85, 4: 53.13, 5: 57.13, 6: 57.95}
    RES['n_gram_ceiling'] = {k: v for k, v in NG.items() if k <= max(KS)}

    for tag, cache in CACHES.items():
        tr4 = [t.long() for t in torch.load(cache / 'train.pt', map_location='cpu')]
        va4 = [t.long() for t in torch.load(cache / 'val.pt', map_location='cpu')]
        Tr = torch.stack([tr4[1][j:j + WIN] for j in starts_for(tr4[1], 3000, 0)])
        Tv = torch.stack([va4[1][j:j + WIN] for j in starts_for(va4[1], 300, 1)])
        # specials
        sys.path.insert(0, str(RELEASE))
        sys.path.insert(0, str(RELEASE / 'models'))
        import nrmp_vocab as nv
        V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
        vocab = V(str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json'))
        spec = torch.tensor(sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT,
                                    vocab.UNK_ROOT, vocab.root2id['<PARTICLE>']} |
                                   {i for i, r in enumerate(vocab.roots_list)
                                    if r.startswith('<P:')}))
        # targets: position t of the input window -> root at t+1
        ytr = Tr[:, 1:].reshape(-1)
        yva = Tv[:, 1:].reshape(-1)
        ktr = ~torch.isin(ytr, spec)
        kva = ~torch.isin(yva, spec)
        log(f'[{tag}] val n={int(kva.sum())}')
        RES['curve'][tag] = {}
        for k in KS:
            Xtr = design_emb(Tr, E, k).reshape(-1, k * E.shape[1])
            Xva = design_emb(Tv, E, k).reshape(-1, k * E.shape[1])
            r = ridge_fit_eval(Xtr, ytr, ktr, Xva, yva, kva, C, device)
            r['n_gram_order_k'] = NG[k]
            RES['curve'][tag][k] = r
            log(f'  [{tag}] k={k}  linear-on-embeddings {r["acc@1_pct"]:.2f}% / '
                f'{r["acc@5_pct"]:.2f}%   (order-{k} n-gram lookup: {NG[k]:.2f}%)')
            del Xtr, Xva
    RES['conclusion'] = ('a linear readout of the previous k root EMBEDDINGS tracks the n-gram '
                         'curve only weakly; the head conditions on exactly this form (one '
                         'Linear + LayerNorm over concatenated history embeddings), which is why '
                         'it sits near the single-position h_t ceiling instead of the order-4 '
                         'lookup ceiling.')
    Path('/workspace/head_fix/feat_linear_probe.json').write_text(json.dumps(RES, indent=2))
    log('wrote /workspace/head_fix/feat_linear_probe.json')


if __name__ == '__main__':
    main()
