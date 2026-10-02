#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alt_probe.py -- the SAME length-1 linear-probe protocol that gave 92.34 % on the morphemically
adapted trunk (echo_test/echo_probe_len1.py), run against an ARBITRARY base's hidden state, with
the root target supplied by the EXTERNAL morphemic annotation.

    question:  h(word)  ->  that word's root ?
    fit:       closed-form ridge on UNIQUE train words
    eval:      unique held-out words; and the subset whose (prefix,root,wazn,suffix) tuple never
               appears in train ("unseen words"), which is the number the 83.92 % refers to.

This is a pure inference measurement: it needs no training run, so it decides whether the root
cross-attention has anything to read on a fresh base BEFORE any GPU time is spent training.

`--base-transmuted` loads the released synthesis-trained checkpoint as the trunk (to reproduce
92.34 % and validate this script against the published number); the default loads the ORIGINAL
untouched Qwen2.5-0.5B WITHOUT the native root path.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

RELEASE = Path('/workspace/hf_v19_2_release')
ALT = Path('/workspace/alt_base')
WIN = 128


def ridge_fit(X, Y, num_classes, device, lambdas=(1e-1, 1, 10, 100, 1e3, 1e4, 1e5)):
    """Closed-form multi-class ridge, lambda chosen by an internal train/holdout split."""
    N, D = X.shape
    g = torch.Generator(device='cpu').manual_seed(0)
    perm = torch.randperm(N, generator=g)
    n_ho = max(1, N // 10)
    ho, tr = perm[:n_ho], perm[n_ho:]
    Xtr, Ytr = X[tr].to(device), Y[tr].to(device)
    Xho, Yho = X[ho].to(device), Y[ho].to(device)
    Y1 = torch.nn.functional.one_hot(Ytr, num_classes).float()
    XtX = Xtr.T @ Xtr
    XtY = Xtr.T @ Y1
    best = (None, -1.0, None)
    for lam in lambdas:
        A = XtX + lam * torch.eye(D, device=device)
        W = torch.linalg.solve(A, XtY)
        acc = float((Xho @ W).argmax(-1).eq(Yho).float().mean())
        if acc > best[1]:
            best = (W, acc, lam)
    # refit on everything with the chosen lambda
    Xa, Ya = X.to(device), Y.to(device)
    Y1a = torch.nn.functional.one_hot(Ya, num_classes).float()
    A = Xa.T @ Xa + best[2] * torch.eye(D, device=device)
    W = torch.linalg.solve(A, Xa.T @ Y1a)
    return W, best[2], best[1]


@torch.no_grad()
def eval_ridge(W, X, y, num_classes):
    if X.shape[0] == 0:
        return {'n': 0, 'acc@1_pct': float('nan'), 'acc@5_pct': float('nan')}
    logits = X @ W
    k = min(5, num_classes)
    top5 = logits.topk(k, dim=-1).indices
    a1 = float(top5[:, 0].eq(y).float().mean()) * 100
    a5 = float((top5 == y.unsqueeze(-1)).any(-1).float().mean()) * 100
    return {'n': int(X.shape[0]), 'acc@1_pct': round(a1, 4), 'acc@5_pct': round(a5, 4)}


def load_matching(model, sd, tag='ckpt'):
    """Load only tensors whose shape matches: the released awzan142 checkpoint predates the
    current blueprint (9868-token / 9114-root vs 10052 / 9490), so the embedding tables differ in
    ROW COUNT.  Every other tensor must match exactly; a non-shape difference is a hard error."""
    ms = model.state_dict()
    keep, skipped = {}, []
    for k, v in sd.items():
        if k in ms and tuple(ms[k].shape) == tuple(v.shape):
            keep[k] = v
        elif k in ms:
            skipped.append((k, tuple(v.shape), tuple(ms[k].shape)))
        else:
            skipped.append((k, tuple(v.shape), None))
    missing = [k for k in ms if k not in keep and not k.startswith(('nrmt_head.', 'nrmp_head.'))]
    model.load_state_dict(keep, strict=False)
    print(f'[*] {tag}: loaded {len(keep)}/{len(sd)} tensors; skipped {len(skipped)}', flush=True)
    for k, a, b in skipped[:12]:
        print(f'    SKIP {k}: ckpt{a} vs model{b}', flush=True)
    print(f'[*] {tag}: model tensors NOT covered by the checkpoint: {len(missing)}', flush=True)
    for k in missing[:12]:
        print(f'    MISSING {k} {tuple(ms[k].shape)}', flush=True)
    return keep, skipped, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/workspace/alt_base/cache/rootqwen')
    ap.add_argument('--base', default='Qwen/Qwen2.5-0.5B')
    ap.add_argument('--ckpt', default=str(RELEASE / 'checkpoints/'
                    'rootformer_v19_2_synthesis_ar_backbone.awzan142.safetensors'))
    ap.add_argument('--base-transmuted', action='store_true',
                    help='load the released morphemic checkpoint as the trunk (reproduces 92.34)')
    ap.add_argument('--tag', default='orig')
    ap.add_argument('--max-train-words', type=int, default=400000)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    args.out = args.out or str(ALT / f'probe_{args.tag}.json')

    T0 = time.time()
    dev = torch.device('cuda')
    sys.path.insert(0, str(RELEASE)); sys.path.insert(0, str(RELEASE / 'models'))
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V(str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json'))
    R, A, SF, PF = vocab.num_roots, vocab.num_awzan, vocab.num_suffixes, vocab.num_prefixes

    tr_w = {n: torch.load(Path(args.cache) / f'{n}_train.pt', map_location='cpu')
            for n in ('word', 'tok', 'soff', 'wtokoff', 'wlen', 'wstart')}
    va_w = {n: torch.load(Path(args.cache) / f'{n}_val.pt', map_location='cpu')
            for n in ('word', 'tok', 'soff', 'wtokoff', 'wlen', 'wstart')}

    def unique_words(d, cap=None, seed=0):
        """One row per distinct (p,r,w,s) word tuple; returns that row's first occurrence index."""
        P, Rw, Wz, S = (d['word'][i].long() for i in (0, 1, 2, 3))
        key = (((P * R + Rw) * A + Wz) * SF + S)
        u, inv = torch.unique(key, return_inverse=True)
        # first occurrence of each unique key (input is in stream order, so this is the first)
        firstkey = torch.full((u.numel(),), key.numel(), dtype=torch.long)
        ar = torch.arange(key.numel())
        firstkey = firstkey.scatter_reduce(0, inv, ar, reduce='amin')
        if cap is not None and u.numel() > cap:
            g = torch.Generator().manual_seed(seed)
            sel = torch.sort(torch.randperm(u.numel(), generator=g)[:cap]).values
            u, firstkey = u[sel], firstkey[sel]
        s_ = u % SF; q = u // SF
        w = q % A; q = q // A
        r = q % R; p = q // R
        return p, r, w, s_, u, firstkey

    tp, tr_, tw, ts, tk, tidx = unique_words(tr_w, cap=args.max_train_words)
    vp, vr, vw, vs, vk, vidx = unique_words(va_w)
    specials = set([vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                    vocab.root2id['<PARTICLE>']]) | \
        {i for i, x in enumerate(vocab.roots_list) if x.startswith('<P:')}
    mt = ~torch.isin(tr_, torch.tensor(sorted(specials)))
    mv = ~torch.isin(vr, torch.tensor(sorted(specials)))
    tp, tr_, tw, ts, tk, tidx = tp[mt], tr_[mt], tw[mt], ts[mt], tk[mt], tidx[mt]
    vp, vr, vw, vs, vk, vidx = vp[mv], vr[mv], vw[mv], vs[mv], vk[mv], vidx[mv]
    seen = torch.isin(vk, tk)
    print(f'[{time.time()-T0:.1f}s] unique train words {tp.numel()} | val words {vp.numel()} '
          f'(seen {int(seen.sum())}, unseen {int((~seen).sum())})', flush=True)

    # ---- trunk -------------------------------------------------------------------------
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT
    base = UnifiedRootformerV12(str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json'),
                                args.base, str(dev), torch.bfloat16).to(dev)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, dev, torch.bfloat16).to(dev)
    model = RootformerNRMT(flash, vocab, dev, torch.bfloat16, hist=3, dropout=0.0,
                           use_features=True, feat_gate=False).to(dev)
    if args.base_transmuted:
        from safetensors.torch import load_file
        sd = load_file(args.ckpt)
        load_matching(model, sd, Path(args.ckpt).name)
        print('[*] trunk = RELEASED morphemic checkpoint', flush=True)
    else:
        # PURE untouched base: no native root score bias / coverage / governance
        n = 0
        for layer in model.backbone.layers:
            layer.self_attn.active_root_ids = None
            layer.self_attn.active_wazn_ids = None
            n += 1
        print(f'[*] trunk = ORIGINAL {args.base}, native root path disabled on {n} layers',
              flush=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print(f'[{time.time()-T0:.1f}s] model ready', flush=True)

    def window_tokens(d, idxs):
        """For each word index, the tokens of its SENTENCE plus the word's last-token position.

        Everything is expressed in the GLOBAL token stream via the cumulative word lengths
        (`wlen`) -- `wtokoff` is a per-SENTENCE offset, so mixing it with a different sentence's
        base (as earlier versions did) yields positions that are off by whole sentences and run
        past the end of the gathered tensor.
        """
        wstart = d['wstart'].long(); wlen = d['wlen'].long(); tok = d['tok'].long()
        cs = torch.zeros(wlen.numel() + 1, dtype=torch.long)
        torch.cumsum(wlen, 0, out=cs[1:])
        first = cs[idxs]
        last = first + wlen[idxs] - 1
        sent = torch.searchsorted(wstart, idxs, right=True) - 1
        sstart = cs[wstart[sent]]
        nxt = torch.searchsorted(wstart, wstart[sent] + 1, right=True)
        has_nxt = nxt < wstart.numel()
        send = torch.where(has_nxt, cs[wstart[nxt.clamp(max=wstart.numel() - 1)]],
                           torch.full_like(sstart, int(tok.numel())))
        send = torch.maximum(send, last + 1)
        return tok, sstart, send, last

    @torch.no_grad()
    def extract(d, idxs, tag, bs=64):
        tok, sstart, send, last = window_tokens(d, idxs)
        out = []
        order = torch.argsort(send - sstart)
        for s in range(0, idxs.numel(), bs):
            sel = order[s:s + bs]
            ss, se, ll = sstart[sel], send[sel], last[sel]
            se = torch.maximum(se, ll + 1)
            T = int((se - ss).max())
            ids = torch.zeros(len(sel), T, dtype=torch.long)
            for i in range(len(sel)):
                n_ = int(se[i] - ss[i])
                ids[i, :n_] = tok[ss[i]:se[i]]
            ids = ids.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = model.backbone(input_ids=ids)
                h = model.final_norm(o.last_hidden_state)
            li = (ll - ss).to(dev)
            out.append(h.gather(1, li.unsqueeze(-1).expand(-1, -1, h.shape[-1]))[:, 0].float().cpu())
        H = torch.cat(out)
        # restore the caller's order
        inv = torch.empty_like(order); inv[order] = torch.arange(order.numel())
        print(f'    h[{tag}] {tuple(H.shape)}', flush=True)
        return H[inv]

    Htr = extract(tr_w, tidx, 'train-words')
    Hva = extract(va_w, vidx, 'val-words')

    W, lam, ho_acc = ridge_fit(Htr, tr_, R, dev)
    res = {'meta': {'tag': args.tag, 'base': args.base, 'trunk': 'transmuted' if
                    args.base_transmuted else 'original-untransmuted',
                    'n_train_words': int(tp.numel()), 'n_val_words': int(vp.numel()),
                    'n_seen': int(seen.sum()), 'n_unseen': int((~seen).sum()),
                    'ridge_lambda': lam, 'ridge_internal_holdout_acc_pct': round(ho_acc * 100, 3),
                    'peak_vram_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
                    'wall_s': round(time.time() - T0, 1)}}
    ones = torch.ones(vp.numel(), dtype=torch.bool)
    res['val_all'] = eval_ridge(W, Hva, vr, R)
    res['val_unseen_words'] = eval_ridge(W, Hva[~seen], vr[~seen], R)
    res['val_seen_words'] = eval_ridge(W, Hva[seen], vr[seen], R)
    res['train_all'] = eval_ridge(W, Htr, tr_, R)
    for k in ('val_all', 'val_unseen_words', 'val_seen_words', 'train_all'):
        print(f'  probe len-1 {k:18s} {res[k]}', flush=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
