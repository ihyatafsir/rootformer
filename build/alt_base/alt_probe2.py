#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alt_probe2.py -- decodability probe on the ORIGINAL (untransmuted) base, minimal and defensive.

Question
--------
The morphemically adapted trunk exposes a word's OWN root from its final hidden state at
**92.34 %** acc@1 (83.92 % on words whose (p,r,w,s) tuple never occurs in train)
(/workspace/echo_test/echo_probe_len1.json).  Is that decodability a property of the BASE
Qwen2.5-0.5B, or was it created by the morphemic adaptation + synthesis training?

Method (same protocol as the reference)
---------------------------------------
  * the root of every word comes from the EXTERNAL morphemic annotation already aligned to the
    base tokenizer by character offset (`build_root_cache.py`), so the base keeps its own vocab;
  * h(word) = the base's last-layer hidden state at the word's LAST subword token, read from the
    sentence the word actually occurs in;
  * a closed-form ridge readout is fit on unique TRAIN words and scored on unique held-out words,
    plus the subset whose (p,r,w,s) tuple is unseen in train.
This uses plain HF Qwen2ForCausalLM -- no custom trunk, no morphemic embedding, no native root
branch -- so nothing but the base model's own representation can influence the number.
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


def load_word_stream(d, keys=('word', 'tok', 'soff', 'wtokoff', 'wlen', 'wstart')):
    return {n: torch.load(Path(d) / f'{n}.pt', map_location='cpu') for n in keys}


def unique_words(d, R, A, SF, cap=None, seed=0):
    P, Rw, Wz, S = (d['word'][i].long() for i in (0, 1, 2, 3))
    key = (((P * R + Rw) * A + Wz) * SF + S)
    u, inv = torch.unique(key, return_inverse=True)
    firstkey = torch.full((u.numel(),), key.numel(), dtype=torch.long)
    firstkey = firstkey.scatter_reduce(0, inv, torch.arange(key.numel()), reduce='amin')
    if cap is not None and u.numel() > cap:
        g = torch.Generator().manual_seed(seed)
        sel = torch.sort(torch.randperm(u.numel(), generator=g)[:cap]).values
        u, firstkey = u[sel], firstkey[sel]
    s_ = u % SF; q = u // SF
    w = q % A; q = q // A
    r = q % R; p = q // R
    return p, r, w, s_, u, firstkey


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/workspace/alt_base/cache/rootqwen')
    ap.add_argument('--base', default='Qwen/Qwen2.5-0.5B')
    ap.add_argument('--hf-home', default='/workspace/alt_base/cache')
    ap.add_argument('--tag', default='orig')
    ap.add_argument('--max-train-words', type=int, default=400000)
    ap.add_argument('--max-len', type=int, default=320)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    args.out = args.out or str(ALT / f'probe_{args.tag}.json')
    import os
    os.environ['HF_HOME'] = args.hf_home
    T0 = time.time()
    def log(*a):
        print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)

    sys.path.insert(0, str(RELEASE)); sys.path.insert(0, str(RELEASE / 'models'))
    import nrmp_vocab as nv
    from transformers import AutoTokenizer, AutoModelForCausalLM

    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V(str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json'))
    R, A, SF = vocab.num_roots, vocab.num_awzan, vocab.num_suffixes
    specials = set([vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                    vocab.root2id['<PARTICLE>']]) | \
        {i for i, x in enumerate(vocab.roots_list) if x.startswith('<P:')}
    spec_t = torch.tensor(sorted(specials))
    log(f'vocab roots={R} awzan={A} suffixes={SF} specials={len(specials)}')

    tokz = AutoTokenizer.from_pretrained(args.base, cache_dir=str(Path(args.hf_home) / 'hub'))
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                cache_dir=str(Path(args.hf_home) / 'hub'))
    model = model.to('cuda').eval()
    VS = int(model.get_input_embeddings().weight.shape[0])
    log(f'base {args.base}: embedding rows={VS} tokenizer_len={len(tokz)}')

    # cache filenames are <name>_<split>.pt
    def load_split(split):
        return {n: torch.load(Path(args.cache) / f'{n}_{split}.pt', map_location='cpu')
                for n in ('word', 'tok', 'soff', 'wtokoff', 'wlen', 'wstart')}
    tr = load_split('train'); va = load_split('val')
    tok_all = tr['tok'].long(); tok_all_v = va['tok'].long()
    log(f'token stream: train {tok_all.numel()} (max id {int(tok_all.max())}) | '
        f'val {tok_all_v.numel()} (max id {int(tok_all_v.max())})')
    assert int(tok_all.max()) < VS and int(tok_all_v.max()) < VS, 'token id outside embedding'
    assert int(tok_all.min()) >= 0 and int(tok_all_v.min()) >= 0, 'negative token id'

    tp, tr_, tw, ts, tk, tidx = unique_words(tr, R, A, SF, cap=args.max_train_words)
    vp, vr, vw, vs, vk, vidx = unique_words(va, R, A, SF)
    mt = ~torch.isin(tr_, spec_t); mv = ~torch.isin(vr, spec_t)
    tp, tr_, tw, ts, tk, tidx = tp[mt], tr_[mt], tw[mt], ts[mt], tk[mt], tidx[mt]
    vp, vr, vw, vs, vk, vidx = vp[mv], vr[mv], vw[mv], vs[mv], vk[mv], vidx[mv]
    seen = torch.isin(vk, tk)
    log(f'unique train words {tp.numel()} | val words {vp.numel()} '
        f'(seen {int(seen.sum())}, unseen {int((~seen).sum())})')

    def spans(d, tokstream, idxs):
        wlen = d['wlen'].long(); wstart = d['wstart'].long()
        cs = torch.zeros(wlen.numel() + 1, dtype=torch.long)
        torch.cumsum(wlen, 0, out=cs[1:])
        first = cs[idxs]; last = first + wlen[idxs] - 1
        sent = torch.searchsorted(wstart, idxs, right=True) - 1
        sstart = cs[wstart[sent]]
        nxt = torch.searchsorted(wstart, wstart[sent] + 1, right=True)
        has = nxt < wstart.numel()
        send = torch.where(has, cs[wstart[nxt.clamp(max=wstart.numel() - 1)]],
                           torch.full_like(sstart, int(tokstream.numel())))
        send = torch.maximum(send, last + 1)
        assert int((sstart > first).sum()) == 0, 'sentence start after word start'
        assert int((last >= send).sum()) == 0, 'word end past sentence end'
        assert int(send.max()) <= tokstream.numel(), 'span past token stream'
        return sstart, send, last

    @torch.no_grad()
    def extract(d, tokstream, idxs, tag):
        ss, se, ll = spans(d, tokstream, idxs)
        lens = (se - ss)
        order = torch.argsort(lens)
        D = int(model.config.hidden_size)
        out = torch.zeros(idxs.numel(), D, dtype=torch.float32)
        for s in range(0, idxs.numel(), args.bs):
            sel = order[s:s + args.bs]
            Lr = lens[sel]
            T = int(Lr.max())
            assert T <= args.max_len, f'{tag}: sentence span {T} > max_len {args.max_len}'
            ids = torch.zeros(len(sel), T, dtype=torch.long)
            for i in range(len(sel)):
                a = int(ss[sel[i]]); b = int(se[sel[i]])
                n_ = b - a
                assert n_ > 0 and n_ <= T, (n_, T, a, b)
                ids[i, :n_] = tokstream[a:b]
            assert int(ids.max()) < VS and int(ids.min()) >= 0
            idx_last = (ll[sel] - ss[sel])
            assert int(idx_last.max()) < T and int(idx_last.min()) >= 0, 'last index out of range'
            o = model(input_ids=ids.cuda(), output_hidden_states=True)
            h = o.hidden_states[-1]
            g = h.gather(1, idx_last.cuda().view(-1, 1, 1).expand(-1, 1, h.shape[-1]))[:, 0]
            out[sel] = g.float().cpu()
            if s == 0:
                log(f'    {tag}: T={T} bs={len(sel)} ok')
        log(f'    h[{tag}] {tuple(out.shape)}')
        return out

    Htr = extract(tr, tok_all, tidx, 'train-words')
    Hva = extract(va, tok_all_v, vidx, 'val-words')
    log(f'peak_vram={torch.cuda.max_memory_allocated()/2**20:.0f}MiB')

    # ---- ridge, chunked so no N x C one-hot is ever materialised -------------------------
    # The GPU is shared with two resident arms (~21 GB), so a 244k x 9490 one-hot (15.6 GiB) is
    # not available.  XtX and XtY are accumulated in chunks; the fitted readout is identical.
    D = Htr.shape[1]
    def acc_chunks(Wm, X, y):
        correct = 0; tot = 0
        for i in range(0, X.shape[0], 8192):
            xb = X[i:i + 8192].cuda(); yb = y[i:i + 8192].cuda()
            correct += int((xb @ Wm).argmax(-1).eq(yb).sum()); tot += int(yb.numel())
            del xb, yb
        return 100.0 * correct / max(tot, 1)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(Htr.shape[0], generator=g)
    n_ho = max(1, Htr.shape[0] // 10)
    ho, trn = perm[:n_ho], perm[n_ho:]
    def gram(idx):
        XtX = torch.zeros(D, D, device='cuda'); XtY = torch.zeros(D, R, device='cuda')
        for i in range(0, idx.numel(), 8192):
            sl = idx[i:i + 8192]
            xb = Htr[sl].cuda()
            yb = tr_[sl].cuda()
            XtX += xb.T @ xb
            XtY.index_add_(1, yb, xb.T)
            del xb, yb
        return XtX, XtY
    XtX, XtY = gram(trn)
    eye = torch.eye(D, device='cuda')
    Xh, Yh = Htr[ho], tr_[ho]
    best = (None, -1.0, None)
    for lam in (1e-1, 1, 10, 100, 1e3, 1e4, 1e5):
        Wm = torch.linalg.solve(XtX + lam * eye, XtY)
        a = acc_chunks(Wm, Xh, Yh)
        log(f'  lambda={lam:g} internal-holdout acc={a:.3f}%')
        if a > best[1]:
            best = (Wm, a, lam)
    Wm, ho_acc, lam = best
    log(f'ridge: lambda={lam} internal-holdout acc={ho_acc:.3f}%')

    def evaluate(X, y, mask, name):
        Xc, yc = X[mask], y[mask]
        if Xc.shape[0] == 0:
            log(f'  {name}: n=0'); return {'n': 0}
        lg = []
        for i in range(0, Xc.shape[0], 8192):
            lg.append((Xc[i:i + 8192].cuda() @ Wm).cpu())
        lg = torch.cat(lg)
        k = min(5, R)
        t5 = lg.topk(k, dim=-1).indices
        a1 = float(t5[:, 0].eq(yc).float().mean()) * 100
        a5 = float((t5 == yc.unsqueeze(-1)).any(-1).float().mean()) * 100
        log(f'  {name}: n={int(Xc.shape[0])} acc@1={a1:.2f}% acc@5={a5:.2f}%')
        return {'n': int(Xc.shape[0]), 'acc@1_pct': round(a1, 4), 'acc@5_pct': round(a5, 4)}

    # marginal baselines on the same samples
    def majority(y, mask):
        yy = y[mask]
        if yy.numel() == 0:
            return float('nan')
        counts = torch.bincount(yy, minlength=R).float()
        return round(100.0 * float(counts.max()) / float(yy.numel()), 4)

    ones = torch.ones(vp.numel(), dtype=torch.bool)
    RES = {'meta': {'tag': args.tag, 'base': args.base,
                    'trunk': 'ORIGINAL untransmuted Qwen2.5-0.5B (plain HF, no root branch)',
                    'n_train_words': int(tp.numel()), 'n_val_words': int(vp.numel()),
                    'n_seen': int(seen.sum()), 'n_unseen': int((~seen).sum()),
                    'ridge_lambda': lam, 'internal_holdout_acc_pct': round(ho_acc * 100, 3),
                    'peak_vram_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
                    'wall_s': round(time.time() - T0, 1)}}
    RES['marginals'] = {
        'val_all': majority(vr, torch.ones(vp.numel(), dtype=torch.bool)),
        'val_unseen_words': majority(vr, ~seen),
        'val_seen_words': majority(vr, seen),
        'train_all': majority(tr_, torch.ones(tp.numel(), dtype=torch.bool))}

    RES['val_all'] = evaluate(Hva, vr, ones, 'val_all')
    RES['val_unseen_words'] = evaluate(Hva, vr, ~seen, 'val_unseen_words')
    RES['val_seen_words'] = evaluate(Hva, vr, seen, 'val_seen_words')
    RES['train_all'] = evaluate(Htr, tr_, torch.ones(tp.numel(), dtype=torch.bool), 'train_all')
    Path(args.out).write_text(json.dumps(RES, ensure_ascii=False, indent=2))
    log(f'wrote {args.out}')


if __name__ == '__main__':
    main()
