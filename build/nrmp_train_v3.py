#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmp_train_v3.py -- fix NRMP training and watch root PPL on ARABIC ONLY, leak-free.

Two changes over the shipped training:

1. CONTEXT-NOVEL EVALUATION (the honesty fix).
   The file-level split is not clean: ~15% of val 13-grams also occur in train. PPL measured on
   all val positions can therefore improve by memorisation. Every metric here is reported twice:
       ALL val     -- the usual number
       NOVEL only  -- positions whose 12-root context never occurs in the train stream
   Root PPL on NOVEL positions is the number that must go down for real progress.

2. EXPLICIT ROOT-HISTORY CONDITIONING (the capability fix).
   The shipped head is a single Linear(896 -> 9114) reading only h_t; a 4-gram over the root
   stream reaches ~33% acc@1 against its ~7.5%. Three heads are trained and compared on frozen
   backbone features:
       h_only    : h_t                              (reproduces the shipped architecture)
       history   : [h_t ; E(r-1) ; E(r-2) ; E(r-3)] (explicit root-history conditioning)
       bigram    : [E(r-1) ; E(r-2) ; E(r-3)]       (no neural context at all)
   plus the count-based interpolated n-gram as a reference.

Usage:
  python nrmp_train_v3.py --checkpoint /workspace/nrmp_trained_final.safetensors \
      --train-windows 4000 --val-windows 400 --steps 12000
"""
import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

WIN, STRIDE, CTX = 128, 64, 12
HASH_B = np.uint64(1000003)


# --------------------------------------------------------------------------------------
# leak-free novelty via 64-bit rolling hashes over the root stream
# --------------------------------------------------------------------------------------
def rolling_hashes(stream, n, B=HASH_B):
    a = np.asarray(stream, dtype=np.uint64)
    L = len(a) - n + 1
    if L <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(L, dtype=np.uint64)
    for k in range(n):
        h = h * B + a[k:k + L]
    return h


def hash_ctx(ctx, B=HASH_B):
    h = np.uint64(0)
    for x in ctx:
        h = h * B + np.uint64(x)
    return h


class NoveltyIndex:
    def __init__(self, train_stream, ctx=CTX):
        t0 = time.time()
        self.ctx = ctx
        self.hashes = np.unique(rolling_hashes(train_stream, ctx))
        print(f'    novelty index: {len(self.hashes)} distinct {ctx}-gram hashes '
              f'({time.time()-t0:.0f}s)', flush=True)

    def is_novel(self, ctx_list):
        h = hash_ctx(ctx_list)
        i = np.searchsorted(self.hashes, h)
        return not (i < len(self.hashes) and self.hashes[i] == h)


# --------------------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, d_h, n_roots, d_e=64, d_ctx=512, use_h=True, hist=0):
        super().__init__()
        self.use_h, self.hist = use_h, hist
        d_in = (d_h if use_h else 0) + hist * d_e
        self.emb = nn.ModuleList([nn.Embedding(n_roots, d_e) for _ in range(hist)]) if hist else None
        self.net = nn.Sequential(
            nn.Linear(d_in, d_ctx), nn.SiLU(), nn.LayerNorm(d_ctx),
            nn.Linear(d_ctx, d_ctx), nn.SiLU(),
            nn.Linear(d_ctx, n_roots),
        )

    def forward(self, h, prevs):
        parts = []
        if self.use_h:
            parts.append(h)
        for i in range(self.hist):
            parts.append(self.emb[i](prevs[i]))
        return self.net(torch.cat(parts, dim=-1))


@torch.no_grad()
def extract(model, streams, starts, device, tag):
    P, R, W, S = streams
    hs, tgs, p1, p2, p3 = [], [], [], [], []
    bs = 16
    t0 = time.time()
    for i in range(0, len(starts), bs):
        idx = starts[i:i + bs]
        p = torch.stack([P[j:j + WIN] for j in idx]).to(device)
        r = torch.stack([R[j:j + WIN] for j in idx]).to(device)
        w = torch.stack([W[j:j + WIN] for j in idx]).to(device)
        s = torch.stack([S[j:j + WIN] for j in idx]).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb = model.morphemic_embed(p, r, w, s)
            o = model.backbone(inputs_embeds=emb)
            h = model.final_norm(o.last_hidden_state)
        hs.append(h[:, :-1, :].float().cpu())
        tgs.append(r[:, 1:].cpu())
        p1.append(r[:, :-1].cpu())
        p2.append(torch.roll(r, 1, dims=1)[:, :-1].cpu())
        p3.append(torch.roll(r, 2, dims=1)[:, :-1].cpu())
        p2[-1][:, 0] = 0
        p3[-1][:, :2] = 0
    seq = torch.stack([R[j:j + WIN] for j in starts])   # for context novelty
    print(f'    {tag}: {len(starts)} windows in {time.time()-t0:.0f}s', flush=True)
    return (torch.cat(hs), torch.cat(tgs), torch.cat(p1), torch.cat(p2), torch.cat(p3), seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--cache', default='/workspace/nrmp_cache')
    ap.add_argument('--train-windows', type=int, default=4000)
    ap.add_argument('--val-windows', type=int, default=400)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default='/workspace/nrmp_v3.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from nrmp_run import load_engine
    model, vocab, device, _ = load_engine(ckpt=args.checkpoint)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    specials = {vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                vocab.root2id['<PARTICLE>']}
    specials |= {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')}
    spec_t = torch.tensor(sorted(specials))

    tr4 = [t.long() for t in torch.load(Path(args.cache) / 'train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(Path(args.cache) / 'val.pt', map_location='cpu')]
    print(f'[*] train {tr4[1].numel()} | val {va4[1].numel()} root tokens', flush=True)
    print('[*] building leak-free novelty index from the TRAIN root stream', flush=True)
    nov = NoveltyIndex(tr4[1].tolist())

    def starts_for(t, n, seed):
        s = list(range(0, t.numel() - WIN - 1, STRIDE))
        rng = np.random.default_rng(seed)
        rng.shuffle(s)
        return sorted(s[:n])

    tr_starts = starts_for(tr4[1], args.train_windows, 0)
    va_starts = starts_for(va4[1], args.val_windows, 1)

    print('[*] extracting frozen backbone features', flush=True)
    Htr, Ttr, Atr, Btr, Ctr, _ = extract(model, tr4, tr_starts, device, 'train')
    Hva, Tva, Ava, Bva, Cva, Vseq = extract(model, va4, va_starts, device, 'val')

    def flat(x):
        return x.reshape(-1, x.shape[-1]) if x.dim() == 3 else x.reshape(-1)

    Htr, Ttr, Atr, Btr, Ctr = map(flat, (Htr, Ttr, Atr, Btr, Ctr))
    Hva, Tva, Ava, Bva, Cva = map(flat, (Hva, Tva, Ava, Bva, Cva))

    # -------- context novelty per val position --------
    novel_mask = torch.zeros(Tva.numel(), dtype=torch.bool)
    n_win = Vseq.shape[0]
    per = WIN - 1
    for w in range(n_win):
        seg = Vseq[w].tolist()
        for t in range(CTX, WIN):
            pos = w * per + (t - 1)
            if pos < novel_mask.numel():
                novel_mask[pos] = nov.is_novel(seg[t - CTX:t])
    keep = ~torch.isin(Tva, spec_t)
    novel_keep = keep & novel_mask
    print(f'    radical positions: all={int(keep.sum())} novel={int(novel_keep.sum())} '
          f'({100*int(novel_keep.sum())/max(int(keep.sum()),1):.1f}%)', flush=True)

    # -------- n-gram reference, tuned on train-dev --------
    tr_roots = tr4[1].tolist()
    ctxs = [defaultdict(Counter) for _ in range(5)]
    counts = Counter(tr_roots)
    for i in range(len(tr_roots)):
        for o in range(2, 5):
            if i >= o - 1:
                ctxs[o][tuple(tr_roots[i - o + 1:i])][tr_roots[i]] += 1

    def ng_dist(hist, lam=(0.1, 0.2, 0.3, 0.4)):
        d = defaultdict(float)
        tot = sum(counts.values())
        for r, c in counts.items():
            if r not in specials:
                d[r] += lam[0] * c / tot
        for o in range(2, 5):
            if len(hist) >= o - 1:
                c = ctxs[o].get(tuple(hist[-(o - 1):]))
                if c:
                    s = sum(c.values())
                    for r, k in c.items():
                        if r not in specials:
                            d[r] += lam[o - 1] * k / s
        s = sum(d.values())
        return {r: v / s for r, v in d.items()} if s > 0 else {}

    print('\n[*] n-gram reference (train-fitted, interpolated 1-4gram)')
    for tag, mask in (('ALL_val', keep), ('NOVEL_only', novel_keep)):
        idx = torch.nonzero(mask).flatten()
        import random as _r
        _r.Random(0).shuffle(idx.tolist())
        sel = idx[:min(4000, idx.numel())]
        hits1 = hits5 = n = 0
        ce = 0.0
        # rebuild histories for the selected flat positions
        for pos in sel.tolist():
            w = pos // per
            t = (pos % per) + 1
            hist = Vseq[w, :t].tolist()
            tgt = int(Tva[pos])
            d = ng_dist(hist)
            if not d:
                continue
            n += 1
            ranked = sorted(d.items(), key=lambda kv: -kv[1])
            if tgt == ranked[0][0]:
                hits1 += 1
            if tgt in [r for r, _ in ranked[:5]]:
                hits5 += 1
            ce += -math.log(max(d.get(tgt, 1e-12), 1e-12))
        if n:
            print(f'  {tag:<12} acc@1 {100*hits1/n:6.2f}%  acc@5 {100*hits5/n:6.2f}%  '
                  f'ppl {math.exp(min(ce/n,25)):8.1f}  (n={n})', flush=True)

    # -------- train the three heads --------
    d_h = Htr.shape[-1]
    results = {}
    for name, use_h, hist in (('h_only', True, 0), ('bigram', False, 3), ('history', True, 3)):
        torch.manual_seed(0)
        head = Head(d_h, vocab.num_roots, use_h=use_h, hist=hist).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
        nb = Htr.shape[0]
        t0 = time.time()
        for step in range(1, args.steps + 1):
            bi = torch.randint(0, nb, (args.batch_size,))
            h = Htr[bi].to(device)
            tg = Ttr[bi].to(device)
            prevs = [Atr[bi].to(device), Btr[bi].to(device), Ctr[bi].to(device)][:hist]
            valid = ~torch.isin(tg, spec_t.to(device))
            if valid.sum() == 0:
                continue
            lg = head(h, prevs)
            loss = F.cross_entropy(lg[valid], tg[valid])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 3000 == 0:
                print(f'    {name} step {step}/{args.steps} loss {float(loss):.4f} '
                      f'({step/(time.time()-t0):.0f} it/s)', flush=True)
        head.eval()
        with torch.no_grad():
            logits = []
            for i in range(0, Hva.shape[0], 8192):
                prevs = [Ava[i:i+8192].to(device), Bva[i:i+8192].to(device),
                         Cva[i:i+8192].to(device)][:hist]
                logits.append(head(Hva[i:i+8192].to(device), prevs).float().cpu())
            LG = torch.cat(logits)
        row = {}
        for tag, mask in (('ALL_val', keep), ('NOVEL_only', novel_keep)):
            lg, tg = LG[mask], Tva[mask]
            if tg.numel() == 0:
                continue
            k1 = int((lg.argmax(-1) == tg).sum())
            k5 = int((lg.topk(5, -1).indices == tg.unsqueeze(-1)).any(-1).sum())
            ce = float(F.cross_entropy(lg, tg))
            row[tag] = {'n': int(tg.numel()), 'acc@1': k1 / tg.numel(),
                        'acc@5': k5 / tg.numel(), 'ppl': math.exp(min(ce, 25.0))}
            print(f'  {name:<10} {tag:<12} acc@1 {100*k1/tg.numel():6.2f}%  '
                  f'acc@5 {100*k5/tg.numel():6.2f}%  ppl {row[tag]["ppl"]:8.1f}', flush=True)
        results[name] = row

    json.dump(results, open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')
    print('\nAll metrics are on the file-level held-out split; NOVEL_only additionally requires')
    print('that the 12-root context never occurs in train, so memorisation cannot help.')


if __name__ == '__main__':
    main()
