#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmp_adapter.py -- make NRMT explicitly autoregressive over the root stream.

Diagnosis
---------
Our 0.5B Rootformer NRMP head reads only the backbone hidden state h_t and reaches ~7.5%
next-root acc@1, while a 4-gram over the ROOT stream reaches ~37.6%. The next-root task is
therefore largely a function of the recent ROOT HISTORY, which the head never receives
explicitly -- it is only implicit in h_t.

Fix
---
Train small adapters on frozen backbone features and compare, strictly on the leak-free
train.pt / val.pt split (val files were never trained on):

  head_adapter   : h_t                     -> root   (baseline: does an MLP on h beat the shipped head?)
  hist_adapter   : h_t + E(r_{t-1},r_{t-2}) -> root  (explicit root-history conditioning)
  bigram_only    : E(r_{t-1},r_{t-2})       -> root  (no neural context at all)
  ngram4         : count-based 4-gram fitted on train.pt

If hist_adapter > head_adapter and > ngram4, explicit root-history conditioning is the fix.

Usage:
  python nrmp_adapter.py --checkpoint /workspace/nrmp_trained_final.safetensors --steps 3000
"""
import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

WIN = 128
STRIDE = 64


class Adapter(nn.Module):
    def __init__(self, d_h, n_roots, d_e=64, d_ctx=512, use_h=True, use_hist=True):
        super().__init__()
        self.use_h, self.use_hist = use_h, use_hist
        d_in = (d_h if use_h else 0) + (2 * d_e if use_hist else 0)
        self.e1 = nn.Embedding(n_roots, d_e) if use_hist else None
        self.e2 = nn.Embedding(n_roots, d_e) if use_hist else None
        self.net = nn.Sequential(
            nn.Linear(d_in, d_ctx), nn.SiLU(), nn.LayerNorm(d_ctx),
            nn.Linear(d_ctx, d_ctx), nn.SiLU(),
            nn.Linear(d_ctx, n_roots),
        )

    def forward(self, h, p1, p2):
        parts = []
        if self.use_h:
            parts.append(h)
        if self.use_hist:
            parts += [self.e1(p1), self.e2(p2)]
        return self.net(torch.cat(parts, dim=-1))


@torch.no_grad()
def backbone_hidden(model, p, r, w, s, dtype=torch.bfloat16):
    emb = model.morphemic_embed(p, r, w, s)
    out = model.backbone(inputs_embeds=emb)
    return model.final_norm(out.last_hidden_state)


def make_windows(streams, split, win=WIN, stride=STRIDE, device='cpu'):
    p, r, w, s = streams
    n = r.numel()
    starts = list(range(0, max(1, n - win - 1), stride))
    P = torch.stack([p[i:i + win] for i in starts]).to(device)
    R = torch.stack([r[i:i + win] for i in starts]).to(device)
    W = torch.stack([w[i:i + win] for i in starts]).to(device)
    S = torch.stack([s[i:i + win] for i in starts]).to(device)
    return P, R, W, S


def build_ngram(stream, order):
    counts = [Counter() for _ in range(order + 1)]
    ctxs = [defaultdict(Counter) for _ in range(order + 1)]
    for i, tok in enumerate(stream):
        counts[1][tok] += 1
        for o in range(2, order + 1):
            if i >= o - 1:
                ctxs[o][tuple(stream[i - o + 1:i])][tok] += 1
    return counts, ctxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--cache', default='/workspace/nrmp_cache')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-train-windows', type=int, default=900)
    ap.add_argument('--max-val-windows', type=int, default=150)
    ap.add_argument('--out', default='/workspace/nrmp_adapter.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from nrmp_run import load_engine
    model, vocab, device, _ = load_engine(ckpt=args.checkpoint)
    for prm in model.parameters():
        prm.requires_grad = False
    model.eval()

    specials = {vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                vocab.root2id['<PARTICLE>']}
    specials |= {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')}
    spec_t = torch.tensor(sorted(specials), device=device)

    cache = Path(args.cache)
    tr = [t.long() for t in torch.load(cache / 'train.pt', map_location='cpu')]
    va = [t.long() for t in torch.load(cache / 'val.pt', map_location='cpu')]
    print(f'[*] train tokens={tr[1].numel()}  val tokens={va[1].numel()} (file-level split)')

    print('[*] building count-based 4-gram on train only...', flush=True)
    counts, ctxs = build_ngram(tr[1].tolist(), 4)

    # ---------------- features ----------------
    PW, RW, WW, SW = make_windows(tr, 'train', device='cpu')
    idx = torch.randperm(PW.shape[0])[:args.max_train_windows]
    PW, RW, WW, SW = PW[idx], RW[idx], WW[idx], SW[idx]
    VP, VR, VW, VS = make_windows(va, 'val', device='cpu')
    vidx = torch.randperm(VP.shape[0])[:args.max_val_windows]
    VP, VR, VW, VS = VP[vidx], VR[vidx], VW[vidx], VS[vidx]
    print(f'[*] windows: train={PW.shape[0]} val={VP.shape[0]} of length {WIN}', flush=True)

    def feats(P, R, W, S, tag):
        hs, p1s, p2s, tgts = [], [], [], []
        bs = 16
        with torch.no_grad():
            for i in range(0, P.shape[0], bs):
                p = P[i:i + bs].to(device); r = R[i:i + bs].to(device)
                w = W[i:i + bs].to(device); s = S[i:i + bs].to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    h = backbone_hidden(model, p, r, w, s)
                h = h[:, :-1, :].float().cpu()
                tgt = r[:, 1:].cpu()
                prev1 = r[:, :-1].cpu()
                prev2 = torch.roll(r, 1, dims=1)[:, :-1].cpu()
                prev2[:, 0] = vocab.BOS_ROOT
                hs.append(h); p1s.append(prev1); p2s.append(prev2); tgts.append(tgt)
        print(f'    {tag}: hidden {torch.cat(hs).shape}', flush=True)
        return torch.cat(hs), torch.cat(p1s), torch.cat(p2s), torch.cat(tgts)

    Htr, Atr, Btr, Ttr = feats(PW, RW, WW, SW, 'train')
    Hva, Ava, Bva, Tva = feats(VP, VR, VW, VS, 'val')

    def flat(*xs):
        return [x.reshape(-1, x.shape[-1]) if x.dim() == 3 else x.reshape(-1) for x in xs]

    Htr, Atr, Btr, Ttr = flat(Htr, Atr, Btr, Ttr)
    Hva, Ava, Bva, Tva = flat(Hva, Ava, Bva, Tva)
    keep_tr = ~torch.isin(Ttr, spec_t.cpu())
    keep_va = ~torch.isin(Tva, spec_t.cpu())
    print(f'[*] radical positions: train={int(keep_tr.sum())} val={int(keep_va.sum())}', flush=True)

    def evaluate(logits, tgt, keep, topk=(1, 5)):
        lg, tg = logits[keep], tgt[keep]
        out = {'n': int(tg.numel())}
        for k in topk:
            kk = min(k, lg.shape[-1])
            out[f'acc@{k}'] = float((lg.topk(kk, -1).indices == tg.unsqueeze(-1)).any(-1)
                                    .float().mean().item())
        ce = float(F.cross_entropy(lg, tg).item())
        out['ppl'] = math.exp(min(ce, 25.0))
        return out

    results = {}

    # ---------------- count-based baselines ----------------
    uni = counts[1]
    top_uni = max(((k, v) for k, v in uni.items() if k not in specials), key=lambda kv: kv[1])[0]
    def ngram_pred(hist):
        for o in (4, 3, 2):
            if len(hist) >= o - 1:
                c = ctxs[o].get(tuple(hist[-(o - 1):]))
                if c:
                    return c.most_common(1)[0][0]
        return top_uni

    hist_list = []
    for win_i in range(VP.shape[0]):
        r = VR[win_i].tolist()
        for t in range(1, len(r)):
            hist_list.append(r[:t])
    tg_flat = Tva
    n = 0; hit1 = 0; hit5 = 0
    for hist, tgt in zip(hist_list, tg_flat.tolist()):
        if tgt in specials:
            continue
        n += 1
        pred = ngram_pred(hist)
        if pred == tgt:
            hit1 += 1
        for o in (4, 3, 2):
            if len(hist) >= o - 1:
                c = ctxs[o].get(tuple(hist[-(o - 1):]))
                if c:
                    if tgt in [x for x, _ in c.most_common(5)]:
                        hit5 += 1
                    break
    results['ngram4'] = {'n': n, 'acc@1': hit1 / max(n, 1), 'acc@5': hit5 / max(n, 1)}
    print(f"\n[*] ngram4 (train-fitted, val) acc@1={100*hit1/max(n,1):.2f}% "
          f"acc@5={100*hit5/max(n,1):.2f}%", flush=True)

    # ---------------- adapters ----------------
    d_h = Htr.shape[-1]
    for name, use_h, use_hist in (('head_adapter', True, False),
                                  ('bigram_only', False, True),
                                  ('hist_adapter', True, True)):
        torch.manual_seed(0)
        ad = Adapter(d_h, vocab.num_roots, use_h=use_h, use_hist=use_hist).to(device)
        opt = torch.optim.AdamW(ad.parameters(), lr=args.lr, weight_decay=0.01)
        nb = Htr.shape[0]
        t0 = time.time()
        for step in range(1, args.steps + 1):
            bi = torch.randint(0, nb, (args.batch_size,))
            h = Htr[bi].to(device)
            a = Atr[bi].to(device); b = Btr[bi].to(device)
            tg = Ttr[bi].to(device)
            valid = ~torch.isin(tg, spec_t)
            if valid.sum() == 0:
                continue
            lg = ad(h, a, b)
            loss = F.cross_entropy(lg[valid], tg[valid])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 500 == 0:
                print(f'    {name} step {step}/{args.steps} loss {float(loss):.4f} '
                      f'({step/(time.time()-t0):.1f} it/s)', flush=True)
        ad.eval()
        with torch.no_grad():
            lg = []
            for i in range(0, Hva.shape[0], 4096):
                lg.append(ad(Hva[i:i + 4096].to(device),
                             Ava[i:i + 4096].to(device),
                             Bva[i:i + 4096].to(device)).float().cpu())
            lg = torch.cat(lg)
        m = evaluate(lg, Tva, keep_va)
        results[name] = m
        print(f'  {name:<16} acc@1 {100*m["acc@1"]:6.2f}%  acc@5 {100*m["acc@5"]:6.2f}%  '
              f'ppl {m["ppl"]:7.1f}  (n={m["n"]})', flush=True)

    json.dump(results, open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')
    print('\nleak-free: adapters and counts fit on train.pt, scored on val.pt (held-out FILES)')


if __name__ == '__main__':
    main()
