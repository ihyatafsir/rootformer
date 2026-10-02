#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmp_eval_clean.py -- memorisation-controlled evaluation of next-root prediction.

The file-level split is NOT clean: ~15% of val 13-grams and ~42% of val 7-grams also occur in
train, because formulaic classical phrasing recurs across works. Any model with a lookup table
can therefore score by memorisation.

This script defines CONTEXT-NOVEL positions: a val position is novel only if its 12-root context
never occurs anywhere in the training stream. Metrics are then reported twice:

    all val      -- the usual number (partly memorisation)
    novel only   -- the honest generalisation number

and for four systems:
    ngram        -- interpolated 4/3/2/1-gram over the root stream, weights tuned on a dev slice
    neural       -- our 0.5B Rootformer NRMP head
    fused        -- log-linear interpolation  P ~ P_neural^alpha * P_ngram^(1-alpha)
    fused+masks  -- the same, with the corrected Al-Khalil/Sibawayh hard masks applied

Usage:
  python nrmp_eval_clean.py --checkpoint /workspace/nrmp_trained_final.safetensors --windows 250
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

WIN, STRIDE, CTX = 128, 64, 12


def build_ngram(stream, order):
    counts = [Counter() for _ in range(order + 1)]
    ctxs = [defaultdict(Counter) for _ in range(order + 1)]
    for i, tok in enumerate(stream):
        counts[1][tok] += 1
        for o in range(2, order + 1):
            if i >= o - 1:
                ctxs[o][tuple(stream[i - o + 1:i])][tok] += 1
    return counts, ctxs


def ngram_prob(counts, ctxs, order, hist, specials, lam):
    """Interpolated model: lam = (l1,l2,l3,l4) summing to 1."""
    d = defaultdict(float)
    tot1 = sum(counts[1].values()) or 1
    for r, c in counts[1].items():
        if r not in specials:
            d[r] += lam[0] * c / tot1
    for o in range(2, order + 1):
        if len(hist) >= o - 1:
            c = ctxs[o].get(tuple(hist[-(o - 1):]))
            if c:
                tot = sum(c.values())
                for r, k in c.items():
                    if r not in specials:
                        d[r] += lam[o - 1] * k / tot
    s = sum(d.values())
    return {r: v / s for r, v in d.items()} if s > 0 else {}


def score(records, prob_fn, topk=(1, 5)):
    n = 0
    hit = {k: 0 for k in topk}
    ce = 0.0
    for tgt, payload in records:
        d = prob_fn(tgt, payload)
        if not d:
            continue
        n += 1
        ranked = sorted(d.items(), key=lambda kv: -kv[1])
        for k in topk:
            if tgt in [r for r, _ in ranked[:k]]:
                hit[k] += 1
        ce += -math.log(max(d.get(tgt, 1e-12), 1e-12))
    if n == 0:
        return None
    return {'n': n, **{f'acc@{k}': hit[k] / n for k in topk}, 'ppl': math.exp(min(ce / n, 25.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--cache', default='/workspace/nrmp_cache')
    ap.add_argument('--windows', type=int, default=250)
    ap.add_argument('--order', type=int, default=4)
    ap.add_argument('--out', default='/workspace/nrmp_eval_clean.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from nrmp_run import load_engine
    model, vocab, device, _ = load_engine(ckpt=args.checkpoint)

    specials = {vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                vocab.root2id['<PARTICLE>']}
    specials |= {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')}
    spec_t = torch.tensor(sorted(specials), device=device)

    tr4 = [t.long() for t in torch.load(Path(args.cache) / 'train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(Path(args.cache) / 'val.pt', map_location='cpu')]
    tr = tr4[1].tolist()          # root stream (context model + novelty)
    va = va4[1].tolist()
    print(f'[*] train {len(tr)} tokens | val {len(va)} tokens', flush=True)

    # ---- context-novel mask -------------------------------------------------------------
    print(f'[*] indexing train {CTX}-gram contexts...', flush=True)
    train_ctx = set()
    for i in range(len(tr) - CTX):
        train_ctx.add(tuple(tr[i:i + CTX]))
    print(f'    distinct train {CTX}-gram contexts: {len(train_ctx)}')

    # ---- n-gram counts ------------------------------------------------------------------
    print(f'[*] building {args.order}-gram counts...', flush=True)
    counts, ctxs = build_ngram(tr, args.order)

    # ---- neural features on a sample of val windows -------------------------------------
    starts = list(range(0, len(va) - WIN - 1, STRIDE))
    import random
    random.Random(0).shuffle(starts)
    starts = sorted(starts[:args.windows])
    print(f'[*] neural forward over {len(starts)} val windows', flush=True)

    recs_all, recs_novel = [], []
    with torch.no_grad():
        for w0 in starts:
            seg = va[w0:w0 + WIN]
            # the real four morphemic streams (prefix 26 / root 9114 / wazn 130 / suffix 22)
            pv = va4[0][w0:w0 + WIN].unsqueeze(0).to(device)
            rv = va4[1][w0:w0 + WIN].unsqueeze(0).to(device)
            wv = va4[2][w0:w0 + WIN].unsqueeze(0).to(device)
            sv = va4[3][w0:w0 + WIN].unsqueeze(0).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                emb = model.morphemic_embed(pv, rv, wv, sv)
                o = model.backbone(inputs_embeds=emb)
                h = model.final_norm(o.last_hidden_state)
                lg = model.nrmp_head.root_head(h)[0].float()
            rs = seg
            for t in range(CTX, WIN):
                tgt = rs[t]
                if tgt in specials:
                    continue
                ctx = rs[t - CTX:t]
                novel = tuple(ctx) not in train_ctx
                recs_all.append((tgt, (lg[t].cpu(), ctx)))
                if novel:
                    recs_novel.append((tgt, (lg[t].cpu(), ctx)))
            del lg, h, o, emb

    print(f'[*] scored positions: all={len(recs_all)}  novel={len(recs_novel)} '
          f'({100*len(recs_novel)/max(len(recs_all),1):.1f}% novel)', flush=True)

    # ---- tune n-gram lambdas on a dev slice of the ALL set (novel untouched) ------------
    dev = recs_all[:len(recs_all) // 4]
    best = (None, -1.0)
    for l4 in (0.2, 0.4, 0.6, 0.8):
        for l3 in (0.0, 0.2, 0.4):
            for l2 in (0.0, 0.2, 0.4):
                l1 = 1.0 - l4 - l3 - l2
                if l1 < 0:
                    continue
                lam = (l1, l2, l3, l4)
                m = score(dev, lambda t, p, lam=lam: ngram_prob(counts, ctxs, args.order, p[1],
                                                               specials, lam))
                if m and m['acc@1'] > best[1]:
                    best = (lam, m['acc@1'])
    lam = best[0] or (0.1, 0.2, 0.3, 0.4)
    print(f'[*] tuned n-gram lambdas (1,2,3,4) = {tuple(round(x,3) for x in lam)} '
          f'(dev acc@1 {100*best[1]:.2f}%)', flush=True)

    def neural_prob(lg):
        finite = torch.isfinite(lg)
        if not finite.any():
            return {}
        lg2 = lg.clone(); lg2[~finite] = -1e4
        probs = F.softmax(lg2, dim=-1)
        return {r: float(probs[r]) for r in torch.nonzero(finite).flatten().tolist()
                if float(probs[r]) > 1e-9}

    def fn_ngram(t, p):
        return ngram_prob(counts, ctxs, args.order, p[1], specials, lam)

    def make_fused(alpha, masked=False):
        def fn(t, p):
            lg, ctx = p
            if masked:
                lg = lg.clone()
                lg[torch.tensor(sorted(mask_ids))] = -float('inf')
            nd = neural_prob(lg)
            gd = ngram_prob(counts, ctxs, args.order, ctx, specials, lam)
            if alpha >= 1.0:
                return nd
            if alpha <= 0.0:
                return gd
            keys = set(nd) | set(gd)
            d = {}
            for r in keys:
                v = (nd.get(r, 1e-12) ** alpha) * (gd.get(r, 1e-12) ** (1 - alpha))
                if v > 1e-12:
                    d[r] = v
            s = sum(d.values())
            return {r: v / s for r, v in d.items()} if s > 0 else {}
        return fn

    mask_ids = set()
    try:
        from classical_governance_v2 import AlKhalilV2
        from collections import Counter as _C
        ak = AlKhalilV2(vocab)
        mask_ids = ak.mask()
        print(f'[*] Al-Khalil corrected mask: {len(mask_ids)} roots', flush=True)
    except Exception as e:
        print(f'[warn] masks unavailable: {e}')

    # tune alpha on a dev slice of the ALL set
    dev2 = recs_all[len(recs_all) // 4: len(recs_all) // 2]
    best_a = (0.5, -1.0)
    for a in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        m = score(dev2, make_fused(a))
        if m and m['acc@1'] > best_a[1]:
            best_a = (a, m['acc@1'])
    alpha = best_a[0]
    print(f'[*] tuned fusion alpha = {alpha:.2f} (dev acc@1 {100*best_a[1]:.2f}%)', flush=True)
    # and for the masked variant
    best_am = (0.5, -1.0)
    if mask_ids:
        for a in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            m = score(dev2, make_fused(a, masked=True))
            if m and m['acc@1'] > best_am[1]:
                best_am = (a, m['acc@1'])
        print(f'[*] tuned masked alpha = {best_am[0]:.2f}', flush=True)

    systems = [('ngram', fn_ngram),
               ('neural', make_fused(1.0)),
               ('fused', make_fused(alpha))]
    if mask_ids:
        systems.append(('fused+masks', make_fused(best_am[0], masked=True)))

    out = {'alpha': alpha, 'alpha_masked': best_am[0], 'lambdas': list(lam),
           'positions_all': len(recs_all), 'positions_novel': len(recs_novel), 'tables': {}}
    for tag, recs in (('ALL_val', recs_all), ('NOVEL_only', recs_novel)):
        print(f'\n===== {tag} ({len(recs)} positions) =====')
        print(f'  {"system":<14}{"acc@1":>9}{"acc@5":>9}{"ppl":>9}')
        out['tables'][tag] = {}
        for name, fn in systems:
            m = score(recs, fn)
            if m:
                out['tables'][tag][name] = m
                print(f'  {name:<14}{100*m["acc@1"]:>9.2f}{100*m["acc@5"]:>9.2f}{m["ppl"]:>9.1f}')

    json.dump(out, open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')
    print('\nNOVEL_only restricts to positions whose 12-root context never occurs in train,')
    print('so no lookup table can score by memorisation.')


if __name__ == '__main__':
    main()
