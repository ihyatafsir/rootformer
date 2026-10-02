#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmp_fuse.py -- is the neural NRMP head complementary to count-based root context models?

Establishes, on a file-level held-out split with everything fit on TRAIN only:
  * neural  : our 0.5B Rootformer NRMP head logits (optionally Al-Khalil/Sibawayh masked)
  * ngram   : trigram / 4-gram over the root stream fit on TRAIN
  * fused   : log-linear interpolation  P ∝ P_neural^alpha * P_ngram^beta

alpha/beta are tuned on a VAL half of the held-out sentences and reported on the TEST half,
so the fusion weight is never fitted on the numbers we report.

If fused > ngram, the head carries information the counts do not. If fused ~= ngram, the head
is redundant and should be replaced by explicit root-history conditioning.

Usage:
  python nrmp_fuse.py --checkpoint /workspace/nrmp_trained_final.safetensors --n 300
"""
import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640]')


def load_sentences(n, seed=0, min_words=5, max_words=60):
    import glob
    sents = []
    for f in (sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))[:4]
              + sorted(glob.glob('/workspace/andalusian_canon_sanitized/*.txt'))[:3]):
        txt = Path(f).read_text(encoding='utf-8', errors='ignore')
        for s in re.split(r'[\n.!?؟]+', txt):
            s = s.strip()
            if min_words <= len(s.split()) <= max_words and re.search(r'[\u0600-\u06FF]', s):
                sents.append(s)
    random.Random(seed).shuffle(sents)
    return sents[:n]


def build_ngram(stream, order, vocab, specials):
    counts = [Counter() for _ in range(order + 1)]
    ctx = [defaultdict(Counter) for _ in range(order + 1)]
    for i, r in enumerate(stream):
        counts[1][r] += 1
        for o in range(2, order + 1):
            if i >= o - 1:
                ctx[o][tuple(stream[i - o + 1:i])][r] += 1
    return counts, ctx


def ngram_dist(counts, ctx, order, history, vocab, specials, alpha_uni=0.4):
    """Stupid-backoff-style mixture over orders, returning a dict root->prob."""
    dist = Counter()
    # unigram
    tot1 = sum(counts[1].values())
    for r, c in counts[1].items():
        if r not in specials:
            dist[r] += alpha_uni * (c / tot1)
    weight = 1.0 - alpha_uni
    for o in range(2, order + 1):
        key = tuple(history[-(o - 1):]) if len(history) >= o - 1 else None
        c = ctx[o].get(key) if key else None
        if c:
            tot = sum(c.values())
            for r, k in c.items():
                if r not in specials:
                    dist[r] += weight * (k / tot)
            weight = 0.0
            break
    s = sum(dist.values())
    if s <= 0:
        return {}
    return {r: v / s for r, v in dist.items()}


@torch.no_grad()
def neural_logits(model, vocab, sentences, device, batches=8):
    """Return list of (target_root, [logits tensor], history_root_ids) per scored position."""
    out = []
    for sent in sentences:
        enc = vocab.encode_sentence(sent)
        if len(enc) < 3:
            continue
        p = torch.tensor([[t[0] for t in enc]], dtype=torch.long, device=device)
        r = torch.tensor([[t[1] for t in enc]], dtype=torch.long, device=device)
        w = torch.tensor([[t[2] for t in enc]], dtype=torch.long, device=device)
        s = torch.tensor([[t[3] for t in enc]], dtype=torch.long, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb = model.morphemic_embed(p, r, w, s)
            o = model.backbone(inputs_embeds=emb)
            h = model.final_norm(o.last_hidden_state)
            lg = model.nrmp_head.root_head(h)[0].float()
        rs = r[0].tolist()
        for t in range(1, lg.shape[0]):
            out.append((rs[t], lg[t].cpu(), rs[:t]))
    return out


def metrics(records, prob_fn, topk=(1, 5)):
    n = 0
    hit = {k: 0 for k in topk}
    ce = 0.0
    for tgt, hist in records:
        d = prob_fn(tgt, hist)
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
    return {'n': n, **{f'acc@{k}': hit[k] / n for k in topk},
            'ppl': math.exp(min(ce / n, 25.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--order', type=int, default=4)
    ap.add_argument('--train-stream', default='/workspace/nrmp_cache/train.pt')
    ap.add_argument('--apply-masks', action='store_true')
    ap.add_argument('--alphas', type=float, nargs='*',
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument('--out', default='/workspace/nrmp_fuse.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from nrmp_run import load_engine
    model, vocab, device, _ = load_engine(ckpt=args.checkpoint)

    specials = {vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                vocab.root2id['<PARTICLE>']}
    specials |= {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')}

    stream = torch.load(args.train_stream, map_location='cpu')[1].tolist()
    print(f'[*] train stream {len(stream)} tokens; building {args.order}-gram...', flush=True)
    counts, ctx = build_ngram(stream, args.order, vocab, specials)

    sents = load_sentences(args.n)
    half = len(sents) // 2
    val_s, test_s = sents[:half], sents[half:]
    print(f'[*] held-out: val={len(val_s)} test={len(test_s)} sentences', flush=True)

    masks = None
    if args.apply_masks:
        from nrmp_governed_eval import ConstraintSet
        cs = ConstraintSet(vocab, Counter())
        masks = cs.base | cs.khalil_bad
        print(f'[*] applying Al-Khalil masks ({len(masks)} roots)', flush=True)

    def prep(sentences):
        recs = []
        for tgt, lg, hist in neural_logits(model, vocab, sentences, device):
            if tgt in specials:
                continue
            if masks:
                lg[[i for i in masks if i < lg.numel()]] = -float('inf')
            recs.append((tgt, (lg, hist)))
        return recs

    val_recs = prep(val_s)
    test_recs = prep(test_s)
    print(f'[*] scored positions: val={len(val_recs)} test={len(test_recs)}\n', flush=True)

    def neural_prob_of(lg):
        finite = torch.isfinite(lg)
        d = {}
        if not finite.any():
            return d
        lg2 = lg.clone()
        lg2[~finite] = -1e4
        probs = F.softmax(lg2, dim=-1)
        idx = torch.nonzero(finite).flatten().tolist()
        for r in idx:
            pr = float(probs[r].item())
            if pr > 1e-9:
                d[r] = pr
        return d

    def make_fn(alpha):
        def fn(tgt, hist):
            lg, h = hist
            nd = neural_prob_of(lg)
            gd = ngram_dist(counts, ctx, args.order, h, vocab, specials)
            if alpha >= 1.0 or not gd:
                return nd
            if alpha <= 0.0:
                return gd
            keys = set(nd) | set(gd)
            d = {}
            for r in keys:
                p = (nd.get(r, 1e-12) ** alpha) * (gd.get(r, 1e-12) ** (1 - alpha))
                if p > 1e-12:
                    d[r] = p
            s = sum(d.values())
            return {r: v / s for r, v in d.items()} if s > 0 else {}
        return fn

    best = None
    print(f'{"alpha(neural)":<14}{"val acc@1":>12}{"val acc@5":>12}{"val ppl":>10}')
    print('-' * 48)
    for a in args.alphas:
        m = metrics(val_recs, make_fn(a))
        if m is None:
            continue
        print(f'{a:<14.2f}{100*m["acc@1"]:>12.2f}{100*m["acc@5"]:>12.2f}{m["ppl"]:>10.1f}', flush=True)
        if best is None or m['acc@1'] > best[0]:
            best = (m['acc@1'], a)

    alpha_star = best[1]
    print(f'\n[*] alpha* = {alpha_star:.2f} (chosen on val acc@1)\n')
    print(f'{"model (TEST)":<26}{"acc@1":>9}{"acc@5":>9}{"ppl":>9}')
    print('-' * 53)
    res = {}
    for name, a in (('neural only', 1.0), ('ngram only', 0.0),
                    (f'fused alpha={alpha_star:.2f}', alpha_star)):
        m = metrics(test_recs, make_fn(a))
        res[name] = m
        if m:
            print(f'{name:<26}{100*m["acc@1"]:>9.2f}{100*m["acc@5"]:>9.2f}{m["ppl"]:>9.1f}')

    res['alpha_star'] = alpha_star
    res['order'] = args.order
    res['masks'] = bool(args.apply_masks)
    json.dump(res, open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')
    print('If fused > ngram only, the head adds complementary signal; if not, it is redundant.')


if __name__ == '__main__':
    main()
