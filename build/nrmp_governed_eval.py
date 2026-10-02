#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmp_governed_eval.py -- Al-Khalil / Sibawayh / Andalusian governed next-root-morph prediction.

Evaluates the project's OWN 0.5B Rootformer NRMP head (no external LLM) on held-out classical
text under progressively stricter classical constraints:

  T0 raw            : unconstrained over all 9,114 roots (only control tokens masked)
  T1 + Al-Khalil    : corrected phonotactics -- C1==C2 impossible; same-makhraj guttural clash
                      (the previous implementation also forbade every bare-alif-initial root,
                       wrongly killing 249 real roots such as اصل/ارض/اخذ/احد/اسس -- removed)
  T2 + Sibawayh     : operator governance (Harf Jarr / Jazm / Nasb / Inna / Kana / Future)
  T3 + Al-Musta'mal : heritage register prior -- restrict to the top-K most frequent roots of
                      classical scholastic usage, K swept over {1000, 2500, 5000, all}

Every tier reports THREE distinct things, because conflating them is how inflated numbers happen:
  coverage  -- share of gold roots still permitted after masking (over-masking shows up here)
  acc@k     -- accuracy over positions where the gold root is reachable
  overall@1 -- accuracy over ALL positions. This is the honest headline.
Narrowing the candidate set mechanically raises acc; `coverage` and `overall@1` keep it honest.

Usage:
  python nrmp_governed_eval.py --checkpoint /workspace/nrmp_trained_final.safetensors --n 150
"""
import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640]')

# Al-Khalil (I'tilaf wa Tanafur al-Huruf): consonants sharing a makhraj (point of articulation)
# cannot be adjacent radicals. Same-makhraj guttural pairs, both orders.
# NB: only SAME-makhraj pairs -- forbidding arbitrary guttural+guttural would wrongly kill
# real roots such as عهد (ع pharyngeal + ه glottal).
GUTTURAL_CLASH = {('ء', 'ه'), ('ه', 'ء'), ('ع', 'ح'), ('ح', 'ع'), ('غ', 'خ'), ('خ', 'غ')}


def strip_diac(s):
    return DIAC.sub('', s)


class ConstraintSet:
    def __init__(self, vocab, root_freq):
        self.vocab = vocab
        self.roots_list = vocab.roots_list
        n = len(self.roots_list)

        self.specials = {vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT}
        for i, r in enumerate(self.roots_list):
            if r == '<PARTICLE>':
                self.specials.add(i)
        self.specials = {i for i in self.specials if i >= 0}
        self.leaked = {i for i, r in enumerate(self.roots_list) if r in ('end', 'start')}
        self.particles = {i for i, r in enumerate(self.roots_list) if r.startswith('<P:')}

        # ---- Al-Khalil phonotactics (corrected) ----
        self.khalil_bad = set(self.leaked)
        for i, r in enumerate(self.roots_list):
            s = strip_diac(r)
            if r.startswith('<') or len(s) != 3:
                continue
            if s[0] == s[1]:
                self.khalil_bad.add(i)
            if (s[0], s[1]) in GUTTURAL_CLASH:
                self.khalil_bad.add(i)

        self.root_freq = root_freq or Counter()
        ranked = [r for r, _ in self.root_freq.most_common()]
        self.ranked = ranked

        self.base = self.specials | self.leaked
        print(f'[*] inventory={n} specials={len(self.specials)} leaked={len(self.leaked)} '
              f'particles={len(self.particles)}')
        print(f'[*] Al-Khalil impossible roots (corrected): {len(self.khalil_bad)}')
        print(f'[*] distinct roots attested in TRAIN: {len(ranked)}')

    def heritage_bad(self, top_k):
        if top_k is None:
            return set()
        keep = set(self.ranked[:top_k])
        return {i for i, r in enumerate(self.roots_list)
                if not r.startswith('<') and r not in keep}


def load_train_root_freq(path, vocab):
    t = torch.load(path, map_location='cpu')
    roots = t[1].tolist()
    c = Counter(vocab.id2root[r] for r in roots if r in vocab.id2root)
    return Counter({r: n for r, n in c.items() if not r.startswith('<')})


def load_eval_sentences(n, seed=0, min_words=5, max_words=60):
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


@torch.no_grad()
def evaluate(model, vocab, cs, sentences, device, forbidden, use_operator_masks, topk=(1, 5, 10)):
    gov = getattr(model, 'sibawayh_gov', None)
    forb_t = torch.tensor(sorted(forbidden), device=device) if forbidden else None

    tot = reach = 0
    hit_r = {k: 0 for k in topk}
    hit_a = {k: 0 for k in topk}
    ce_sum, ce_n = 0.0, 0

    for sent in sentences:
        enc = vocab.encode_sentence(sent)
        if len(enc) < 2:
            continue
        p = torch.tensor([[t[0] for t in enc]], dtype=torch.long, device=device)
        r = torch.tensor([[t[1] for t in enc]], dtype=torch.long, device=device)
        w = torch.tensor([[t[2] for t in enc]], dtype=torch.long, device=device)
        s = torch.tensor([[t[3] for t in enc]], dtype=torch.long, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb = model.morphemic_embed(p, r, w, s)
            out = model.backbone(inputs_embeds=emb)
            h = model.final_norm(out.last_hidden_state)
            logits = model.nrmp_head.root_head(h)[0].float()

        for t in range(logits.shape[0] - 1):
            tgt = int(r[0, t + 1].item())
            if tgt in cs.specials:
                continue
            lg = logits[t].clone()
            if forb_t is not None and forb_t.numel():
                lg[forb_t] = -float('inf')
            if use_operator_masks and gov is not None:
                prev = (int(p[0, t]), int(r[0, t]), int(w[0, t]), int(s[0, t]))
                lg = gov.apply_root_exclusion_mask(lg, gov.get_operator_state(prev),
                                                   r[0, :t + 1].tolist(), t, 0)

            tot += 1
            if not torch.isfinite(lg[tgt]):
                continue
            reach += 1
            nf = int(torch.isfinite(lg).sum().item())
            idx = torch.topk(lg, min(max(topk), nf)).indices.tolist()
            for k in topk:
                if tgt in idx[:k]:
                    hit_r[k] += 1
                    hit_a[k] += 1
            lg2 = lg.clone()
            lg2[~torch.isfinite(lg2)] = -1e4
            ce_sum += float(-F.log_softmax(lg2, dim=-1)[tgt].item())
            ce_n += 1

    res = {'positions': tot, 'reachable': reach, 'coverage': reach / max(tot, 1),
           'candidates_mean': None,
           'ppl_restricted': math.exp(min(ce_sum / max(ce_n, 1), 25.0)) if ce_n else None}
    for k in topk:
        res[f'acc@{k}_reachable'] = hit_r[k] / max(reach, 1)
        res[f'acc@{k}_overall'] = hit_a[k] / max(tot, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--train-stream', default='/workspace/nrmp_cache/train.pt')
    ap.add_argument('--heritage-ks', type=int, nargs='*', default=[1000, 2500, 5000])
    ap.add_argument('--out', default='/workspace/nrmp_governed_eval.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from nrmp_run import load_engine
    model, vocab, device, _ = load_engine(ckpt=args.checkpoint)

    freq = Counter()
    if Path(args.train_stream).exists():
        freq = load_train_root_freq(args.train_stream, vocab)
    else:
        print(f'[warn] no train stream at {args.train_stream}')

    cs = ConstraintSet(vocab, freq)
    sents = load_eval_sentences(args.n)
    print(f'[*] held-out sentences: {len(sents)}\n')

    tiers = [
        ('T0_raw', cs.base, False),
        ('T1_khalil', cs.base | cs.khalil_bad, False),
        ('T2_sibawayh', cs.base | cs.khalil_bad, True),
    ]
    for k in args.heritage_ks:
        tiers.append((f'T3_heritage_top{k}', cs.base | cs.khalil_bad | cs.heritage_bad(k), True))

    results = {}
    hdr = f'{"tier":<22}{"cover":>8}{"acc@1":>8}{"acc@5":>8}{"acc@10":>9}{"overall@1":>11}{"ppl":>9}'
    print(hdr)
    print('-' * len(hdr))
    for name, forb, ops in tiers:
        res = evaluate(model, vocab, cs, sents, device, forb, ops)
        results[name] = res
        print(f'{name:<22}{100*res["coverage"]:>7.1f}%{100*res["acc@1_reachable"]:>8.2f}'
              f'{100*res["acc@5_reachable"]:>8.2f}{100*res["acc@10_reachable"]:>9.2f}'
              f'{100*res["acc@1_overall"]:>11.2f}{res["ppl_restricted"]:>9.1f}', flush=True)

    json.dump(results, open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')
    print('\ncoverage   = share of gold roots still permitted (over-masking shows up here)')
    print('acc@k      = accuracy over positions where the gold root is reachable')
    print('overall@1  = accuracy over ALL positions (the honest headline)')


if __name__ == '__main__':
    main()
