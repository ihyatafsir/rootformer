#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmt_governed_eval.py -- Al-Khalil / Sibawayh / Al-Musta'mal governed next-root prediction,
evaluated with the NEW NRMT ARCHITECTURE (v20.1) instead of the shipped v20 head.

Same tiers, same held-out sentences, same metric definitions as nrmp_governed_eval.py, so the
two tables are directly comparable:

  T0 raw            : all 9,114 roots (only control tokens masked)
  T1 + Al-Khalil    : corrected phonotactics
  T2 + Sibawayh     : operator governance (Harf Jarr / Jazm / Nasb / Inna / Kana / Future)
  T3 + Al-Musta'mal : attested-register prior, K swept over {1000, 2500, 5000}

Differences from the v20 evaluation: the head reads root history E(r-1,r-2,r-3), Sibawayh's
operator state, and the previous morph tuple, and it is applied to the SAME backbone the v20.1
head was trained on (rootformer_v19_2_synthesis_ar_backbone.safetensors) in float32, because that
head was trained in float32 over cached float32 hidden states.

Reports coverage / acc@k_reachable / acc@1_overall, because narrowing the candidate set raises
accuracy mechanically.

Usage:
  python nrmt_governed_eval.py --n 150
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

# reuse the v20 constraint machinery verbatim so the tiers are identical
import nrmp_governed_eval as G


@torch.no_grad()
def evaluate(nrmt, vocab, cs, sentences, device, op_table, forbidden, use_ops, gov,
             topk=(1, 5, 10)):
    forb_t = torch.tensor(sorted(forbidden), device=device) if forbidden else None


    tot = reach = 0
    hit_r = {k: 0 for k in topk}
    hit_a = {k: 0 for k in topk}
    ce_sum, ce_n = 0.0, 0
    cand_sum = 0

    for sent in sentences:
        enc = vocab.encode_sentence(sent)
        if len(enc) < 2:
            continue
        p = torch.tensor([[t[0] for t in enc]], dtype=torch.long, device=device)
        r = torch.tensor([[t[1] for t in enc]], dtype=torch.long, device=device)
        w = torch.tensor([[t[2] for t in enc]], dtype=torch.long, device=device)
        s = torch.tensor([[t[3] for t in enc]], dtype=torch.long, device=device)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb = nrmt.morphemic_embed(p, r, w, s)
            nrmt._set_flash(r, w)
            out = nrmt.backbone(inputs_embeds=emb)
            h = nrmt.final_norm(out.last_hidden_state)
        # the v20.1 head was trained in float32 on float32 hidden states
        opf = op_table.to(device)[r.clamp(min=0)]
        logits = nrmt.nrmt_head(h.float(), r, opf, w, p, s)['root_logits'][0].float()

        for t in range(logits.shape[0] - 1):
            tgt = int(r[0, t + 1].item())
            if tgt in cs.specials:
                continue
            lg = logits[t].clone()
            if forb_t is not None and forb_t.numel():
                lg[forb_t] = -float('inf')
            if use_ops and gov is not None:
                prev = (int(p[0, t]), int(r[0, t]), int(w[0, t]), int(s[0, t]))
                lg = gov.apply_root_exclusion_mask(lg, gov.get_operator_state(prev),
                                                   r[0, :t + 1].tolist(), t, 0)
            tot += 1
            if not torch.isfinite(lg[tgt]):
                continue
            reach += 1
            nf = int(torch.isfinite(lg).sum().item())
            cand_sum += nf
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
           'candidates_mean': cand_sum / max(reach, 1),
           'ppl_restricted': math.exp(min(ce_sum / max(ce_n, 1), 25.0)) if ce_n else None}
    for k in topk:
        res[f'acc@{k}_reachable'] = hit_r[k] / max(reach, 1)
        res[f'acc@{k}_overall'] = hit_a[k] / max(tot, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',
                    default='/workspace/hf_v19_2_release/checkpoints/'
                            'rootformer_v19_2_synthesis_ar_backbone.safetensors')
    ap.add_argument('--head', default='/workspace/v20_analysis/nrmt_head_full.pt')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--train-stream', default='/workspace/nrmp_cache/train.pt')
    ap.add_argument('--heritage-ks', type=int, nargs='*', default=[1000, 2500, 5000])
    ap.add_argument('--out', default='/workspace/nrmt_governed_eval.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from safetensors.torch import load_file
    from nrmt_arch import RootformerNRMT, build_operator_table
    from sibawayh_governance_engine import SibawayhNRMPGovernance

    # constructed exactly as nrmt_train.py built it, so the head's inputs match its training
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
    vocab = V(bp)
    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    nrmt = RootformerNRMT(flash, vocab, device, torch.bfloat16, hist=3,
                          dropout=0.0, use_features=True).to(device)
    missing, unexpected = nrmt.load_state_dict(load_file(args.checkpoint), strict=False)
    newk = [k for k in missing if k.startswith('nrmt_head.')]
    print(f'[*] backbone {Path(args.checkpoint).name}: missing={len(missing)} '
          f'(nrmt_head new={len(newk)}) unexpected={len(unexpected)}')
    print(f'[*] extra-conditioning norm at init: {nrmt.nrmt_head.extra_norm():.6f}')

    sd = torch.load(args.head, map_location='cpu')
    miss2, unexp2 = nrmt.nrmt_head.load_state_dict(sd, strict=False)
    print(f'[*] NRMT head {Path(args.head).name}: missing={len(miss2)} unexpected={len(unexp2)}')
    nrmt.nrmt_head.to(torch.float32)
    nrmt.eval()

    gov = SibawayhNRMPGovernance(vocab)
    op_table = build_operator_table(vocab).to(device)
    print(f'[*] non-zero operator roots: {int((op_table != 0).sum())}')

    freq = Counter()
    if Path(args.train_stream).exists():
        freq = G.load_train_root_freq(args.train_stream, vocab)
    cs = G.ConstraintSet(vocab, freq)
    sents = G.load_eval_sentences(args.n)
    print(f'[*] held-out sentences: {len(sents)}  (identical set to the v20 evaluation)\n')

    tiers = [('T0_raw', cs.base, False),
             ('T1_khalil', cs.base | cs.khalil_bad, False),
             ('T2_sibawayh', cs.base | cs.khalil_bad, True)]
    for k in args.heritage_ks:
        tiers.append((f'T3_heritage_top{k}', cs.base | cs.khalil_bad | cs.heritage_bad(k), True))

    results = {}
    hdr = (f'{"tier":<22}{"cover":>8}{"acc@1":>8}{"acc@5":>8}{"acc@10":>9}'
           f'{"overall@1":>11}{"cand":>8}{"ppl":>9}')
    print(hdr); print('-' * len(hdr))
    for name, forb, ops in tiers:
        res = evaluate(nrmt, vocab, cs, sents, device, op_table, forb, ops, gov)
        results[name] = res
        print(f'{name:<22}{100*res["coverage"]:>7.1f}%{100*res["acc@1_reachable"]:>8.2f}'
              f'{100*res["acc@5_reachable"]:>8.2f}{100*res["acc@10_reachable"]:>9.2f}'
              f'{100*res["acc@1_overall"]:>11.2f}{res["candidates_mean"]:>8.0f}'
              f'{res["ppl_restricted"]:>9.1f}', flush=True)

    json.dump(results, open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
