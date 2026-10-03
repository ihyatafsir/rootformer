#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""echo_probe_len1.py -- the readout-fair version of the user's length-1 echo test.

The trained root head (trained to predict the NEXT root) sits at the marginal when asked for a
word's OWN root.  That is uninformative about the representation, because the head's readout
direction is fixed.  This script fits a free closed-form linear readout (ridge) on the SAME
frozen length-1 hidden states and asks the identical question:

    h(prefix, root, wazn, suffix)  ->  that word's root?

train = unique train words, eval = unique held-out val words, plus a train/unseen split and the
<=9014 / >9014 clamp split.  Inference only; one GPU process; writes under /workspace/echo_test/.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

RELEASE = Path('/workspace/hf_v19_2_release')
OUTDIR = Path('/workspace/echo_test')
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'models'))
sys.path.insert(0, '/workspace/echo_test')

T0 = time.time()
RES = {'meta': {}, 'arms': {}}


def log(*a):
    print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)


def packed_words(streams, num_roots, num_awzan, num_suffixes, cap=None, seed=0):
    """Unique (p, r, w, s) tuples over the common index range, plus the packed keys."""
    L = min(int(t.numel()) for t in streams)
    P, R, W, S = [t.long()[:L] for t in streams]
    key = (((P * num_roots + R) * num_awzan + W) * num_suffixes + S)
    u = torch.unique(key)
    if cap is not None and u.numel() > cap:
        u = u[torch.randperm(u.numel(), generator=torch.Generator().manual_seed(seed))[:cap]]
        u = torch.sort(u).values
    s = u % num_suffixes
    q = u // num_suffixes
    w = q % num_awzan
    q = q // num_awzan
    r = q % num_roots
    p = q // num_roots
    return p.contiguous(), r.contiguous(), w.contiguous(), s.contiguous(), u


def main():
    import echo_identity as E
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from safetensors.torch import load_file
    from nrmt_arch import RootformerNRMT
    import nrmp_vocab as nv

    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json')
    vocab = V(bp)
    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16, hist=3, dropout=0.1,
                           use_features=True, feat_gate=False).to(device)
    sd = load_file(str(E.CKPT))
    model.load_state_dict(sd, strict=False)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    log('model ready')

    tr4 = [t.long() for t in torch.load(E.CACHE / 'train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(E.CACHE / 'val.pt', map_location='cpu')]
    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec_t = torch.tensor(specials)
    R = vocab.num_roots
    A, SF = vocab.num_awzan, vocab.num_suffixes

    tp, tr_, tw, ts, tk = packed_words(tr4, R, A, SF, cap=400000)
    vp, vr, vw, vs, vk = packed_words(va4, R, A, SF)
    mt = ~torch.isin(tr_, spec_t)
    mv = ~torch.isin(vr, spec_t)
    tp, tr_, tw, ts, tk = tp[mt], tr_[mt], tw[mt], ts[mt], tk[mt]
    vp, vr, vw, vs, vk = vp[mv], vr[mv], vw[mv], vs[mv], vk[mv]
    seen = torch.isin(vk, tk)
    log(f'unique train words {tp.numel()} | unique val words {vp.numel()} '
        f'(seen-in-train tuples {int(seen.sum())}, unseen {int((~seen).sum())})')

    @torch.no_grad()
    def extract_words(P, Rr, W, S, tag, bs=1024):
        out = []
        for i in range(0, P.numel(), bs):
            sl = slice(i, min(i + bs, P.numel()))
            p = P[sl].view(-1, 1).to(device)
            r = Rr[sl].view(-1, 1).to(device)
            w = W[sl].view(-1, 1).to(device)
            s = S[sl].view(-1, 1).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                emb = model.morphemic_embed(p, r, w, s)
                model._set_flash(r, w)
                h = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state)
            out.append(h[:, 0, :].float().cpu())
        H = torch.cat(out)
        log(f'    h[{tag}] {tuple(H.shape)}')
        return H

    Htr = extract_words(tp, tr_, tw, ts, 'train-words')
    Hva = extract_words(vp, vr, vw, vs, 'val-words')

    t0 = time.time()
    W, lam = E.fit_ridge(Htr, tr_, R, device)
    log(f'ridge fit in {time.time()-t0:.1f}s lam={lam:.4g}')

    C = R
    ones_v = torch.ones(vp.numel(), dtype=torch.bool)
    probe = {
        'val_all': E.eval_ridge(W, Hva, vr, ones_v, R, device, C),
        'val_unseen_words': E.eval_ridge(W, Hva, vr, ~seen, R, device, C),
        'val_seen_words': E.eval_ridge(W, Hva, vr, seen, R, device, C),
        'train_all': E.eval_ridge(W, Htr, tr_, torch.ones(tp.numel(), dtype=torch.bool),
                                  R, device, C),
    }
    # clamp split inside the val rows
    RES['arms']['ridge_probe_len1'] = probe
    # explicit gt/le split for the headline row
    m_gt = vr >= E.CUT
    RES['arms']['ridge_probe_len1_split_val'] = {
        'le_9014': E.eval_ridge(W, Hva[~m_gt], vr[~m_gt], torch.ones(int((~m_gt).sum()),
                                 dtype=torch.bool), R, device, C)['all'],
        'gt_9014': E.eval_ridge(W, Hva[m_gt], vr[m_gt], torch.ones(int(m_gt.sum()),
                                 dtype=torch.bool), R, device, C)['all'],
    }
    for k, v in probe.items():
        log(f'  probe len-1 {k:18s} {v["all"]}')
    log(f'  split: <=9014 {RES["arms"]["ridge_probe_len1_split_val"]["le_9014"]}')
    log(f'         >9014  {RES["arms"]["ridge_probe_len1_split_val"]["gt_9014"]}')

    # marginal baselines on the same val word sample
    RES['arms']['marginals_len1'] = {
        'val_all': E.majority_rate(vr, R),
        'val_unseen_words': E.majority_rate(vr[~seen], R),
        'val_seen_words': E.majority_rate(vr[seen], R),
        'train_all': E.majority_rate(tr_, R),
    }
    RES['arms']['note'] = ('the TRAINED head (head_CONTROL.pt) on this same length-1 task scored '
                           '3.39% acc@1 / 11.22% acc@5 on the val words; this probe is the '
                           'readout-fair ceiling for a linear map of the same frozen h.')
    RES['meta'].update({'peak_vram_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
                        'peak_vram_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1),
                        'wall_seconds': round(time.time() - T0, 1),
                        'n_train_words': int(tp.numel()), 'n_val_words': int(vp.numel())})
    json.dump(RES, open(OUTDIR / 'echo_probe_len1.json', 'w'), indent=2)
    log(f'wrote {OUTDIR/"echo_probe_len1.json"} peak_vram={RES["meta"]["peak_vram_allocated_mib"]}MiB')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        json.dump(RES, open(OUTDIR / 'echo_probe_len1.json', 'w'), indent=2)
        sys.exit(1)
