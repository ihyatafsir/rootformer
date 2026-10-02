#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repro_eval.py -- INDEPENDENT fresh-process reproduction of an arm's final held-out number.

Loads the arm's saved head (`head_<TAG>.pt`, head-local keys) and trunk
(`head_<TAG>.pt.trunk.pt`, the unfrozen backbone layers + the cross-attention state) on top of
the shipped checkpoint, rebuilds the metric from the stream cache with the SAME
val starts / novelty mask / ʿāmil stream / specials mask as the trainer, and prints
ALL_val and NOVEL_only acc@1 / acc@5 / CE_z.

Nothing here calls the trainer: it is a second implementation of the evaluation, which is what
makes it a check rather than a re-run.

Usage:
  python repro_eval.py --checkpoint <ckpt> --cache <stream cache> \
      --head /workspace/root_attn/head_RCA_UNFREEZE_A2.pt \
      --trunk /workspace/root_attn/head_RCA_UNFREEZE_A2.pt.trunk.pt
"""
import argparse
import json
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/root_attn')

WIN, STRIDE, CTX = 128, 64, 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--head', required=True)
    ap.add_argument('--trunk', default=None)
    ap.add_argument('--train-windows', type=int, default=3000)
    ap.add_argument('--val-windows', type=int, default=300)
    ap.add_argument('--tag', default='REPRO')
    ap.add_argument('--no-rca', action='store_true', help='disable the cross-attention hook')
    ap.add_argument('--rca-train', action='store_true',
                    help='leave the RCA in TRAIN mode (dropout active), as the trainer accidentally does')
    ap.add_argument('--gate-off', action='store_true',
                    help='force the cross-attention gates to 0 (reproduce the RCA_OFF ablation)')
    args = ap.parse_args()

    import nrmp_vocab as nv
    import nrmt_train as NT
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT, build_operator_table
    from safetensors.torch import load_file
    from root_cross_attn import RootCrossAttentionStack

    device = torch.device('cuda')
    bp = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V(bp)

    tr4 = [t.long() for t in torch.load(f'{args.cache}/train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(f'{args.cache}/val.pt', map_location='cpu')]

    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16).to(device)
    model.load_state_dict(load_file(args.checkpoint), strict=False)
    for p in model.parameters():
        p.requires_grad = False
    model.nrmt_head.to(torch.float32)

    # ---- the arm's own saved trunk + cross-attention --------------------------------------
    stack = None
    exclude_current = False
    if args.trunk:
        d = torch.load(args.trunk, map_location='cpu')
        print('[*] trunk checkpoint:', {k: v for k, v in d.items() if k != 'state'})
        layers = d.get('rca_layers') or []
        stack = RootCrossAttentionStack(
            model.d_model, model.morphemic_embed.root_embed, layers,
            num_heads=int(d.get('rca_heads', 8)), dropout=float(d.get('rca_dropout', 0.0)),
            dtype=torch.float32, out_norm=bool(d.get('rca_out_norm', False))).to(device)
        exclude_current = bool(d.get('rca_exclude_current', False))
        stack.exclude_current = exclude_current
        model.root_cross = stack
        res = model.load_state_dict({k: v.to(model.dtype) if v.is_floating_point() and
                                     'root_cross' not in k else v
                                     for k, v in d['state'].items()}, strict=False)
        print(f'[*] trunk state loaded: missing={len(res.missing_keys)} '
              f'unexpected={len(res.unexpected_keys)}')
        stack.attach(model.backbone.layers)
    model.eval()
    model.backbone.eval()
    model.nrmt_head.eval()
    if args.rca_train and stack is not None:
        stack.train()
        print('[*] RCA left in TRAIN mode (dropout active during eval)')

    hsd = torch.load(args.head, map_location='cpu')
    res = model.nrmt_head.load_state_dict(hsd, strict=False)
    print(f'[*] head loaded: {len(hsd)} tensors, missing={len(res.missing_keys)}')

    # ---- val windows + masks, exactly as the trainer ---------------------------------------
    def starts_for(t, n, seed):
        s = list(range(0, t.numel() - WIN - 1, STRIDE))
        np.random.default_rng(seed).shuffle(s)
        return sorted(s[:n])

    va_st = starts_for(va4[1], args.val_windows, 1)

    def win_tensor(t, starts):
        return torch.stack([t[j:j + WIN] for j in starts])

    Tv, Wv, Pv, Sv = (win_tensor(va4[i], va_st) for i in (1, 2, 0, 3))
    Rt_va = Tv[:, 1:].contiguous()

    nov = NT.Novelty(tr4[1].tolist())
    per = WIN - 1
    novel_mask = torch.zeros(Tv.shape[0] * per, dtype=torch.bool)
    for w in range(Tv.shape[0]):
        seg = Tv[w].tolist()
        for t in range(CTX, WIN):
            novel_mask[w * per + (t - 1)] = nov.novel(seg[t - CTX:t])
    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec_t = torch.tensor(specials)
    keep_all = ~torch.isin(Rt_va.reshape(-1), spec_t)
    keep_novel = keep_all & novel_mask

    op_table = build_operator_table(vocab)
    try:
        from sibawayh_governor import SibawayhGovernor
        gov = SibawayhGovernor(vocab)
        rows = [gov.op_ids_for_ids(rr, ww, pp)
                for pp, rr, ww in zip(Pv.tolist(), Tv.tolist(), Wv.tolist())]
        Ova_w = torch.tensor(rows, dtype=torch.long)[:, :-1]
        print('[*] ʿāmil stream: SibawayhGovernor (persistent)')
    except Exception as e:                                   # pragma: no cover
        print(f'[warn] governor unavailable ({e}); using the t-1 operator table')
        Ova_w = op_table[Tv.clamp(min=0)][:, :-1]

    # ---- fresh evaluation ------------------------------------------------------------------
    lg = []
    with torch.no_grad():
        for i in range(0, Tv.shape[0], 16):
            sl = slice(i, min(i + 16, Tv.shape[0]))
            p, r, w, s = (t[sl].to(device) for t in (Pv, Tv, Wv, Sv))
            if stack is not None and not args.no_rca:
                stack.set_root_ids(r)
                if args.gate_off:
                    for m in stack.mods:
                        m.gate.data.zero_()
            emb = model.morphemic_embed(p, r, w, s)
            h = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state)
            h = h.float()[:, :-1].contiguous()
            out = model.nrmt_head(h, r[:, :-1], Ova_w[sl].to(device), w[:, :-1], p[:, :-1],
                                  s[:, :-1])
            lg.append(out['root_logits'].float().cpu())
    LG = torch.cat(lg).reshape(-1, vocab.num_roots)
    tg = Rt_va.reshape(-1)
    assert LG.shape[0] == tg.shape[0] == Tv.shape[0] * per, (LG.shape, tg.shape)
    res = {}
    for tag, m in (('ALL_val', keep_all), ('NOVEL_only', keep_novel)):
        l, t = LG[m], tg[m]
        ce_z = float(torch.nn.functional.cross_entropy(
            (l - l.mean(-1, keepdim=True)) / l.std(-1, keepdim=True).clamp_min(1e-6), t))
        res[tag] = {'n': int(t.numel()),
                    'acc@1': float((l.argmax(-1) == t).float().mean()),
                    'acc@5': float((l.topk(5, -1).indices == t.unsqueeze(-1)).any(-1).float().mean()),
                    'ce_z': ce_z, 'ppl_z': math.exp(min(ce_z, 25.0))}
        print(f'  {tag}: acc@1 {100*res[tag]["acc@1"]:.3f}%  acc@5 {100*res[tag]["acc@5"]:.3f}%  '
              f'CE_z {ce_z:.4f}  (n={res[tag]["n"]})')
    if args.tag:
        json.dump(res, open(f'/workspace/root_attn/repro_{args.tag}.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
