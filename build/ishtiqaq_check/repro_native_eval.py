#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repro_native_eval.py -- the native root path's effect on HELD-OUT ACC@1, measured on a
CONVERGED head with a FROZEN trunk.  No training, CPU-only.

Method: a second implementation of the trainer's evaluation (val starts derived from the stream
cache, the trainer's novelty mask, the Siba wayh ʿāmil stream, the specials mask), with the
shipped `head_ALIGNED_FIX.pt` (the exact control for RCA_UNFREEZE_A2, published 6.619 %).
`model._set_flash(r, w)` IS called, because that is what the frozen-cache arms did
(`nrmt_train.py:100`) -- the live RCA arms do NOT call it (`nrmt_train.py:593-604`), which is a
separate pre-existing difference between the ladder's frozen and live rungs.  Fixing it here
means the ONLY thing that varies between variants is the native root path.

Variants measured in one process on the same windows:
  off                    the frozen baseline
  shipped_force_enable   the shipped module with `active_root_ids` set on layers 20-23 (the
                         historical "force-enable": root score bias + identical_root_bonus +
                         Pillar I coverage_weight 2.0 + Pillar II governance_strength 1.5)
  root_terms_shared      the FIXED path, shared 448-dim source, root_gate=1, Pillar gate=0
                         (= exactly the mechanism arm RCA_NATIVE_A2 switches on)
  bonus_only_shared      the same, with stream_mix[1]=0 so only identical_root_bonus acts
"""
import argparse
import json
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
# APPEND, not insert: `import nrmt_train as NT` below must resolve to the UNPATCHED release
# module, not to the experiment copy that shares this directory.
sys.path.append('/workspace/ishtiqaq_check')

WIN, STRIDE, CTX = 128, 64, 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--head', required=True)
    ap.add_argument('--layers', default='top4')
    ap.add_argument('--train-windows', type=int, default=3000)
    ap.add_argument('--val-windows', type=int, default=300)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--set-flash', dest='set_flash', action='store_true', default=True)
    ap.add_argument('--no-set-flash', dest='set_flash', action='store_false')
    ap.add_argument('--variants', default='off,root_terms_shared,bonus_only_shared,'
                                          'shipped_force_enable')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import nrmp_vocab as nv
    import nrmt_train as NT
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT, build_operator_table
    from safetensors.torch import load_file
    from ishtiqaq_root_bias import swap_in_root_bias, parse_layer_spec

    device = torch.device(args.device)
    bp = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V(bp)

    tr4 = [t.long() for t in torch.load(f'{args.cache}/train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(f'{args.cache}/val.pt', map_location='cpu')]

    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    # EXACTLY the flags the FIX / ALIGNED_FIX arms were trained under: `--feat-gate`
    # matters -- without it the root-history branch is applied UNGATED, and the trained
    # head collapses to 0.94 % instead of 6.62 %.
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16, hist=3, dropout=0.1,
                           use_features=True, feat_gate=True,
                           feat_gate_proj_std=1e-3, ngram_mode='none').to(device)
    model.load_state_dict(load_file(args.checkpoint), strict=False)
    for p in model.parameters():
        p.requires_grad = False
    model.nrmt_head.to(torch.float32)

    hsd = torch.load(args.head, map_location='cpu')
    res = model.nrmt_head.load_state_dict(hsd, strict=False)
    print(f'[*] head {args.head}: {len(hsd)} tensors, missing={len(res.missing_keys)}')
    model.eval()
    model.backbone.eval()
    model.nrmt_head.eval()

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
    print(f'[*] val windows {Tv.shape[0]}, positions {int(keep_all.sum())} all / '
          f'{int(keep_novel.sum())} novel')

    op_table = build_operator_table(vocab)
    try:
        from sibawayh_governor import SibawayhGovernor
        gov = SibawayhGovernor(vocab)
        rows = [gov.op_ids_for_ids(rr, ww, pp)
                for pp, rr, ww in zip(Pv.tolist(), Tv.tolist(), Wv.tolist())]
        Ova_w = torch.tensor(rows, dtype=torch.long)[:, :-1]
        print('[*] ʿāmil stream: SibawayhGovernor')
    except Exception as e:                                   # pragma: no cover
        print(f'[warn] governor unavailable ({e}); using the t-1 operator table')
        Ova_w = op_table[Tv.clamp(min=0)][:, :-1]

    layers_idx = parse_layer_spec(args.layers, len(model.backbone.layers))

    def evaluate(tag, native=None, active_ship=None):
        """native: list of fixed modules (or None); active_ship: shipped-module layer list."""
        lg = []
        with torch.no_grad():
            for i in range(0, Tv.shape[0], 16):
                sl = slice(i, min(i + 16, Tv.shape[0]))
                p, r, w, s = (t[sl].to(device) for t in (Pv, Tv, Wv, Sv))
                if native is not None:
                    for m in native:
                        m.active_root_ids = r
                if active_ship:
                    for li in active_ship:
                        model.backbone.layers[li].self_attn.active_root_ids = r
                if args.set_flash:
                    model._set_flash(r, w)
                emb = model.morphemic_embed(p, r, w, s)
                h = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state)
                h = h.float()[:, :-1].contiguous()
                out = model.nrmt_head(h, r[:, :-1], Ova_w[sl].to(device), w[:, :-1],
                                      p[:, :-1], s[:, :-1])
                lg.append(out['root_logits'].float().cpu())
        LG = torch.cat(lg).reshape(-1, vocab.num_roots)
        tg = Rt_va.reshape(-1)
        assert LG.shape[0] == tg.shape[0] == Tv.shape[0] * per, (LG.shape, tg.shape)
        res = {}
        for t_, m in (('ALL_val', keep_all), ('NOVEL_only', keep_novel)):
            l, t = LG[m], tg[m]
            ce_z = float(torch.nn.functional.cross_entropy(
                (l - l.mean(-1, keepdim=True)) / l.std(-1, keepdim=True).clamp_min(1e-6), t))
            res[t_] = {'n': int(t.numel()),
                       'acc@1': float((l.argmax(-1) == t).float().mean()),
                       'acc@5': float((l.topk(5, -1).indices == t.unsqueeze(-1)).any(-1)
                                      .float().mean()),
                       'ce_z': ce_z, 'ppl_z': math.exp(min(ce_z, 25.0))}
            print(f'  [{tag}] {t_}: acc@1 {100*res[t_]["acc@1"]:.3f}%  '
                  f'acc@5 {100*res[t_]["acc@5"]:.3f}%  CE_z {ce_z:.4f}  (n={res[t_]["n"]})',
                  flush=True)
        return res

    OUT = {'variants': {}, 'set_flash': bool(args.set_flash), 'layers': layers_idx,
           'head': args.head, 'val_windows': int(Tv.shape[0]),
           'n_all': int(keep_all.sum()), 'n_novel': int(keep_novel.sum())}
    want = [v.strip() for v in args.variants.split(',') if v.strip()]

    for v in want:
        if v == 'off':
            OUT['variants']['off'] = evaluate('off')
            if args.out:
                json.dump(OUT, open(args.out, 'w'), indent=2)
        elif v == 'shipped_force_enable':
            OUT['variants']['shipped_force_enable'] = evaluate('shipped', active_ship=layers_idx)
            if args.out:
                json.dump(OUT, open(args.out, 'w'), indent=2)
            for li in layers_idx:
                model.backbone.layers[li].self_attn.active_root_ids = None
        elif v in ('root_terms_shared', 'bonus_only_shared'):
            mods, _ = swap_in_root_bias(
                model.backbone.layers, layers_idx,
                shared_root_embed=model.morphemic_embed.root_embed,
                root_source='shared', gate_init=1.0, pillar_gate_init=0.0)
            if v == 'bonus_only_shared':
                for m in mods:
                    m.stream_mix.data[1] = 0.0
            OUT['variants'][v] = evaluate(v, native=mods)
            if args.out:
                json.dump(OUT, open(args.out, 'w'), indent=2)
            for m in mods:
                m.active_root_ids = None
        else:
            print(f'[warn] unknown variant {v!r}')

    if 'off' in OUT['variants']:
        base_a1 = OUT['variants']['off']['ALL_val']['acc@1']
        print('\n=== DELTAS vs the frozen baseline (ALL_val acc@1) ===')
        for k, r in OUT['variants'].items():
            d = 100 * (r['ALL_val']['acc@1'] - base_a1)
            print(f'  {k:24s} {100*r["ALL_val"]["acc@1"]:.3f} %   delta {d:+.3f} pp'
                  f'   (NOVEL {100*r["NOVEL_only"]["acc@1"]:.3f} %, '
                  f'delta {100*(r["NOVEL_only"]["acc@1"]-OUT["variants"]["off"]["NOVEL_only"]["acc@1"]):+.3f} pp)')
            OUT.setdefault('delta_pp_ALL', {})[k] = d
    if args.out:
        json.dump(OUT, open(args.out, 'w'), indent=2)
        print(f'[*] wrote {args.out}')


if __name__ == '__main__':
    main()
