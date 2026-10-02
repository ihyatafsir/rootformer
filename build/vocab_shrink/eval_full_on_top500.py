#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_full_on_top500.py -- the APPLES-TO-APPLES baseline for the shrunk-vocab experiment.

Loads the FULL 9490-class FIX head (`/workspace/head_fix/head_ALIGNED_FIX.pt`, the aligned-cache
20k arm) and evaluates it on exactly the 300 val windows / 18 869 non-special positions that
every 20k arm uses, then splits those SAME positions into

    (i)  all non-special positions                -> the published full-vocab number
    (ii) positions whose gold root is in the top-500 train-frequency set -> the SAME position
         subset the shrunk arm trains and is scored on

so the shrunk arm's accuracy can be compared with the full arm's accuracy on the identical
positions, not merely with a marginal.  Read-only on every shipped file.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

RELEASE = Path('/workspace/hf_v19_2_release')
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'models'))
import nrmt_train as T                                                    # noqa: E402


def starts_for(t, n, seed):
    """EXACT copy of the nested helper in nrmt_train.main() -- same windows, same seed."""
    s = list(range(0, t.numel() - T.WIN - 1, T.STRIDE))
    np.random.default_rng(seed).shuffle(s)
    return sorted(s[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--head', default='/workspace/head_fix/head_ALIGNED_FIX.pt')
    ap.add_argument('--root-map', default='/workspace/vocab_shrink/root_map_top500.pt')
    ap.add_argument('--out', default='/workspace/vocab_shrink/eval_full_on_top500.json')
    args = ap.parse_args()
    t0 = time.time()

    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from safetensors.torch import load_file
    from nrmt_arch import RootformerNRMT, build_operator_table, NRMTHead   # noqa: F401

    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json')
    vocab = V(bp)
    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16, hist=3, dropout=0.1,
                           use_features=True, feat_gate=True).to(device)
    ck = (RELEASE / 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.'
          'roots9490.tok10052.safetensors')
    missing, unexpected = model.load_state_dict(load_file(str(ck)), strict=False)
    print(f'[*] backbone {ck.name}: missing={len(missing)} unexpected={len(unexpected)}')
    for p in model.parameters():
        p.requires_grad = False
    model.nrmt_head.to(torch.float32)
    sd = torch.load(args.head, map_location='cpu')
    r = model.nrmt_head.load_state_dict(sd, strict=True)
    print(f'[*] head {Path(args.head).name}: {len(sd)} tensors, {r}')
    head = model.nrmt_head
    head.eval()
    model.eval()

    va4 = [t.long() for t in torch.load(Path(args.cache) / 'val.pt', map_location='cpu')]
    va_st = starts_for(va4[1], 300, 1)

    with torch.no_grad():
        Hva = T.extract_backbone(model, va4, va_st, device, 'val')

    def win_tensor(t):
        return torch.stack([t[j:j + T.WIN] for j in va_st])

    Tv, Wv, Pv, Sv = (win_tensor(va4[i]) for i in (1, 2, 0, 3))
    Rt_va = Tv[:, 1:].contiguous()
    Hva_w = Hva[:, :-1, :].contiguous()

    op_table = build_operator_table(vocab)
    try:
        from sibawayh_governor import SibawayhGovernor
        gov = SibawayhGovernor(vocab)
        Ova_w = torch.tensor([gov.op_ids_for_ids(rr, ww, pp)
                              for pp, rr, ww in zip(Pv.tolist(), Tv.tolist(), Wv.tolist())],
                             dtype=torch.long)[:, :-1].contiguous()
        op_src = 'SibawayhGovernor'
    except Exception as e:                                                # pragma: no cover
        print(f'[warn] governor unavailable: {e!r}')
        Ova_w = op_table[Tv.clamp(min=0)][:, :-1].contiguous()
        op_src = f'fallback {e!r}'

    lg = []
    with torch.no_grad():
        for i in range(0, Hva_w.shape[0], 16):
            sl = slice(i, min(i + 16, Hva_w.shape[0]))
            out = head(Hva_w[sl].to(device), Tv[sl, :-1].to(device), Ova_w[sl].to(device),
                       Wv[sl, :-1].to(device), Pv[sl, :-1].to(device), Sv[sl, :-1].to(device))
            lg.append(out['root_logits'].float().cpu())
    LG = torch.cat(lg).reshape(-1, vocab.num_roots)
    tg = Rt_va.reshape(-1)
    assert LG.shape[0] == tg.shape[0]

    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r_ in enumerate(vocab.roots_list) if r_.startswith('<P:')})
    keep_all = ~torch.isin(tg, torch.tensor(specials))

    rm = torch.load(args.root_map, map_location='cpu')
    in_topk = torch.zeros(vocab.num_roots, dtype=torch.bool)
    in_topk[rm['class_to_orig'].long()] = True
    sub = keep_all & in_topk[tg]                    # the shrunk arm's exact position subset

    def metrics(mask):
        l, t = LG[mask], tg[mask]
        if t.numel() == 0:
            return None
        c = torch.bincount(t, minlength=vocab.num_roots)
        marg = float(c.max()) / int(t.numel())
        a1 = float((l.argmax(-1) == t).float().mean())
        a5 = float((l.topk(5, -1).indices == t.unsqueeze(-1)).any(-1).float().mean())
        sd_ = l.std(-1, keepdim=True).clamp_min(1e-6)
        cez = float(torch.nn.functional.cross_entropy(
            (l - l.mean(-1, keepdim=True)) / sd_, t))
        pred = torch.bincount(l.argmax(-1), minlength=vocab.num_roots)
        top_pred = int(pred.argmax())
        return {'n': int(t.numel()), 'acc@1': a1, 'acc@5': a5, 'ce_z': cez,
                'majority_class': vocab.roots_list[int(c.argmax())], 'marginal': marg,
                'lift': a1 / marg,
                'most_predicted_class': vocab.roots_list[top_pred],
                'most_predicted_share': float(pred.max()) / int(t.numel()),
                'pred_is_marginal_class': bool(top_pred == int(c.argmax()))}

    RES = {'head': str(args.head), 'cache': str(args.cache), 'operator_stream': op_src,
           'full_space_ALL': metrics(keep_all),
           'top500_subset_same_positions': metrics(sub),
           'root_map': str(args.root_map),
           'peak_vram_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
           'wall_s': round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps(RES, indent=2, ensure_ascii=False))
    for k in ('full_space_ALL', 'top500_subset_same_positions'):
        v = RES[k]
        print(f'  {k}: n={v["n"]} acc@1 {100*v["acc@1"]:.3f}% acc@5 {100*v["acc@5"]:.3f}% '
              f'CE_z {v["ce_z"]:.4f} | marginal {100*v["marginal"]:.3f}% '
              f'({v["majority_class"]}) | LIFT acc@1/marginal = {v["lift"]:.3f}x | '
              f'most predicted {v["most_predicted_class"]} '
              f'{100*v["most_predicted_share"]:.2f}%')
    print(f'[*] wrote {args.out} in {RES["wall_s"]}s')


if __name__ == '__main__':
    main()
