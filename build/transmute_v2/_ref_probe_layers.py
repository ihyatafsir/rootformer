#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_layers.py -- did prior training / calibration damage the upper layers' root heads?

Two independent tests:

  A) LAYER-WISE ROOT PROBE
     Run the backbone with output_hidden_states and apply the released root head to every
     layer's hidden state (after the model's final_norm). If next-root accuracy peaks in a
     MIDDLE layer and degrades toward the top, the upper layers were damaged/drifted away
     from root-relevant representations by later training (e.g. the v19.x LLM-realizer
     transmutation objective). If it is monotone increasing to the last layer, they are intact.

  B) WEIGHT SANITY SWEEP
     Per-tensor statistics over the checkpoint: NaN/Inf, all-zero tensors, and per-layer
     weight norms for backbone and heads. Catches genuinely destroyed/dead parameters.

Usage:
  python probe_layers.py --checkpoint /workspace/nrmp_trained_final.safetensors
  python probe_layers.py --compare /workspace/nrmp_trained_final.safetensors \
      --checkpoint checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))


@torch.no_grad()
def layer_profile(model, vocab, windows, device, specials, max_windows=120):
    P, R, W, S = windows
    n = min(max_windows, P.shape[0])
    prof = None
    head = model.nrmp_head.root_head
    spec = torch.tensor(sorted(specials), device=device)

    for i in range(n):
        p, r, w, s = (x[i:i + 1].to(device) for x in (P, R, W, S))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb = model.morphemic_embed(p, r, w, s)
            out = model.backbone(inputs_embeds=emb, output_hidden_states=True)
        hs = out.hidden_states                      # (embedding_output, layer1..layerN)
        tgt = r[0, 1:]
        keep = ~torch.isin(tgt, spec)
        if keep.sum() == 0:
            continue
        if prof is None:
            prof = {j: {'n': 0, 'h1': 0, 'h5': 0} for j in range(len(hs))}
        for j, hj in enumerate(hs):
            hn = model.final_norm(hj.to(torch.bfloat16))[0, :-1, :].float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = head(hn.to(torch.bfloat16))[keep].float()
            del hn
            tg = tgt[keep]
            prof[j]['n'] += int(tg.numel())
            prof[j]['h1'] += int((lg.argmax(-1) == tg).sum().item())
            k = min(5, lg.shape[-1])
            prof[j]['h5'] += int((lg.topk(k, -1).indices == tg.unsqueeze(-1)).any(-1).sum().item())

    return {j: {'n': v['n'], 'acc@1': v['h1'] / max(v['n'], 1),
                'acc@5': v['h5'] / max(v['n'], 1)} for j, v in (prof or {}).items()}


def weight_sweep(path):
    from safetensors.torch import load_file
    sd = load_file(str(path))
    rows = []
    for k, v in sd.items():
        f = v.float()
        rows.append({
            'key': k,
            'shape': list(v.shape),
            'dtype': str(v.dtype).replace('torch.', ''),
            'nan': int(torch.isnan(f).sum().item()),
            'inf': int(torch.isinf(f).sum().item()),
            'all_zero': bool((f == 0).all().item()),
            'norm': float(f.norm().item()),
            'absmax': float(f.abs().max().item()) if f.numel() else 0.0,
            'std': float(f.std().item()) if f.numel() > 1 else 0.0,
        })
    n_nan = sum(1 for r in rows if r['nan'])
    n_inf = sum(1 for r in rows if r['inf'])
    n_zero = sum(1 for r in rows if r['all_zero'])
    print(f'  tensors={len(rows)}  with NaN={n_nan}  with Inf={n_inf}  all-zero={n_zero}')
    if n_nan or n_inf or n_zero:
        for r in rows:
            if r['nan'] or r['inf'] or r['all_zero']:
                print(f"    PROBLEM {r['key']}: nan={r['nan']} inf={r['inf']} zero={r['all_zero']}")
    layers = {}
    for r in rows:
        m = None
        for tag in ('backbone.layers.', 'layers.'):
            if r['key'].startswith(tag):
                m = int(r['key'][len(tag):].split('.')[0])
        if m is not None:
            layers.setdefault(m, []).append(r['norm'])
    if layers:
        print('  per-layer total weight norm:')
        for m in sorted(layers):
            tot = sum(layers[m])
            print(f'    layer {m:2d}: norm={tot:9.2f}  ({len(layers[m])} tensors)')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='/workspace/nrmp_trained_final.safetensors')
    ap.add_argument('--compare', nargs='*', default=[],
                    help='additional checkpoints to profile for comparison')
    ap.add_argument('--cache', default='/workspace/nrmp_cache')
    ap.add_argument('--out', default='/workspace/probe_layers.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from nrmp_run import load_engine
    from nrmp_adapter import make_windows

    results = {}

    for path in [args.checkpoint] + list(args.compare):
        if not Path(path).exists():
            print(f'[skip] {path} not found')
            continue
        print('\n' + '=' * 78)
        print(f'CHECKPOINT {path}')
        print('=' * 78)
        print('\n[B] weight sanity sweep')
        weight_sweep(path)

        print('\n[A] layer-wise next-root probe')
        model, vocab, device, _ = load_engine(ckpt=path)
        va = [t.long() for t in torch.load(Path(args.cache) / 'val.pt', map_location='cpu')]
        VP, VR, VW, VS = make_windows(va, 'val', device='cpu')
        specials = {vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                    vocab.root2id['<PARTICLE>']}
        specials |= {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')}
        prof = layer_profile(model, vocab, (VP, VR, VW, VS), device, specials)
        results[path] = {'profile': prof}
        print(f'  {"block":<28}{"acc@1":>9}{"acc@5":>9}')
        for j in sorted(prof):
            label = 'embedding' if j == 0 else f'layer {j-1}'
            if j in (0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, len(prof) - 1):
                print(f'  {label:<28}{100*prof[j]["acc@1"]:>9.2f}{100*prof[j]["acc@5"]:>9.2f}')
        if prof:
            best = max(prof, key=lambda j: prof[j]['acc@1'])
            last = max(prof)
            print(f'  => best block: {"embedding" if best==0 else f"layer {best-1}"} '
                  f'({100*prof[best]["acc@1"]:.2f}%)  |  final block: layer {last-1} '
                  f'({100*prof[last]["acc@1"]:.2f}%)')
            if best < last - 2:
                print('  => WARNING: root information PEAKS in a middle layer and degrades '
                      'toward the top: upper layers look drifted/damaged')
            else:
                print('  => monotone/near-monotone: no evidence that upper layers were destroyed')

    json.dump(results, open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
