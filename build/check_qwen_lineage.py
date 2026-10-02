#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_qwen_lineage.py -- do the Rootformer checkpoint's weights descend from Qwen2.5-0.5B?

Method
------
The architecture replaces attention with IshtiqaqAttention and uses a different vocabulary, so
attention and embedding tensors cannot be compared. The MLP blocks are shape-compatible
(hidden 896 -> intermediate 4864), and a transplant would have copied them.

The model was then trained on a small corpus (~7M tokens for a 0.5B model), so transplanted weights
would have drifted only slightly. That makes the test sensitive.

Decisive statistic: for each layer i, cosine similarity between the Rootformer MLP tensor and the
Qwen tensor from layer j, averaged over j. A genuine transplant shows a strong DIAGONAL -- (i, i)
similarity far above the off-diagonal (i, j != i) background. Independent initialisation shows a flat
matrix with no diagonal structure.

Also reports exact-equality counts, in case a tensor was copied and never touched.

Usage:  python check_qwen_lineage.py
"""
import json
import sys
from pathlib import Path

import torch

CKPT = '/workspace/hf_v19_2_release/checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors'
QWEN = 'Qwen/Qwen2.5-0.5B'


def load_ckpt():
    from safetensors.torch import load_file
    return load_file(CKPT)


def load_qwen():
    """Fetch only Qwen's state dict (no model instantiation)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import os
    os.environ.setdefault('HF_HOME', '/workspace/.hf_home')
    p = hf_hub_download(QWEN, 'model.safetensors')
    print(f'[*] qwen weights: {p}')
    return load_file(p)


def mlp_keys(sd):
    """Group MLP tensors by layer index."""
    out = {}
    for k, v in sd.items():
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            if proj in k and v.dim() == 2:
                # layer index = first integer in the key
                idx = None
                for part in k.split('.'):
                    if part.isdigit():
                        idx = int(part)
                        break
                if idx is None:
                    continue
                out.setdefault(idx, {})[proj] = (k, v.float())
    return out


def cos(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main():
    ck = load_ckpt()
    print(f'[*] checkpoint tensors: {len(ck)}')
    ck_mlp = mlp_keys(ck)
    print(f'[*] checkpoint MLP layers: {sorted(ck_mlp)[:5]} ... total {len(ck_mlp)}')

    qw = load_qwen()
    qw_mlp = mlp_keys(qw)
    print(f'[*] qwen MLP layers: total {len(qw_mlp)}')
    print(f'[*] shapes checkpoint L0: '
          f'{ {p: tuple(t.shape) for p, (_, t) in ck_mlp[min(ck_mlp)].items()} }')
    print(f'[*] shapes qwen      L0: '
          f'{ {p: tuple(t.shape) for p, (_, t) in qw_mlp[min(qw_mlp)].items()} }\n')

    layers = sorted(set(ck_mlp) & set(qw_mlp))
    if not layers:
        print('!! no overlapping layer indices -- cannot compare'); return

    results = {}
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        M = []
        for i in layers:
            if proj not in ck_mlp[i] or proj not in qw_mlp[i]:
                M.append(None); continue
            _, a = ck_mlp[i][proj]
            row = []
            for j in layers:
                _, b = qw_mlp[j][proj]
                row.append(cos(a, b) if a.shape == b.shape else float('nan'))
            M.append(row)
        if M and M[0] is not None:
            T = torch.tensor(M)
            diag = torch.diagonal(T)
            off = T[~torch.eye(len(layers), dtype=torch.bool)]
            results[proj] = {'diag_mean': float(diag.mean()),
                             'offdiag_mean': float(off.mean()),
                             'diag_max': float(diag.max()),
                             'gap': float(diag.mean() - off.mean())}
            print(f'{proj:<12} same-layer {diag.mean():+.4f}   other-layer {off.mean():+.4f}   '
                  f'GAP {diag.mean()-off.mean():+.4f}')

    # exact-equality check on every shared tensor name
    shared = [k for k in ck if k in qw]
    exact = [k for k in shared if ck[k].shape == qw[k].shape
             and torch.equal(ck[k].float(), qw[k].float())]
    print(f'\n[*] identical tensor NAMES in both: {len(shared)}')
    print(f'[*] byte-identical tensors: {len(exact)}')
    for k in exact[:10]:
        print('     ', k)

    verdict = 'NO EVIDENCE of a Qwen transplant'
    if results:
        gap = max(r['gap'] for r in results.values())
        if gap > 0.15:
            verdict = f'LINEAGE LIKELY (max same-layer gap {gap:+.3f})'
        elif gap > 0.05:
            verdict = f'WEAK/AMBIGUOUS (max gap {gap:+.3f})'
    print(f'\n==> {verdict}')
    json.dump({'results': results, 'shared_names': len(shared),
               'identical_tensors': len(exact), 'verdict': verdict},
              open('/workspace/qwen_lineage.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
