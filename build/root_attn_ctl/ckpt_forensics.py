#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ckpt_forensics.py -- READ-ONLY forensics on the confounded run's checkpoints.

Answers two questions from my brief:
  (1) is head_ROOTATTN.pt (51 MB) a usable artifact or a collapsed one?
  (2) does head_ROOTATTN.pt.trunk.pt (144 MB) hold the damaged trunk?

NOTHING IS WRITTEN TO EITHER FILE.  CPU only, so it can run while the GPU trains.
"""
import json
import os
import sys

import torch

OUT = '/workspace/root_attn'
CK = ('/workspace/hf_v19_2_release/checkpoints/'
      'rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors')
rep = {}


def tensor_report(d, tag):
    n_par = n_nan = n_inf = 0
    for k, v in d.items():
        n_par += v.numel()
        if v.is_floating_point():
            n_nan += int(torch.isnan(v).sum())
            n_inf += int(torch.isinf(v).sum())
    print(f'[{tag}] tensors={len(d)} params={n_par/1e6:.3f}M nan={n_nan} inf={n_inf}')
    return {'tensors': len(d), 'params': n_par, 'nan': n_nan, 'inf': n_inf}


# ---------------------------------------------------------------- (1) the head
hp = os.path.join(OUT, 'head_ROOTATTN.pt')
head = torch.load(hp, map_location='cpu')
print(f'=== {hp} ({os.path.getsize(hp)/1e6:.2f} MB, type={type(head).__name__}) ===')
if isinstance(head, dict):
    rep['head'] = tensor_report(head, 'head_ROOTATTN')
    print('  keys:', ', '.join(sorted(head.keys())))
    rep['head_keys'] = sorted(head.keys())
    for k in sorted(head.keys()):
        v = head[k]
        if v.is_floating_point():
            print(f'    {k:34s} {tuple(v.shape)!s:16s} {str(v.dtype):16s} '
                  f'rms={float(v.float().pow(2).mean().sqrt()):.5f} '
                  f'amax={float(v.abs().max()):.4f}')
    # the warm-start reference: the released head in the original checkpoint
    try:
        from safetensors import safe_open
        with safe_open(CK, framework='pt', device='cpu') as f:
            ks = [k for k in f.keys() if 'head' in k or k.startswith(('root_head', 'wazn_head',
                  'prefix_head', 'suffix_head', 'cond_proj'))]
        print(f'  original-checkpoint head-ish tensors: {sorted(ks)[:12]}')
        rep['orig_head_keys'] = sorted(ks)
    except Exception as e:                                     # noqa: BLE001
        print('  [warn] could not list original checkpoint keys:', e)
else:
    print('  UNEXPECTED payload:', str(head)[:300])

# ------------------------------------------------- (2) the trunk / RCA payload
tp = os.path.join(OUT, 'head_ROOTATTN.pt.trunk.pt')
blob = torch.load(tp, map_location='cpu')
print(f'\n=== {tp} ({os.path.getsize(tp)/1e6:.2f} MB, type={type(blob).__name__}) ===')
if isinstance(blob, dict) and 'state' in blob:
    meta = {k: v for k, v in blob.items() if k != 'state'}
    print('  metadata:', json.dumps(meta, default=str))
    rep['trunk_meta'] = {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                         for k, v in meta.items()}
    st = blob['state']
    rep['trunk_state'] = tensor_report(st, 'head_ROOTATTN.pt.trunk')
    rca = {k: v for k, v in st.items() if k.startswith('root_cross.')}
    tr = {k: v for k, v in st.items() if not k.startswith('root_cross.')}
    print(f'  root_cross.* tensors={len(rca)} params={sum(v.numel() for v in rca.values())/1e6:.3f}M'
          f'   trunk tensors={len(tr)} params={sum(v.numel() for v in tr.values())/1e6:.3f}M')
    print('  layers present:',
          sorted({int(k.split(".")[2]) for k in tr if k.startswith('backbone.layers.')}))
    print('  --- RCA gates and o_proj scale (the trained values) ---')
    for i in range(8):
        gk = f'root_cross.mods.{i}.gate'
        if gk not in st:
            continue
        ok = f'root_cross.mods.{i}.o_proj.weight'
        qk = f'root_cross.mods.{i}.q_proj.weight'
        print(f'    layer[{i}] gate={float(st[gk].float()):+.6f}'
              f'  ||o_proj||_rms={float(st[ok].float().pow(2).mean().sqrt()):.6f}'
              f'  ||q_proj||_rms={float(st[qk].float().pow(2).mean().sqrt()):.6f}')
        rep.setdefault('gates', {})[f'layer{i}'] = float(st[gk].float())
    # how far did the unfrozen trunk layers move from the RELEASED weights?
    try:
        from safetensors import safe_open
        print('  --- unfrozen trunk layers 20-23 vs the released checkpoint ---')
        with safe_open(CK, framework='pt', device='cpu') as f:
            okeys = set(f.keys())
            for L in (20, 21, 22, 23):
                num = den = 0.0
                n_missing = 0
                for k, v in tr.items():
                    pre = f'backbone.layers.{L}.'
                    if not k.startswith(pre):
                        continue
                    cand = [c for c in (k, 'backbone.' + k, k.replace('backbone.layers.', 'layers.'))
                            if c in okeys]
                    if not cand:
                        n_missing += 1
                        continue
                    o = f.get_tensor(cand[0]).to(torch.float32)
                    a = v.to(torch.float32)
                    if a.shape != o.shape:
                        n_missing += 1
                        continue
                    num += float((a - o).pow(2).sum())
                    den += float(o.pow(2).sum())
                rel = (num ** 0.5) / max(den ** 0.5, 1e-12)
                print(f'    layer {L}: ||dW||/||W|| = {rel:.6f}   (unmatched keys {n_missing})')
                rep.setdefault('layer_rel_change', {})[f'layer{L}'] = rel
    except Exception as e:                                     # noqa: BLE001
        print('  [warn] trunk comparison failed:', repr(e))
else:
    print('  UNEXPECTED payload:', str(blob)[:300])

json.dump(rep, open('/workspace/root_attn_ctl/ckpt_forensics.json', 'w'), indent=2, default=str)
print('\nwrote /workspace/root_attn_ctl/ckpt_forensics.json')
