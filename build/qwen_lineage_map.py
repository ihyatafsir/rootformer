#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_lineage_map.py -- WHICH tensors are Qwen2.5-0.5B's, and in WHICH checkpoints?

Answers two questions:
  (1) Which tensor families in the current architecture carry Qwen lineage, and which are new?
      Attention was replaced by IshtaqaqAttention and the vocabulary differs, so the transplant
      need not be uniform. For every tensor family present in both models we compute the
      same-layer vs other-layer cosine diagonal. A family that was transplanted shows a strong
      diagonal; a family that is new shows a flat matrix.
  (2) Does the lineage persist across the project's checkpoints, or did later versions diverge?
      The same MLP diagonal test is run on every *_master.safetensors / backbone checkpoint found.

A strong diagonal cannot arise by chance: independently trained high-dimensional matrices have
~0 expected cosine, and the off-diagonal here is the empirical null.

Usage:  python qwen_lineage_map.py
"""
import glob
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

os.environ.setdefault('HF_HOME', '/workspace/.hf_home')
QWEN = 'Qwen/Qwen2.5-0.5B'
CKPT_DIR = '/workspace/hf_v19_2_release/checkpoints'
CURRENT = f'{CKPT_DIR}/rootformer_v19_2_synthesis_ar_backbone.safetensors'

ROLE_RE = re.compile(r'(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|'
                     r'embed_tokens|lm_head|root_embed|morphemic|hist_emb|op_emb|'
                     r'cond_proj|feat_proj|root_head)')


def load_safetensors(path):
    from safetensors.torch import load_file
    return load_file(path)


def layer_of(key):
    m = re.search(r'\.(\d+)\.', key)
    return int(m.group(1)) if m else None


def role_of(key):
    m = ROLE_RE.search(key)
    return m.group(1) if m else None


def cos(a, b):
    return float(F.cosine_similarity(a.reshape(-1).float(), b.reshape(-1).float(), dim=0))


def diagonal_stats(ck_sd, qw_sd, role, min_layers=4):
    """same-layer vs other-layer cosine for one tensor role (2-D tensors only)."""
    def collect(sd):
        out = {}
        for k, v in sd.items():
            if v.dim() != 2 or role_of(k) != role:
                continue
            li = layer_of(k)
            if li is not None:
                out.setdefault(li, v)
        return out

    A, B = collect(ck_sd), collect(qw_sd)
    layers = sorted(set(A) & set(B))
    layers = [l for l in layers if A[l].shape == B[l].shape]
    if len(layers) < min_layers:
        return None
    diag, off = [], []
    for i in layers:
        for j in layers:
            c = cos(A[i], B[j])
            (diag if i == j else off).append(c)
    d, o = sum(diag) / len(diag), sum(off) / len(off)
    return {'n_layers': len(layers), 'shape': tuple(A[layers[0]].shape),
            'same_layer': d, 'other_layer': o, 'gap': d - o}


def main():
    from huggingface_hub import hf_hub_download
    qw = load_safetensors(hf_hub_download(QWEN, 'model.safetensors'))
    print(f'[*] qwen2.5-0.5b tensors: {len(qw)}\n')

    # ---------------- (1) per-family map on the CURRENT architecture ----------------
    ck = load_safetensors(CURRENT)
    print(f'=== current architecture: {os.path.basename(CURRENT)} ({len(ck)} tensors) ===')
    print(f'{"family":<14}{"layers":>8}{"shape":>16}{"same-layer":>12}{"other":>10}{"gap":>10}')
    print('-' * 70)
    fams = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
            'embed_tokens', 'lm_head']
    fam_res = {}
    for role in fams:
        st = diagonal_stats(ck, qw, role)
        if st is None:
            print(f'{role:<14}{"--":>8}{"(no comparable 2-D tensors)":>36}')
            continue
        fam_res[role] = st
        verdict = 'QWEN' if st['gap'] > 0.5 else ('partial' if st['gap'] > 0.15 else 'NEW')
        print(f'{role:<14}{st["n_layers"]:>8}{str(st["shape"]):>16}'
              f'{st["same_layer"]:>+12.4f}{st["other_layer"]:>+10.4f}{st["gap"]:>+10.4f}  {verdict}')

    # tensors that exist in the checkpoint but have no Qwen analogue
    ck_roles = {}
    for k, v in ck.items():
        ck_roles.setdefault(role_of(k) or 'other', 0)
        ck_roles[role_of(k) or 'other'] += 1
    print(f'\ncheckpoint tensor families (count): '
          f'{ {k: v for k, v in sorted(ck_roles.items(), key=lambda x: -x[1])} }')

    # ---------------- (2) lineage across checkpoints ----------------
    print('\n=== MLP lineage across the project\'s checkpoints (down_proj diagonal) ===')
    paths = ([CURRENT]
             + sorted(glob.glob('/workspace/rootformer_v12/v18_next_root_morph/checkpoints/*_master.safetensors'))
             + sorted(glob.glob(f'{CKPT_DIR}/*.safetensors'))
             + ['/workspace/nrmp_trained_final.safetensors'])
    seen, rows = set(), []
    for p in paths:
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        try:
            sd = load_safetensors(p)
        except Exception as e:
            print(f'  {os.path.basename(p):<58} load failed: {type(e).__name__}')
            continue
        st = diagonal_stats(sd, qw, 'down_proj', min_layers=3)
        if st is None:
            print(f'  {os.path.basename(p):<58} no comparable down_proj')
            continue
        rows.append((os.path.basename(p), st))
        print(f'  {os.path.basename(p):<58} gap {st["gap"]:+.4f}  '
              f'(same {st["same_layer"]:+.3f}, other {st["other_layer"]:+.3f}, n={st["n_layers"]})')

    json.dump({'current_families': fam_res,
               'checkpoints': [{'name': n, **s} for n, s in rows]},
              open('/workspace/qwen_lineage_map.json', 'w'), indent=2)
    print('\nwrote /workspace/qwen_lineage_map.json')


if __name__ == '__main__':
    main()
