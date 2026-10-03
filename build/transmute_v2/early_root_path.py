#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
early_root_path.py -- attach the root pathway from the INPUT STAGE UPWARD.

WHY EARLY
---------
Measured on the raw base, a single word in isolation, 6,000 words / 2,807 roots:

    layer  4  21.83 %      layer 12  14.17 %      layer 20  13.58 %
    layer  8  16.50 %      layer 16  12.92 %      layer 23  13.00 %
    transmuted trunk 92.34 %, MONOTONE UPWARD to layer 23

Root information DECAYS with depth on an unadapted base: a raw base organises for next-token
prediction over a 152k BPE vocabulary and loses root structure on the way down.  Attaching root
attention at layers 20-23 therefore reads it where it is WEAKEST (13.00 % vs 21.83 % at layer 4).
The morphemic adaptation is what reversed the decay; the corrected arm re-attaches from layer 0
so the pathway sees the root signal while it is still there and can MAINTAIN it downward.

TWO MECHANISMS, COMPLEMENTARY, NOT REDUNDANT (both built here)
--------------------------------------------------------------
  score bias   total_score += mix*cond*score_root + identical_root_bonus
               added to the attention SCORE before softmax; score_root is a function of
               (r_i, r_j) only, never of h -> REWEIGHTS the value stream, carries NO new
               information.  Implemented by `ishtiqaq_root_bias.IshtiqaqRootBiasAttention`
               (a gated subclass; the shipped module file is NOT edited).
  residual     h <- h + gate*LayerNorm(o_proj(softmax(Wq.h . Wk.E(r)) @ Wv.E(r)))
               a different insertion point AND a different source -> INJECTS.
               Implemented by `root_cross_attn.RootHistoryCrossAttention`.

Both read the SAME K/V source: the RELEASED 448-dim `morphemic_embed.root_embed` (9490 x 448).
The stale per-layer `self_attn.root_embed` (9015 x 64) is NEVER used: 475 live ids alias onto
row 9014 there (5.01 %), and the table is at N(0,1) default init because the native path never
received a single gradient (proved: stream_mix[1].grad == 0.0 exactly in all 24 layers).

SCHEDULES
---------
  all       -> 0..23          "from the input stage upward" (the arm-C/C-vs-A test)
  earlyN    -> 0..N-1         a declared early-start schedule (early12 = the bottom half)
  everyK    -> K-1, 2K-1, ... a declared period
  topN/midN -> the shipped late schedules, kept for continuity
  a,b,c     -> explicit
"""
import sys
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from root_space import LIVE_ROOTS, LIVE_AWZAN, SpaceViolation

if '/workspace/root_attn' not in sys.path:
    sys.path.insert(0, '/workspace/root_attn')


def parse_layer_spec_v13(spec: str, n_layers: int) -> List[int]:
    """`none` | `all` | `earlyN` | `topN` | `midN` | `everyK` | explicit `a,b,c`."""
    s = str(spec).strip().lower()
    if s in ('', 'none', 'off', '0'):
        return []
    if s == 'all':
        return list(range(n_layers))
    if s.startswith('early'):
        k = int(s[5:] or (n_layers // 2))
        return list(range(min(k, n_layers)))
    if s.startswith('top'):
        k = int(s[3:] or 1)
        return list(range(max(0, n_layers - k), n_layers))
    if s.startswith('mid'):
        k = int(s[3:] or 1)
        st = max(0, (n_layers - k) // 2)
        return list(range(st, st + k))
    if s.startswith('every'):
        k = int(s[5:] or 4)
        return list(range(k - 1, n_layers, k))
    return sorted({int(x) for x in s.split(',') if x.strip()})


def _guard_root_source(root_embed: nn.Embedding, where: str) -> Dict[str, int]:
    rows, dim = int(root_embed.weight.shape[0]), int(root_embed.weight.shape[1])
    if rows != LIVE_ROOTS:
        raise SpaceViolation(
            f'{where}: the K/V root source has {rows} rows, the live root space is {LIVE_ROOTS}. '
            f'Refusing to attach a pathway in a mixed id space.')
    return {'rows': rows, 'dim': dim}


def attach_early_root_residual(model, layers: nn.ModuleList, layer_indices: Sequence[int],
                               root_embed: nn.Embedding, num_heads: int = 8,
                               dropout: float = 0.0, out_std: float = 1e-3,
                               out_norm: bool = False, exclude_current: bool = False,
                               dtype: torch.dtype = torch.float32):
    """The RESIDUAL injector (h <- h + gate*LN(o_proj(softmax(Wq h . Wk E(r)) @ Wv E(r)))).

    Returns (stack, layers).  `stack.gate` is exactly 0 at init, so the attached stack is a
    BITWISE no-op; `--rca-ablate-eval` / `zero_gates()` gives the causal ablation.
    """
    from root_cross_attn import RootCrossAttentionStack
    info = _guard_root_source(root_embed, 'residual injector')
    idx = [int(i) for i in layer_indices]
    st = RootCrossAttentionStack(
        model.d_model, root_embed, idx, num_heads=num_heads, dropout=dropout,
        out_std=out_std, dtype=dtype, out_norm=out_norm).to(
        device=root_embed.weight.device)
    st.exclude_current = bool(exclude_current)
    st.attach(layers)
    return st, {'kind': 'residual_injector', 'layers': idx, **info,
                'num_heads': num_heads, 'out_norm': bool(out_norm),
                'exclude_current': bool(exclude_current),
                'new_params_M': round(sum(p.numel() for p in st.parameters()) / 1e6, 3)}


def attach_native_root_bias(model, layers: nn.ModuleList, layer_indices: Sequence[int],
                            root_embed: nn.Embedding, gate_init: float = 0.0,
                            pillar_gate_init: float = 0.0):
    """The SCORE-BIAS mechanism, gated, reading the SHARED 448-dim table.

    `root_source='shared'` is not a preference: with `native` the keys are the stale 9015x64
    table and 475 ids alias onto row 9014.
    """
    if '/workspace/ishtiqaq_check' not in sys.path:
        sys.path.insert(0, '/workspace/ishtiqaq_check')
    from ishtiqaq_root_bias import swap_in_root_bias
    info = _guard_root_source(root_embed, 'score-bias attention')
    mods, params = swap_in_root_bias(
        layers, [int(i) for i in layer_indices], shared_root_embed=root_embed,
        root_source='shared', gate_init=gate_init, pillar_gate_init=pillar_gate_init)
    return mods, params, {'kind': 'native_score_bias', 'layers': [int(i) for i in layer_indices],
                          **info, 'gate_init': gate_init,
                          'pillar_gate_init': pillar_gate_init,
                          'new_params_M': round(sum(p.numel() for p in params) / 1e6, 3)}
