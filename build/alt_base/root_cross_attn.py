#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
root_cross_attn.py -- root-history cross-attention wired into the TRUNK.

Measured motivation (identical val positions, 18,869 radical / 15,046 NOVEL)
----------------------------------------------------------------------------
  * next-word root, linear readout of a single h_t ..............  6.48 %   (ceiling)
  * order-4 root n-gram LOOKUP over the root history ............ 53.13 %
  * current-word root, linear readout of h_t ................... 92.34 %
The trunk provably CONTAINS the root (92.34 %); what it cannot expose through one h_t is
a *history-dependent* readout.  A single h_t is a linear bottleneck (6.48 %); the root
sequence is not (53.13 %).  A DISCRETE order-4 table is expressible enough (53.13 %) but
trains to 6.06 % and declines, because 92.7 % of order-4 contexts occur exactly once in
train -- a table can only memorise singletons.  A soft, learned attention over root
EMBEDDINGS is the continuous relaxation of that lookup: كتب/كاتب/مكتوب/كتابة share
representation and similar contexts share attention mass, so it can generalise off the
singleton.  Cardinality is not the constraint either (9,490 -> 500 classes left the lift
flat at 1.86x), so the limitation is upstream of the head: it is the input pathway.

The operation
-------------
For a selected trunk layer i, AFTER its token-level self-attention + MLP block:

    h <- h + gate * RootCrossAttention( Q = W_q . h , K = W_k . E_root(r_{0..t}) ,
                                        V = W_v . E_root(r_{0..t}) )

Queries come from the TOKEN stream; keys and values come from the ROOT HISTORY.

Causality / which roots are visible
-----------------------------------
The mask is causal over root positions (j <= t): root r_t IS visible to h_t.  r_t is not a
leak -- it is an INPUT of this model at position t (morphemic_embed is fed
(prefix_t, root_t, wazn_t, suffix_t)); the target is r_{t+1}.  It is also required for
fidelity to the number this module exists to approximate: the order-4 lookup that scores
53.13 % uses the context r_{t-3..t} ENDING at t (`nrmt_arch.DiscreteNgramFeatures._codes`
documents exactly this), and order-1 = r_t alone already scores 1.74 % while order-2
(r_{t-1}, r_t) jumps to 11.81 %.  Handing the module only r_{<t} would deny it the single
most informative element of the context it is meant to generalise.  `--rca-exclude-current`
reproduces the strictly-past r_{t-w..t-1} variant for the ablation.

Why a DEDICATED module and not IshtiqaqAttentionV12's root_q_proj / root_k_proj
-------------------------------------------------------------------------------
IshtiqaqAttentionV12 already owns, per layer, a trained `root_embed` (9015x64),
`root_q_proj` (64->896) and `root_k_proj` (64->128), and in the NRMT path they are dead
code: the trunk is driven by `model.backbone(inputs_embeds=emb)`, which never sets
`active_root_ids`, so 0 of 456 self-attention calls carry root ids (measured).  They do NOT
fit the operation above, for three structural reasons:

  1. Q must come from the TOKEN stream.  `root_q_proj`'s domain is `root_embed`, so it can
     only produce queries from roots -- root-to-root attention, not root-history attention
     from h.
  2. K AND V must both come from the root history.  There V is `v_proj(hidden_states)` --
     the value path is SURFACE, not radical.
  3. The root term is a FUSED additive bias on the surface attention SCORES
     (`total_score += stream_mix[1] * ishtiqaq_cond * score_root`), i.e. it re-weights the
     token attention, it is not a separate residual stream that can carry root history into
     h.

What IS reused: the RELEASED, synthesis-trained root embedding table
`morphemic_embed.root_embed` (the same table the -- also dead -- linear conditioning branch
reads) is the K/V source.  Keeping 24 per-layer copies instead would add 24 x 9015 x 448 =
96.9 M new, untrained embedding parameters and would fragment one root space into 24.

Everything here is OFF unless the trainer explicitly attaches a stack, and the stack's
residual gate is zero-initialised, so attached-but-untrained is a numerical no-op.
"""
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RootHistoryCrossAttention(nn.Module):
    """h + gate * MHA(Q=W_q h, K=V=W_{k,v} E_root(root history)), causal over roots.

    All arithmetic is float32 (the trunk is bf16); the residual is cast back to the trunk
    dtype before the add.  `gate` starts at exactly 0 and `o_proj` at a small random value,
    so the module is EXACTLY a no-op at init while `d(loss)/d(gate) != 0` (the same
    dead-saddle escape that the v18fix `--feat-gate` measurement established: zeroing BOTH
    o_proj and gate makes every branch gradient exactly 0 forever).
    """

    def __init__(self, d_model: int, d_root: int, num_heads: int = 8,
                 head_dim: Optional[int] = None, dropout: float = 0.0,
                 out_std: float = 1e-3, proj_std: float = 0.02,
                 layer_idx: int = -1, dtype: torch.dtype = torch.float32,
                 out_norm: bool = False):
        super().__init__()
        head_dim = int(head_dim or (d_model // num_heads))
        if num_heads * head_dim != d_model:
            raise ValueError(f'num_heads*head_dim must equal d_model '
                             f'({num_heads}*{head_dim} != {d_model})')
        self.d_model, self.d_root = int(d_model), int(d_root)
        self.num_heads, self.head_dim, self.layer_idx = int(num_heads), int(head_dim), int(layer_idx)
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.out_norm_enabled = bool(out_norm)

        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_root, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_root, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=False, dtype=dtype)
        self.q_norm = nn.LayerNorm(self.head_dim, dtype=dtype)
        self.k_norm = nn.LayerNorm(self.head_dim, dtype=dtype)
        # OPTIONAL magnitude bound on the injected residual.  `NRMTHead._linear_features` puts a
        # LayerNorm (`feat_norm`) directly after its projection, and the v18fix measurement is
        # explicit that this is what keeps an additive branch from inflating h.  Without it the
        # residual `gate * o_proj(ctx)` is UNBOUNDED: the gate is a free scalar driven by Adam, so
        # the branch magnitude is not controlled by anything.  MEASURED consequence in the first
        # 20k arm (unfrozen trunk, no bound): total memorisation of the 381,000 training positions
        # (train root CE 0.05 vs the frozen arm's 4.23) and held-out acc@1 0.23 %.
        self.out_norm = nn.LayerNorm(d_model, dtype=dtype) if out_norm else None
        self.drop = nn.Dropout(dropout)
        # zero-init gate: the residual is EXACTLY 0 at init, whatever the trunk does.
        self.gate = nn.Parameter(torch.zeros((), dtype=dtype))

        with torch.no_grad():
            for lin in (self.q_proj, self.k_proj, self.v_proj):
                lin.weight.normal_(0.0, proj_std)
            self.o_proj.weight.normal_(0.0, out_std)

    def forward(self, h: torch.Tensor, root_ids: torch.Tensor,
                root_weight: torch.Tensor,
                exclude_current: bool = False) -> torch.Tensor:
        """h:[B,T,d_model] (any dtype), root_ids:[B,T] int, root_weight:[R,d_root].

        Returns the float32 residual delta (NOT yet added), so the caller controls the
        dtype of the residual sum.
        """
        B, T, _ = h.shape
        H, D = self.num_heads, self.head_dim
        hf = h.float()
        q = self.q_norm(self.q_proj(hf).view(B, T, H, D)).transpose(1, 2)      # [B,H,T,D]

        safe = root_ids.clamp_(0, root_weight.shape[0] - 1)
        r = root_weight.index_select(0, safe.reshape(-1)).view(B, T, -1).float()
        k = self.k_norm(self.k_proj(r).view(B, T, H, D)).transpose(1, 2)       # [B,H,T,D]
        v = self.v_proj(r).view(B, T, H, D).transpose(1, 2)                    # [B,H,T,D]

        att = torch.matmul(q, k.transpose(-1, -2)) * self.scale               # [B,H,T,T]
        pos = torch.arange(T, device=h.device)
        causal = pos.unsqueeze(0) >= pos.unsqueeze(1)                          # j <= i
        if exclude_current:
            causal = pos.unsqueeze(0) > pos.unsqueeze(1)                       # j <  i
        att = att.masked_fill(~causal, float('-inf'))
        # a row with no visible key can only happen for the strictly-past variant at t=0
        att = torch.nan_to_num(att.softmax(dim=-1), nan=0.0)
        ctx = torch.matmul(att, v).transpose(1, 2).reshape(B, T, H * D)
        out = self.o_proj(ctx)
        if self.out_norm is not None:
            out = self.out_norm(out)
        return self.gate.float() * self.drop(out)

    @torch.no_grad()
    def gate_value(self) -> float:
        return float(self.gate.detach().float())


class RootCrossAttentionStack(nn.Module):
    """Owns one `RootHistoryCrossAttention` per selected trunk layer and hooks them in.

    The module is registered on the OWNING model (so its parameters appear in
    `named_parameters()` and can be selected for the optimizer) while the K/V embedding
    table is bound with `object.__setattr__` so the shared released `root_embed` is NOT
    registered a second time.
    """

    def __init__(self, d_model: int, root_embed: nn.Embedding, layer_indices: Sequence[int],
                 num_heads: int = 8, head_dim: Optional[int] = None, dropout: float = 0.0,
                 out_std: float = 1e-3, dtype: torch.dtype = torch.float32,
                 out_norm: bool = False):
        super().__init__()
        self.layer_indices = [int(i) for i in layer_indices]
        d_root = int(root_embed.weight.shape[1])
        self.mods = nn.ModuleList([
            RootHistoryCrossAttention(d_model, d_root, num_heads=num_heads, head_dim=head_dim,
                                      dropout=dropout, out_std=out_std, layer_idx=i,
                                      dtype=dtype, out_norm=out_norm)
            for i in self.layer_indices])
        object.__setattr__(self, '_root_embed', root_embed)   # shared, NOT re-registered
        self.exclude_current = False
        self._root_ids: Optional[torch.Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self.counts = {'calls': 0, 'carrying_root_ids': 0}

    # ------------------------------------------------------------------ wiring
    def attach(self, layers: nn.ModuleList) -> None:
        """Register a forward hook per selected layer.  Idempotent per stack object."""
        if self._handles:
            raise RuntimeError('RootCrossAttentionStack.attach called twice')
        for pos, idx in enumerate(self.layer_indices):
            self._handles.append(layers[idx].register_forward_hook(self._make_hook(pos)))
        print(f'[*] root cross-attention attached to trunk layers {self.layer_indices} '
              f'({sum(p.numel() for p in self.parameters())/1e6:.2f}M new params, '
              f'd_root={self.d_root}, heads={self.mods[0].num_heads}x{self.mods[0].head_dim}, '
              f'observe={"r_{<=t}" if not self.exclude_current else "r_{<t}"})', flush=True)

    @property
    def d_root(self) -> int:
        return int(self._root_embed.weight.shape[1])

    def _make_hook(self, pos: int):
        mod = self.mods[pos]

        def hook(module, args, output):
            self.counts['calls'] += 1
            hs = output[0] if isinstance(output, tuple) else output
            r = self._root_ids
            if r is None or hs is None:
                return output
            if r.shape[0] != hs.shape[0] or r.shape[1] != hs.shape[1]:
                raise RuntimeError(f'root_ids {tuple(r.shape)} do not match hidden states '
                                   f'{tuple(hs.shape)} at layer {mod.layer_idx}')
            self.counts['carrying_root_ids'] += 1
            delta = mod(hs, r, self._root_embed.weight,
                        exclude_current=self.exclude_current)
            out2 = (hs.float() + delta).to(hs.dtype)
            return (out2,) + output[1:] if isinstance(output, tuple) else out2

        return hook

    def set_root_ids(self, root_ids: Optional[torch.Tensor]) -> None:
        self._root_ids = root_ids

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def gate_values(self) -> List[float]:
        return [m.gate_value() for m in self.mods]

    @torch.no_grad()
    def zero_gates(self) -> List[float]:
        """Temporarily force every gate to 0 (inference-time ablation); returns the old values."""
        old = [float(m.gate.detach()) for m in self.mods]
        for m in self.mods:
            m.gate.data.zero_()
        return old

    @torch.no_grad()
    def restore_gates(self, old: Sequence[float]) -> None:
        for m, v in zip(self.mods, old):
            m.gate.data.fill_(float(v))

    @torch.no_grad()
    def out_rms(self, h: torch.Tensor, root_ids: torch.Tensor) -> float:
        """RMS of the (gated) residual the stack injects, measured on one batch."""
        tot, n = 0.0, 0
        for m in self.mods:
            d = m(h, root_ids, self._root_embed.weight, exclude_current=self.exclude_current)
            tot += float(d.float().pow(2).mean().sqrt())
            n += 1
        return tot / max(n, 1)

    def stats(self) -> Dict[str, object]:
        g = self.gate_values()
        return {'rca_layers': self.layer_indices, 'rca_gate': g,
                'rca_gate_absmean': sum(abs(x) for x in g) / max(len(g), 1),
                'rca_calls': self.counts['calls'],
                'rca_calls_with_root_ids': self.counts['carrying_root_ids']}


def parse_layer_spec(spec: str, n_layers: int) -> List[int]:
    """`none` | `topN` | `midN` | `everyK` | explicit `a,b,c`."""
    spec = str(spec).strip().lower()
    if spec in ('', 'none', 'off', '0'):
        return []
    if spec.startswith('top'):
        k = int(spec[3:] or 1)
        return list(range(max(0, n_layers - k), n_layers))
    if spec.startswith('mid'):
        k = int(spec[3:] or 1)
        start = max(0, (n_layers - k) // 2)
        return list(range(start, start + k))
    if spec.startswith('every'):
        k = int(spec[5:] or 4)
        return list(range(k - 1, n_layers, k))
    return sorted({int(x) for x in spec.split(',') if x.strip()})
