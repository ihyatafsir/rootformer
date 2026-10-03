#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ishtiqaq_root_bias.py -- make IshtiqaqAttentionV12's NATIVE root path gated, ablatable and
trainable, WITHOUT changing the shipped module.

The shipped module (`models/ishtiqaq_attention_v12.py`, md5 d192ac9f...) is left BIT-IDENTICAL.
This file defines a subclass and an in-place swap helper, so:

  * with the new flag OFF the trainer never imports this file -> behaviour is literally the
    old code object, not a re-derivation of it;
  * with the flag ON and `root_gate == 0` the native root path is a bitwise no-op against the
    no-root baseline (the ablation), and with `root_gate == 1` it reproduces the shipped
    arithmetic exactly (proved in verify_ishtiqaq_fixed.py).

What the shipped path computes (ishtiqaq_attention_v12.py):
  l.177  score_root = (W_qr E(r_i)) . (W_kr E(r_j)) * scale          <- a function of (r_i, r_j)
  l.180  ishtiqaq_cond[i,j] = [r_i != 0] * [r_j != 0]
  l.184  identical_root_bonus[i,j] = [r_i == r_j != 0] * ishtiqaq_gamma
  l.186  total_score += stream_mix[1] * ishtiqaq_cond * score_root + identical_root_bonus
  l.270  context = softmax(total_score) @ v_proj(h)                  <- VALUE STAYS SURFACE
so the root term is a pure additive BIAS on the surface attention scores.  It never touches the
residual stream and it can only re-weight the surface value vectors.  A `gate` (zero-initialised)
makes that whole contribution ablatable; because the inner `stream_mix[1]` and `ishtiqaq_gamma`
are already nonzero (0.25), d(loss)/d(gate) != 0 at step 0 and the gate escapes zero -- zeroing
BOTH the gate and the branch weights is the dead saddle the v18fix note warns about.

`root_source`:
  'native' -- keep the shipped per-layer `root_embed` (9015, 64) + `root_q_proj` (64->896) +
              `root_k_proj` (64->128).  NOTE: 9015 rows against a 9490-id root stream, and the
              table is at N(0,1) default init (never trained) because the path was never wired.
  'shared' -- bind the RELEASED, synthesis-trained `morphemic_embed.root_embed` (9490, 448) --
              the same table the residual cross-attention uses -- and widen the two projections
              to 448 input features.  This makes the ONLY difference from the RCA the INSERTION
              POINT (score bias vs residual), which is the hypothesis under test.
"""
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from models.ishtiqaq_attention_v12 import (IshtiqaqAttentionV12,
                                           apply_rotary_pos_emb, Qwen2RMSNorm)


class IshtiqaqRootBiasAttention(IshtiqaqAttentionV12):
    """IshtiqaqAttentionV12 + `root_gate`/`pillar_gate` and an optional 448-dim root source."""

    def __init__(self, *a, root_source: str = 'native', shared_root_embed=None,
                 gate_init: float = 0.0, pillar_gate_init: float = 0.0, **kw):
        super().__init__(*a, **kw)
        d = self.root_embed.weight.dtype
        self.root_source = str(root_source)
        # zero-init gates => the whole native root path is exactly 0 at step 0
        self.root_gate = nn.Parameter(torch.tensor(float(gate_init), dtype=d))
        self.pillar_gate = nn.Parameter(torch.tensor(float(pillar_gate_init), dtype=d))

        if self.root_source == 'shared':
            if shared_root_embed is None:
                raise ValueError("root_source='shared' needs shared_root_embed")
            object.__setattr__(self, '_shared_root_embed', shared_root_embed)  # NOT registered
            d_root = int(shared_root_embed.weight.shape[1])
            self.root_q_proj = nn.Linear(d_root, self.num_heads * self.head_dim, bias=False,
                                         dtype=d)
            self.root_k_proj = nn.Linear(d_root, self.num_kv_heads * self.head_dim, bias=False,
                                         dtype=d)
            with torch.no_grad():
                self.root_q_proj.weight.normal_(0.0, 0.02)
                self.root_k_proj.weight.normal_(0.0, 0.02)
        elif self.root_source != 'native':
            raise ValueError(f'unknown root_source {self.root_source!r}')

    # ------------------------------------------------------------------ helpers
    def _root_table(self) -> nn.Embedding:
        if self.root_source == 'shared':
            return self._shared_root_embed
        return self.root_embed

    def root_path_parameters(self) -> List[nn.Parameter]:
        """Exactly the parameters of the NATIVE ROOT PATH (not the rest of the attention)."""
        ps = [self.root_gate, self.pillar_gate, self.root_q_proj.weight, self.root_k_proj.weight]
        if self.root_source == 'native':
            ps.append(self.root_embed.weight)
        return ps

    # ------------------------------------------------------------------ forward
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, past_key_value=None, use_cache=False,
                root_ids=None, wazn_ids=None, rotary_emb=None, **kwargs):
        """Byte-for-byte the shipped forward, with `root_gate` on the two root score terms and
        `pillar_gate` on the Pillar-I/II penalties."""
        batch_size, seq_len, _ = hidden_states.shape

        q_s = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_s = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q_s = self.q_norm(q_s).transpose(1, 2)
        k_s = self.k_norm(k_s).transpose(1, 2)
        v = v.transpose(1, 2)

        pos_emb = position_embeddings if position_embeddings is not None else rotary_emb
        if pos_emb is not None:
            cos, sin = pos_emb
            q_s, k_s = apply_rotary_pos_emb(q_s, k_s, cos, sin)

        pkv = past_key_values if past_key_values is not None else past_key_value
        if pkv is not None:
            if hasattr(pkv, 'update'):
                k_s, v = pkv.update(k_s, v, self.layer_idx)
            elif isinstance(pkv, (tuple, list)):
                k_s = torch.cat([pkv[0], k_s], dim=2)
                v = torch.cat([pkv[1], v], dim=2)

        total_kv_len = k_s.shape[2]

        if self.num_kv_groups > 1:
            k_s_rep = k_s.repeat_interleave(self.num_kv_groups, dim=1)
            v_rep = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_s_rep = k_s
            v_rep = v

        score_surface = torch.matmul(q_s, k_s_rep.transpose(-2, -1)) * self.scale

        if root_ids is None and self.active_root_ids is not None:
            root_ids = self.active_root_ids

        total_score = self.stream_mix[0] * score_surface

        if root_ids is not None:
            table = self._root_table()
            safe_root_ids = torch.clamp(root_ids, 0, table.num_embeddings - 1)
            if seq_len == safe_root_ids.shape[1]:
                r_emb = table(safe_root_ids).to(self.root_q_proj.weight.dtype)
                q_r = self.root_q_proj(r_emb).view(
                    batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k_r = self.root_k_proj(r_emb).view(
                    batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                if self.num_kv_groups > 1:
                    k_r = k_r.repeat_interleave(self.num_kv_groups, dim=1)

                score_root = torch.matmul(q_r, k_r.transpose(-2, -1)) * self.scale

                has_root = (safe_root_ids != 0).to(score_surface.dtype)
                ishtiqaq_cond = (has_root.unsqueeze(-1) * has_root.unsqueeze(-2)).unsqueeze(1)

                r_i = safe_root_ids.unsqueeze(-1)
                r_j = safe_root_ids.unsqueeze(-2)
                identical_root_bonus = ((r_i == r_j) & (r_i != 0)).to(
                    score_surface.dtype).unsqueeze(1) * self.ishtiqaq_gamma

                # THE ONLY STRUCTURAL CHANGE: both terms pass through the zero-init gate.
                total_score = (total_score
                               + self.root_gate * (self.stream_mix[1] * ishtiqaq_cond * score_root)
                               + self.root_gate * identical_root_bonus)

        if seq_len > 1 and total_kv_len == seq_len:
            causal_mask = torch.triu(
                torch.full((seq_len, total_kv_len), float('-inf'), device=hidden_states.device),
                diagonal=1)
            total_score = total_score + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = (1.0 - attention_mask[:, None, None, :].to(total_score.dtype)) * -10000.0
                total_score = total_score + mask
            elif attention_mask.dim() == 4:
                total_score = total_score + attention_mask

        cov_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        if root_ids is not None:
            table = self._root_table()
            safe_root_ids = torch.clamp(root_ids, 0, table.num_embeddings - 1)
            if safe_root_ids.shape[1] == total_kv_len:
                is_root = (safe_root_ids != 0).to(total_score.dtype)
                is_generated = (safe_root_ids == 0).to(total_score.dtype)

                if seq_len > 1 and total_kv_len == seq_len:
                    if self.coverage_weight.abs() > 1e-4 or self.training:
                        with torch.no_grad():
                            base_attn = torch.softmax(total_score, dim=-1)
                            mean_attn = base_attn.mean(dim=1)
                            cum_attn = torch.cumsum(mean_attn, dim=-2)
                            shifted_cum = torch.cat(
                                [torch.zeros_like(cum_attn[:, :1, :]), cum_attn[:, :-1, :]], dim=1)

                        num_roots = is_root.sum(dim=-1, keepdim=True).clamp(min=1.0)
                        norm_cum = shifted_cum / num_roots.unsqueeze(1)
                        bounded_penalty = torch.sigmoid(
                            (norm_cum - self.coverage_threshold) / self.coverage_tau) * is_root.unsqueeze(1)
                        total_score = total_score - self.pillar_gate * (
                            self.coverage_weight * bounded_penalty.unsqueeze(1))

                        if self.training:
                            cur_attn = torch.softmax(total_score, dim=-1).mean(dim=1)
                            cov_loss = torch.min(cur_attn, shifted_cum).sum(dim=-1).mean()

                    if self.governance_strength.abs() > 1e-4:
                        positions = torch.arange(seq_len, device=hidden_states.device)
                        query_pos = positions.unsqueeze(-1)
                        distance = (query_pos - positions.unsqueeze(0)).clamp(min=0)
                        decay = torch.exp(-self.governance_decay * distance).clamp(min=0.10)
                        damping = (1.0 - decay) * is_generated.unsqueeze(1)
                        total_score = total_score - self.pillar_gate * (
                            self.governance_strength * damping.unsqueeze(1))

                elif seq_len == 1:
                    if self.coverage_cache is None or self.coverage_cache.shape[-1] != total_kv_len:
                        self.coverage_cache = torch.zeros(
                            (batch_size, total_kv_len), device=hidden_states.device,
                            dtype=total_score.dtype)

                    num_roots = is_root.sum(dim=-1, keepdim=True).clamp(min=1.0)
                    norm_cache = self.coverage_cache / num_roots
                    step_penalty = torch.sigmoid(
                        (norm_cache - self.coverage_threshold) / self.coverage_tau) * is_root
                    total_score = total_score - self.pillar_gate * (
                        self.coverage_weight * step_penalty.unsqueeze(1).unsqueeze(2))

                    if self.governance_strength.abs() > 1e-4:
                        curr_pos = total_kv_len - 1
                        positions = torch.arange(total_kv_len, device=hidden_states.device)
                        distance = (curr_pos - positions).clamp(min=0)
                        decay = torch.exp(-self.governance_decay * distance).clamp(min=0.10)
                        step_damping = (1.0 - decay) * is_generated
                        total_score = total_score - self.pillar_gate * (
                            self.governance_strength * step_damping.unsqueeze(1).unsqueeze(2))

        self.last_cov_loss = cov_loss

        attn_weights = torch.softmax(total_score, dim=-1)

        if seq_len == 1 and self.coverage_cache is not None:
            new_attn_step = attn_weights.mean(dim=1).squeeze(1).detach()
            self.coverage_cache = self.coverage_cache + new_attn_step

        attn_probs = self.dropout(attn_weights).to(v_rep.dtype)

        context = torch.matmul(attn_probs, v_rep)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(context.to(self.o_proj.weight.dtype))

        return output, attn_weights


def parse_layer_spec(spec: str, n_layers: int) -> List[int]:
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


def swap_in_root_bias(layers: nn.ModuleList, layer_indices: Sequence[int],
                      shared_root_embed: Optional[nn.Embedding] = None,
                      root_source: str = 'shared', gate_init: float = 0.0,
                      pillar_gate_init: float = 0.0):
    """Replace `layers[i].self_attn` with `IshtiqaqRootBiasAttention`, keeping weights.

    Returns (new_modules, trainable_native_params).  The shipped `IshtiqaqAttentionV12` code
    object is not modified anywhere.
    """
    mods, params = [], []
    for i in layer_indices:
        old = layers[i].self_attn
        if not isinstance(old, IshtiqaqAttentionV12):
            raise TypeError(f'layer {i}: self_attn is {type(old).__name__}, not '
                            f'IshtiqaqAttentionV12')
        new = IshtiqaqRootBiasAttention(
            hidden_size=old.hidden_size, num_heads=old.num_heads,
            num_kv_heads=old.num_kv_heads, head_dim=old.head_dim,
            num_roots=old.root_embed.num_embeddings, num_awzan=old.wazn_embed.num_embeddings,
            ishtiqaq_init_strength=float(old.ishtiqaq_gamma.detach()),
            dropout=float(old.dropout.p), layer_idx=old.layer_idx,
            dtype=old.root_embed.weight.dtype,
            root_source=root_source, shared_root_embed=shared_root_embed,
            gate_init=gate_init, pillar_gate_init=pillar_gate_init)
        # The module was created on the DEFAULT device; `swap_in_root_bias` runs AFTER
        # `model.to(device)`, so without this the new root_q_proj/root_k_proj/gates stay on CPU
        # while the trunk is on cuda -> "mat1 is on cuda:0, different from other tensors on cpu"
        # at the first forward.  (The CPU equivalence proof cannot catch this.)
        dev = old.root_embed.weight.device
        new = new.to(device=dev)
        for _p in new.parameters():
            if _p.device != dev:
                raise RuntimeError(f'layer {i}: {_p.shape} stayed on {_p.device} != {dev}')
        sd = old.state_dict()
        # the shipped tensors that survive the swap; `root_embed.weight` is kept only for
        # 'native' (for 'shared' the table is unregistered and this one is deliberately unused).
        keep = {k: v for k, v in sd.items()
                if not (root_source == 'shared' and k.startswith('root_q_proj'))
                and not (root_source == 'shared' and k.startswith('root_k_proj'))}
        missing, unexpected = new.load_state_dict(keep, strict=False)
        real_missing = [k for k in missing if not k.startswith(('root_q_proj', 'root_k_proj',
                                                               'root_gate', 'pillar_gate'))]
        if real_missing:
            raise RuntimeError(f'layer {i}: missing {real_missing}')
        if root_source == 'shared':
            # the shipped 9015x64 table is now unreachable; keep it out of the optimizer
            new.root_embed.weight.requires_grad_(False)
        new.train(old.training)          # preserve the module's train/eval flag
        layers[i].self_attn = new
        mods.append(new)
        params.extend(new.root_path_parameters())
    return mods, params


@torch.no_grad()
def gate_values(mods: Sequence[IshtiqaqRootBiasAttention]):
    return [(float(m.root_gate.detach()), float(m.pillar_gate.detach())) for m in mods]


@torch.no_grad()
def zero_gates(mods: Sequence[IshtiqaqRootBiasAttention]):
    """Force every native root gate to 0 for one pass (inference-time ablation)."""
    old = [(float(m.root_gate.detach()), float(m.pillar_gate.detach())) for m in mods]
    for m in mods:
        m.root_gate.data.zero_()
        m.pillar_gate.data.zero_()
    return old


@torch.no_grad()
def restore_gates(mods: Sequence[IshtiqaqRootBiasAttention], old):
    for m, (g, p) in zip(mods, old):
        m.root_gate.data.fill_(float(g))
        m.pillar_gate.data.fill_(float(p))


@torch.no_grad()
def set_enabled(mods: Sequence[IshtiqaqRootBiasAttention], enabled: bool):
    """`False` => the native path is off (root_gate 0); `True` => on (root_gate 1)."""
    for m in mods:
        m.root_gate.data.fill_(1.0 if enabled else 0.0)
