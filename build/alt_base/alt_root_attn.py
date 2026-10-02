#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alt_root_attn.py -- IN-ARCHITECTURE root attention for an arbitrary base trunk, in two parts:

  1. `ScoreBiasModule`  -- the NATIVE Farahidian idea, made trainable and testable:
        total_score += gate * root_score(r_i, r_j) + gamma * 1[r_i == r_j]
     added to the ATTENTION SCORE before softmax.  `root_score` is a function of the two root
     ids and their (learnable) embeddings ONLY -- never of h -- so this REWEIGHTS the surface
     value stream.  In the released trunk these paths exist but were never trained (the root
     branch sits at its init and the ids that reach it are Qwen subword token ids clamped to
     [0, 9014], i.e. noise), so the idea has never actually been measured.  Both `gate` and the
     embeddings are ours, and `gate` is zero-initialised => bitwise no-op at init.

  2. `AltRootCrossAttentionStack` -- h <- h + gate * LayerNorm(o_proj(A @ (V_s + Wv E(r))))
     where A is the SAME attention distribution the surface path uses, perturbed by the score
     bias above, and the value stream is AUGMENTED with a root-derived term.  That is the real
     INJECTION: the score bias alone can only reweight v_s.  Hooks one module per selected layer,
     rewrites that layer's attention, and returns the same (output, attn_weights) tuple.

Both mechanisms live in ONE forward so they compose: `--score-bias` alone, `--rca-residual`
alone, or both.  With both gates at 0 the layer output is bitwise identical to the unpatched
base (verified in `verify_bitwise`).
"""
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreBiasModule(nn.Module):
    """total_score + gate * q_r.k_r^T * scale + gamma * 1[r_i == r_j], on a [0, num_roots) space."""

    def __init__(self, num_roots: int, head_dim: int, num_heads: int, num_kv_heads: int,
                 scale: float, dtype=torch.float32, emb_std: float = 0.02):
        super().__init__()
        self.num_roots, self.head_dim = int(num_roots), int(head_dim)
        self.num_heads, self.num_kv_heads, self.scale = int(num_heads), int(num_kv_heads), scale
        self.root_embed = nn.Embedding(num_roots, head_dim, dtype=dtype)
        self.q_proj = nn.Linear(head_dim, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(head_dim, num_kv_heads * head_dim, bias=False, dtype=dtype)
        with torch.no_grad():
            self.root_embed.weight.normal_(0.0, emb_std)
            self.q_proj.weight.normal_(0.0, 0.02)
            self.k_proj.weight.normal_(0.0, 0.02)
        # zero-init gates: the whole mechanism is exactly off at step 0
        self.gate = nn.Parameter(torch.zeros((), dtype=dtype))
        self.gamma = nn.Parameter(torch.zeros((), dtype=dtype))

    def score(self, root_ids: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """[B,T] ids -> [B,1,T,T] additive score term (bias includes its own gate/scale)."""
        H, D = self.num_heads, self.head_dim
        Kv = self.num_kv_heads
        e = self.root_embed(root_ids.long()).to(self.q_proj.weight.dtype)
        q = self.q_proj(e).view(B, T, H, D).transpose(1, 2)                 # [B,H,T,D]
        k = self.k_proj(e).view(B, T, Kv, D).transpose(1, 2)               # [B,Kv,T,D]
        if H != Kv:
            k = k.repeat_interleave(H // Kv, dim=1)
        sc = torch.matmul(q, k.transpose(-1, -2)) * self.scale             # [B,H,T,T]
        r_i = root_ids.unsqueeze(-1)
        r_j = root_ids.unsqueeze(-2)
        has = (root_ids != 0)
        cond = (has.unsqueeze(-1) & has.unsqueeze(-2)).unsqueeze(1).to(sc.dtype)
        bonus = ((r_i == r_j) & (r_i != 0)).to(sc.dtype).unsqueeze(1) * self.gamma.float()
        return (self.gate.float() * cond * sc).mean(dim=1, keepdim=True) + bonus


class AltRootCrossAttentionStack(nn.Module):
    """Rewrites selected trunk self-attention layers with an optional score bias + root residual."""

    def __init__(self, d_model: int, d_root: int, root_weight: torch.Tensor,
                 layer_indices: Sequence[int], score_bias: Optional[ScoreBiasModule] = None,
                 residual: bool = True, num_heads: int = 8, head_dim: Optional[int] = None,
                 dropout: float = 0.0, out_std: float = 1e-3, out_norm: bool = True,
                 dtype=torch.float32):
        super().__init__()
        self.layer_indices = [int(i) for i in layer_indices]
        self.residual_enabled = bool(residual)
        self.score_bias = score_bias
        self.d_model, self.d_root = int(d_model), int(d_root)
        self.root_weight = root_weight                # shared frozen (9490,448) table, NOT re-registered
        self.num_heads = int(num_heads)
        if residual:
            self.head_dim = int(head_dim or (d_model // num_heads))
            if self.num_heads * self.head_dim != d_model:
                raise ValueError('num_heads*head_dim must equal d_model')
            d_attn = self.num_heads * self.head_dim
            self.q_proj = nn.Linear(d_model, d_attn, bias=False, dtype=dtype)
            self.k_proj = nn.Linear(d_root, d_attn, bias=False, dtype=dtype)
            self.v_proj = nn.Linear(d_root, d_attn, bias=False, dtype=dtype)
            self.o_proj = nn.Linear(d_attn, d_model, bias=False, dtype=dtype)
            self.q_norm = nn.LayerNorm(self.head_dim, dtype=dtype)
            self.k_norm = nn.LayerNorm(self.head_dim, dtype=dtype)
            self.out_norm = nn.LayerNorm(d_model, dtype=dtype) if out_norm else None
            self.drop = nn.Dropout(dropout)
            self.gate = nn.Parameter(torch.zeros((), dtype=dtype))
            with torch.no_grad():
                for lin in (self.q_proj, self.k_proj, self.v_proj):
                    lin.weight.normal_(0.0, 0.02)
                self.o_proj.weight.normal_(0.0, out_std)
        self._root_ids: Optional[torch.Tensor] = None
        self._attn_mask: Optional[torch.Tensor] = None
        self._position_embeddings = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self.counts = {'calls': 0, 'with_root_ids': 0, 'score_bias': 0, 'residual': 0}
        self.last_checks = {}

    def _pre_hook(self):
        def pre(module, args, kwargs):
            pe = kwargs.get('position_embeddings', None)
            if pe is None:
                pe = kwargs.get('rotary_emb', None)
            self._position_embeddings = pe
        return pre

    # ------------------------------------------------------------------ wiring
    def attach(self, layers: Sequence[nn.Module]) -> None:
        if self._handles:
            raise RuntimeError('attach called twice')
        for idx in self.layer_indices:
            lay = layers[idx]
            self._handles.append(lay.register_forward_hook(self._hook(idx)))
            # capture the exact positional/mask tensors the trunk passes to attention
            self._handles.append(lay.self_attn.register_forward_pre_hook(self._pre_hook(), with_kwargs=True))
        n = sum(p.numel() for p in self.parameters()) / 1e6
        print(f'[*] alt root attention on layers {self.layer_indices} ({n:.2f}M new params; '
              f'score_bias={self.score_bias is not None} residual={self.residual_enabled})',
              flush=True)

    @property
    def d_attn(self):
        return self.num_heads * self.head_dim if self.residual_enabled else 0

    def set_root_ids(self, r: Optional[torch.Tensor]) -> None:
        self._root_ids = r

    def _hook(self, layer_idx: int):
        def hook(module, args, output):
            self.counts['calls'] += 1
            hs = output[0] if isinstance(output, tuple) else output
            rest = output[1:] if isinstance(output, tuple) else ()
            r = self._root_ids
            if r is None or hs is None or r.shape[:2] != hs.shape[:2]:
                return output
            self.counts['with_root_ids'] += 1
            new_hs = self._layer_forward(module, hs, r)
            return (new_hs,) + rest
        return hook

    def _layer_forward(self, layer: nn.Module, hs: torch.Tensor, root_ids: torch.Tensor):
        att = layer.self_attn
        B, T, _ = hs.shape
        cfg = getattr(att, 'config', None)
        dev = hs.device
        # ---- rebuild the layer's input norm + attention with our terms -------------------
        drop = getattr(att, 'attention_dropout', 0.0)
        with torch.no_grad():
            inp = layer.input_layernorm(hs)
        q_s = att.q_proj(inp).view(B, T, att.num_heads, att.head_dim)
        k_s = att.k_proj(inp).view(B, T, att.num_kv_heads, att.head_dim)
        v_s = att.v_proj(inp).view(B, T, att.num_kv_heads, att.head_dim)
        q_s = att.q_norm(q_s).transpose(1, 2)
        k_s = att.k_norm(k_s).transpose(1, 2)
        v_s = v_s.transpose(1, 2)
        # rotary: the trunk passes position_embeddings=[cos,sin] for the FULL sequence length
        pos_emb = self._position_embeddings
        if pos_emb is not None:
            cos, sin = pos_emb
            if cos.shape[-2] > T:
                cos, sin = cos[..., :T, :], sin[..., :T, :]
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
            q_s, k_s = apply_rotary_pos_emb(q_s, k_s, cos.to(q_s.dtype), sin.to(k_s.dtype))
        else:
            pos = torch.arange(T, device=dev).unsqueeze(0)
            if getattr(att, 'rotary_emb', None) is not None:
                cos, sin = att.rotary_emb(v_s, pos)
                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
                q_s, k_s = apply_rotary_pos_emb(q_s, k_s, cos, sin)
        kv = att.num_kv_groups
        k_rep = k_s.repeat_interleave(kv, dim=1) if kv > 1 else k_s
        v_rep = v_s.repeat_interleave(kv, dim=1) if kv > 1 else v_s
        score = torch.matmul(q_s, k_rep.transpose(-2, -1)) * att.scale
        if self.score_bias is not None:
            score = score + self.score_bias.score(root_ids, B, T)
            self.counts['score_bias'] += 1
        # causal (+ padding, if the caller passed a mask)
        cm = torch.triu(torch.full((T, T), float('-inf'), device=dev), diagonal=1)
        score = score + cm.unsqueeze(0).unsqueeze(0)
        am = self._attn_mask
        if am is not None and am.shape[-1] == T:
            if am.dim() == 2:
                score = score + (1.0 - am[:, None, None, :].float()) * -10000.0
            elif am.dim() == 4:
                score = score + am
        # EXACT dtype path of IshtiqaqAttentionV12: softmax happens in the score's own dtype
        # (the trunk's bf16), and the context matmul casts to v_rep's dtype.  Keeping this
        # identical is what makes the zero-gate case bitwise equal to the unpatched layer.
        score = score.to(v_s.dtype)
        A = F.softmax(score, dim=-1)
        A = att.dropout(A).to(v_s.dtype) if hasattr(att, 'dropout') else A
        # ---- residual: h + gate * LN(o_proj(A @ (V_s + Wv E(r)))) ------------------------
        if self.residual_enabled:
            Rw = self.root_weight
            safe = root_ids.clamp(0, Rw.shape[0] - 1)
            e = Rw.index_select(0, safe.reshape(-1)).view(B, T, -1).float()
            v_r = self.v_proj(e).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            if kv > 1 and self.num_heads != kv:
                v_r = v_r.repeat_interleave(self.num_heads // kv, dim=1)
            v_aug = (v_rep.float() + v_r).to(v_s.dtype)
            ctx = torch.matmul(A, v_aug).transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
            out = self.o_proj(ctx)
            if self.out_norm is not None:
                out = self.out_norm(out)
            delta = self.gate.float() * self.drop(out)
            self.counts['residual'] += 1
            return (hs.float() + delta).to(hs.dtype)
        # score bias only: the value stream and o_proj stay the trunk's own; replace the layer's
        # attention output inside its residual stream and keep the trunk MLP.
        ctx = torch.matmul(A.to(v_rep.dtype), v_rep)
        ctx = ctx.transpose(1, 2).reshape(B, T, att.num_heads * att.head_dim)
        attn_out = att.o_proj(ctx.to(att.o_proj.weight.dtype))
        h = hs.float() + attn_out.float()
        before_mlp = layer.post_attention_layernorm(h.to(hs.dtype))
        mlp = layer.mlp(before_mlp)
        return (h + mlp.float()).to(hs.dtype)

    @torch.no_grad()
    def gate_values(self):
        g = []
        if self.residual_enabled:
            g.append(float(self.gate.detach()))
        if self.score_bias is not None:
            g += [float(self.score_bias.gate.detach()), float(self.score_bias.gamma.detach())]
        return g

    @torch.no_grad()
    def zero_gates(self):
        old = []
        if self.residual_enabled:
            old.append(float(self.gate.detach())); self.gate.data.zero_()
        if self.score_bias is not None:
            old.append(float(self.score_bias.gate.detach())); self.score_bias.gate.data.zero_()
            old.append(float(self.score_bias.gamma.detach())); self.score_bias.gamma.data.zero_()
        return old

    @torch.no_grad()
    def restore_gates(self, old):
        i = 0
        if self.residual_enabled:
            self.gate.data.fill_(float(old[i])); i += 1
        if self.score_bias is not None:
            self.score_bias.gate.data.fill_(float(old[i])); i += 1
            self.score_bias.gamma.data.fill_(float(old[i])); i += 1


def disable_native_root_path(model) -> int:
    """Make the trunk a PURE base model: no native root score bias, no coverage/governance.

    `UnifiedRootformerV12.extract_morphemic_ids` maps SUBWORD token ids through
    `id_to_root_table` (clamped to [0, 9014]); with a 151,643-entry base vocabulary those ids are
    meaningless, so the released IshtiqaqAttentionV12 root branch would inject noise.  Returning
    None removes the native root term entirely.  Returns the number of modules patched.
    """
    n = 0
    layers = model.backbone.model.layers
    for layer in layers:
        att = layer.self_attn
        att.active_root_ids = None
        att.active_wazn_ids = None
        n += 1
    return n
