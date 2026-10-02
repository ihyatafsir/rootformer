#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ishtiqaq_attention_v12.py
Rootformer v12: Kitāb al-ʿAyn Sovereign Ishtiqāq Dual-Stream Attention (Qwen2.5-0.5B Backbone).
Audited and enhanced per AynEngine AI Coding Edition 5-Pillar Epistemic Specifications:
1. Bounded Farāhīdian Root-Coverage Attention (Sigmoid Saturation Gate).
2. Sībawayhian Governance Decay (Dissolves decoder-only self-attention sink on generated Latin tokens).
3. Activated stream_mix dual projections.
Dimensions: hidden_size=896, num_heads=14, num_kv_heads=2, head_dim=64.
"""

import math
from typing import Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class IshtiqaqAttentionV12(nn.Module):
    def __init__(
        self,
        hidden_size: int = 896,
        num_heads: int = 14,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        num_roots: int = 9015,
        num_awzan: int = 128,
        ishtiqaq_init_strength: float = 0.25,
        dropout: float = 0.0,
        layer_idx: int = 0,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 1. Surface Projections from Hidden States
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True, dtype=dtype)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False, dtype=dtype)

        # Q/K Normalization
        self.q_norm = Qwen2RMSNorm(head_dim)
        self.k_norm = Qwen2RMSNorm(head_dim)

        # 2. Farāhīdian Radical & Wazn Coordinate Embeddings
        self.root_embed = nn.Embedding(num_roots, head_dim, dtype=dtype)
        self.wazn_embed = nn.Embedding(num_awzan, head_dim, dtype=dtype)

        with torch.no_grad():
            self.root_embed.weight[0].normal_(mean=0.0, std=0.02)
            self.wazn_embed.weight[0].normal_(mean=0.0, std=0.02)

        # 3. Morphemic Projections (Root Stream)
        self.root_q_proj = nn.Linear(head_dim, num_heads * head_dim, bias=False, dtype=dtype)
        self.root_k_proj = nn.Linear(head_dim, num_kv_heads * head_dim, bias=False, dtype=dtype)

        # 4. Learned Ishtiqāq Resonance Scalars
        self.ishtiqaq_gamma = nn.Parameter(torch.tensor(ishtiqaq_init_strength, dtype=dtype))
        self.stream_mix = nn.Parameter(torch.tensor([0.75, 0.25], dtype=dtype))

        # 5. AynEngine 5-Pillar Mechanisms:
        # Bounded Farāhīdian Root-Coverage (Pillar I)
        self.coverage_weight = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.coverage_threshold = nn.Parameter(torch.tensor(0.08, dtype=dtype))
        self.coverage_tau = 0.5
        
        # Sībawayhian Governance Decay (Pillar II)
        self.governance_strength = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.governance_decay = 0.08

        self.coverage_cache = None
        self.last_cov_loss = None

        self.dropout = nn.Dropout(dropout)
        self.active_root_ids = None
        self.active_wazn_ids = None

    def reset_coverage(self):
        """Reset the autoregressive coverage accumulation cache."""
        self.coverage_cache = None
        self.last_cov_loss = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        root_ids: Optional[torch.Tensor] = None,
        wazn_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        batch_size, seq_len, _ = hidden_states.shape

        # A. Surface Projections from Hidden States
        q_s = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_s = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q_s = self.q_norm(q_s).transpose(1, 2)  # [B, H, T, D]
        k_s = self.k_norm(k_s).transpose(1, 2)  # [B, H_kv, T, D]
        v = v.transpose(1, 2)                  # [B, H_kv, T, D]

        # B. Rotary Position Embeddings
        pos_emb = position_embeddings if position_embeddings is not None else rotary_emb
        if pos_emb is not None:
            cos, sin = pos_emb
            q_s, k_s = apply_rotary_pos_emb(q_s, k_s, cos, sin)

        # C. KV Caching
        pkv = past_key_values if past_key_values is not None else past_key_value
        if pkv is not None:
            if hasattr(pkv, 'update'):
                k_s, v = pkv.update(k_s, v, self.layer_idx)
            elif isinstance(pkv, (tuple, list)):
                k_s = torch.cat([pkv[0], k_s], dim=2)
                v = torch.cat([pkv[1], v], dim=2)

        total_kv_len = k_s.shape[2]

        # D. GQA Expansion
        if self.num_kv_groups > 1:
            k_s_rep = k_s.repeat_interleave(self.num_kv_groups, dim=1)
            v_rep = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_s_rep = k_s
            v_rep = v

        # E. Surface Attention Energy
        score_surface = torch.matmul(q_s, k_s_rep.transpose(-2, -1)) * self.scale

        # F. Farāhīdian Root Stream
        if root_ids is None and self.active_root_ids is not None:
            root_ids = self.active_root_ids

        total_score = self.stream_mix[0] * score_surface

        if root_ids is not None:
            safe_root_ids = torch.clamp(root_ids, 0, self.root_embed.num_embeddings - 1)
            if seq_len == safe_root_ids.shape[1]:
                r_emb = self.root_embed(safe_root_ids)
                q_r = self.root_q_proj(r_emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k_r = self.root_k_proj(r_emb).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                if self.num_kv_groups > 1:
                    k_r = k_r.repeat_interleave(self.num_kv_groups, dim=1)

                score_root = torch.matmul(q_r, k_r.transpose(-2, -1)) * self.scale

                has_root = (safe_root_ids != 0).to(score_surface.dtype)
                ishtiqaq_cond = (has_root.unsqueeze(-1) * has_root.unsqueeze(-2)).unsqueeze(1)

                r_i = safe_root_ids.unsqueeze(-1)
                r_j = safe_root_ids.unsqueeze(-2)
                identical_root_bonus = ((r_i == r_j) & (r_i != 0)).to(score_surface.dtype).unsqueeze(1) * self.ishtiqaq_gamma

                total_score = total_score + (self.stream_mix[1] * ishtiqaq_cond * score_root) + identical_root_bonus

        # G. Causal Mask & Padding
        if seq_len > 1 and total_kv_len == seq_len:
            causal_mask = torch.triu(
                torch.full((seq_len, total_kv_len), float('-inf'), device=hidden_states.device),
                diagonal=1
            )
            total_score = total_score + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = (1.0 - attention_mask[:, None, None, :].to(total_score.dtype)) * -10000.0
                total_score = total_score + mask
            elif attention_mask.dim() == 4:
                total_score = total_score + attention_mask

        # H. AynEngine 5-Pillar Two-Sided Mechanism:
        # Side 1: Bounded Sigmoid Root-Coverage (Pillar I)
        # Side 2: Sībawayhian Governance Decay on Generated Non-Root Tokens (Pillar II)
        cov_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        if root_ids is not None:
            safe_root_ids = torch.clamp(root_ids, 0, self.root_embed.num_embeddings - 1)
            if safe_root_ids.shape[1] == total_kv_len:
                is_root = (safe_root_ids != 0).to(total_score.dtype)
                is_generated = (safe_root_ids == 0).to(total_score.dtype)

                # --- Parallel Training / Sequence Mode (seq_len > 1) ---
                if seq_len > 1 and total_kv_len == seq_len:
                    # 1. Bounded Root Coverage
                    if self.coverage_weight.abs() > 1e-4 or self.training:
                        with torch.no_grad():
                            base_attn = F.softmax(total_score, dim=-1)
                            mean_attn = base_attn.mean(dim=1)  # [B, T, T]
                            cum_attn = torch.cumsum(mean_attn, dim=-2)
                            shifted_cum = torch.cat([torch.zeros_like(cum_attn[:, :1, :]), cum_attn[:, :-1, :]], dim=1)

                        num_roots = is_root.sum(dim=-1, keepdim=True).clamp(min=1.0)
                        norm_cum = shifted_cum / num_roots.unsqueeze(1)
                        bounded_penalty = torch.sigmoid((norm_cum - self.coverage_threshold) / self.coverage_tau) * is_root.unsqueeze(1)
                        total_score = total_score - (self.coverage_weight * bounded_penalty.unsqueeze(1))

                        if self.training:
                            cur_attn = F.softmax(total_score, dim=-1).mean(dim=1)
                            cov_loss = torch.min(cur_attn, shifted_cum).sum(dim=-1).mean()

                    # 2. Sībawayhian Governance Decay (Dissolves attention sink on generated non-root tokens)
                    if self.governance_strength.abs() > 1e-4:
                        positions = torch.arange(seq_len, device=hidden_states.device)
                        query_pos = positions.unsqueeze(-1)
                        distance = (query_pos - positions.unsqueeze(0)).clamp(min=0)
                        decay = torch.exp(-self.governance_decay * distance).clamp(min=0.10)
                        damping = (1.0 - decay) * is_generated.unsqueeze(1)
                        total_score = total_score - (self.governance_strength * damping.unsqueeze(1))

                # --- Autoregressive Step-by-Step Mode (seq_len == 1) ---
                elif seq_len == 1:
                    if self.coverage_cache is None or self.coverage_cache.shape[-1] != total_kv_len:
                        self.coverage_cache = torch.zeros((batch_size, total_kv_len), device=hidden_states.device, dtype=total_score.dtype)

                    num_roots = is_root.sum(dim=-1, keepdim=True).clamp(min=1.0)
                    norm_cache = self.coverage_cache / num_roots
                    step_penalty = torch.sigmoid((norm_cache - self.coverage_threshold) / self.coverage_tau) * is_root
                    total_score = total_score - (self.coverage_weight * step_penalty.unsqueeze(1).unsqueeze(2))

                    if self.governance_strength.abs() > 1e-4:
                        curr_pos = total_kv_len - 1
                        positions = torch.arange(total_kv_len, device=hidden_states.device)
                        distance = (curr_pos - positions).clamp(min=0)
                        decay = torch.exp(-self.governance_decay * distance).clamp(min=0.10)
                        step_damping = (1.0 - decay) * is_generated
                        total_score = total_score - (self.governance_strength * step_damping.unsqueeze(1).unsqueeze(2))

        self.last_cov_loss = cov_loss

        # I. Softmax & Context Projection
        attn_weights = F.softmax(total_score, dim=-1)

        if seq_len == 1 and self.coverage_cache is not None:
            new_attn_step = attn_weights.mean(dim=1).squeeze(1).detach()
            self.coverage_cache = self.coverage_cache + new_attn_step

        attn_probs = self.dropout(attn_weights).to(v_rep.dtype)

        context = torch.matmul(attn_probs, v_rep)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(context.to(self.o_proj.weight.dtype))

        return output, attn_weights
