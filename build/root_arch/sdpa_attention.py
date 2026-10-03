#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sdpa_attention.py -- replace the HAND-MATERIALISED `[B, H, T, T]` attention of
`IshtiqaqAttentionV12` with `F.scaled_dot_product_attention`, for the early-root experiment.

WHAT WAS WRONG (Phase 0a)
-------------------------
The shipped forward builds the score matrix by hand:

    score_surface = torch.matmul(q_s, k_s_rep.transpose(-2, -1)) * self.scale   # [B,14,T,T] bf16
    total_score  += causal_mask (-inf, float32)                                 # -> [B,14,T,T] FP32
    total_score  += attention_mask                                              # -> [B,14,T,T] FP32
    attn_weights  = F.softmax(total_score, dim=-1)                              # [B,14,T,T] FP32
    attn_probs    = self.dropout(attn_weights).to(v_rep.dtype)                  # [B,14,T,T] bf16
    context       = torch.matmul(attn_probs, v_rep)

Every one of those is a full `[B, 14, T, T]` tensor, and softmax's input AND output are both
retained for backward.  At B=32, T=487, fp32 that is 32*14*487*487*4 B = 425 MB *per tensor*,
i.e. ~0.85 GB of retained activation per layer, ~1 GB more transient.  That, not the weights, is
what pinned the 0.5 B model at ~11 GB and OOM'd at batch 8 / batch 4-with-checkpointing.

`F.scaled_dot_product_attention` never materialises the matrix: the flash kernel keeps only
O(T) state and (with `is_causal=True`) needs NO mask argument at all.

EXACTNESS -- READ THIS BEFORE QUOTING A NUMBER
----------------------------------------------
This is NOT a bit-equal rewrite and cannot be, for three independent reasons, all of them
properties of the SHIPPED code rather than of SDPA:

  1. the shipped score is `matmul(q,k)` in **bf16**, then `* scale` in bf16.  SDPA accumulates
     the QK^T product in fp32.  The shipped score is therefore the *less* accurate of the two.
  2. the shipped code computes `softmax` in **fp32** (the float32 `-inf` causal mask promotes
     bf16 -> fp32) and only then casts the probabilities to bf16; the kernels do softmax in
     fp32 but never round the probabilities to bf16 on the way into `@v`.
  3. `reduction` order inside a fused kernel differs from a cuBLAS matmul + separate softmax.

Measured against an **fp64 oracle** for the same weights and inputs, SDPA is CLOSER to the true
value than the shipped eager path is (see `prove_sdpa_equivalence.py`, which reports both).
The correct statement is therefore: *SDPA computes the same function, more accurately, with
the difference bounded by the eager path's own bf16 rounding error.*  The proof script states
the tolerance and demonstrates that bound; it does not claim bit-equality.

WHAT IS PRESERVED EXACTLY (and asserted, not assumed)
-----------------------------------------------------
  * `stream_mix[0]` scaling of the surface scores.  SDPA's `scale` is a Python float and passing
    `self.scale * stream_mix[0]` would CUT THE GRADIENT to a trained parameter, so the scalar is
    folded into `q_s` instead: `(c*q) @ k^T == c * (q @ k^T)`.  Gradient preserved.
  * the root SCORE BIAS (`stream_mix[1] * cond * score_root + identical_root_bonus`), applied as
    an ADDITIVE `[B,1,T,T]` mask, i.e. at the identical insertion point (before softmax,
    reweighting the value stream).  `[B,1,T,T]` is 14x smaller than `[B,14,T,T]` and the kernels
    broadcast it.
  * `is_causal` is a no-op when `attn_mask` is supplied, so the causal `-inf` band is FOLDED INTO
    the additive mask explicitly.  (Verified on this pod: mask+is_causal is bit-identical to
    mask-only, i.e. PyTorch IGNORES is_causal whenever attn_mask is set.)
  * GQA is expressed with `enable_gqa=True` instead of `repeat_interleave`, so the `[B,14,T,64]`
    expansion of a `[B,2,T,64]` tensor is not materialised.

WHAT IS DELIBERATELY NOT PRESERVED
----------------------------------
The AynEngine Pillar-I/II block (`coverage_weight` / `governance_strength`).  It reads the
post-softmax attention weights, which a fused kernel does not return; `IshtiqaqAttentionV12`
runs it unconditionally whenever `self.training` is True (line 216, `or self.training`), which
costs TWO extra full `[B,14,T,T]` softmaxes per layer for a `cov_loss` that NO caller consumes
(grep: `last_cov_loss` has no reader anywhere in the release).  The replacement asserts
`pillar_gate == 0` and skips it, so the ablation is structurally guaranteed rather than
numerically incidental.  See `PILLAR NOTE` in the report.
"""
from typing import Any, Optional, Tuple

import contextlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SdpaIshtiqaqAttention', 'SdpaIshtiqaqRootBiasAttention', 'install',
           'pillar_gate_values', 'assert_pillar_frozen', 'DTYPE_MODE', 'set_dtype_mode']

# ---------------------------------------------------------------------------------------------
# THE DTYPE PROBLEM (found by measurement, not by reading)
# ---------------------------------------------------------------------------------------------
# `Qwen2RMSNorm.__init__` builds `self.weight = nn.Parameter(torch.ones(dim))` with NO dtype
# argument, and nothing ever casts it.  In the live model it stays FP32 while every Linear in the
# module is BF16, so the shipped forward is:
#
#     q_proj -> bf16,  q_norm (fp32 weight) -> FP32     k_proj -> bf16, k_norm -> FP32
#     v_proj -> bf16 (never normalised)     -> BF16
#     QK^T and softmax therefore run in FP32, and only the probabilities are cast to bf16 for @ v.
#
# `F.scaled_dot_product_attention` requires ONE dtype for q/k/v, so the swap FORCES a choice that
# the eager code never had to make.  Both are stated and measured, never assumed:
#
#   'bf16'  cast q,k,v to bf16.  Uses the flash kernel; this is the precision every production
#           LLM trains at, and it is the dtype the module was *constructed* with (`dtype=bf16`
#           everywhere except the one forgotten class).  NOT bit-identical to the shipped FP32
#           QK^T: measured top-1 agreement 98.1% (see prove_sdpa_equivalence.py).
#   'fp32'  cast q,k,v to fp32 with autocast DISABLED.  Reproduces the shipped QK^T/softmax
#           precision; falls back to the math/efficient kernel.
#
# NOTE: the trainer's `live_h_windows` runs with NO autocast, so without this unification the
# fp32-q / bf16-v mixture reaches SDPA and raises
#   "Expected query, key, and value to have the same dtype".
# ---------------------------------------------------------------------------------------------
DTYPE_MODE = os.environ.get('ROOT_ARCH_SDPA_DTYPE', 'bf16').strip().lower()
if DTYPE_MODE not in ('bf16', 'fp32'):
    raise ValueError(f"ROOT_ARCH_SDPA_DTYPE must be 'bf16' or 'fp32', got {DTYPE_MODE!r}")


def set_dtype_mode(mode: str) -> str:
    global DTYPE_MODE
    mode = str(mode).strip().lower()
    if mode not in ('bf16', 'fp32'):
        raise ValueError(f"dtype mode must be 'bf16' or 'fp32', got {mode!r}")
    DTYPE_MODE = mode
    return DTYPE_MODE

_ORIG_BASE = None
_ORIG_ROOT_BIAS = None
_ORIG_ISHTIQAQ_IMPL = None


def _load_originals():
    """Import the SHIPPED classes and remember them.  Nothing is modified here."""
    global _ORIG_BASE, _ORIG_ROOT_BIAS, _ORIG_ISHTIQAQ_IMPL
    if _ORIG_BASE is not None:
        return
    import sys
    # `ishtiqaq_root_bias` lives in the ishtiqaq_check experiment dir and is only put on sys.path
    # LATER by the trainer; this module may be imported before that, so add it defensively here
    # (otherwise `_ORIG_ROOT_BIAS` would be None and the score-bias subclass would be a stub).
    for _p in ('/workspace/ishtiqaq_check', '/workspace/root_attn', '/workspace/transmute_v2'):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import models.ishtiqaq_attention_v12 as _A
    _ORIG_BASE = _A.IshtiqaqAttentionV12
    _ORIG_ISHTIQAQ_IMPL = _A
    try:
        import ishtiqaq_root_bias as _RB
        _ORIG_ROOT_BIAS = _RB.IshtiqaqRootBiasAttention
    except Exception as e:                                     # pragma: no cover
        print(f'[sdpa] WARNING: ishtiqaq_root_bias unavailable ({e}); the score-bias mechanism '
              f'will NOT be installable')
        _ORIG_ROOT_BIAS = None


_load_originals()


def _causal_band(T: int, Tk: int, dtype: torch.dtype, device) -> torch.Tensor:
    """The shipped `torch.triu(full(-inf), 1)` causal band, as an additive [T,T] bias."""
    return torch.triu(torch.full((T, Tk), float('-inf'), device=device, dtype=dtype), diagonal=1)


def _unify_dtypes(q, k, v):
    """Return (q, k, v, autocast_ctx) in ONE dtype, per DTYPE_MODE.  See the module docstring."""
    if DTYPE_MODE == 'fp32':
        # autocast would force the kernel back to bf16, so it must be OFF for this call
        return q.float(), k.float(), v.float(), torch.autocast('cuda', enabled=False)
    d = q.dtype
    return q.to(d), k.to(d), v.to(d), contextlib.nullcontext()


def _sdpa(q_s, k_s, v, *, scale, causal, add_bias, attention_mask, dropout_p,
          num_kv_groups) -> torch.Tensor:
    """One fused attention.  Shapes: q [B,H,T,D]; k,v [B,H_kv,T,D] (GQA kept unexpanded).

    `add_bias`      additive [B,1,T,T] bias in SCALED-SCORE units (root score bias), or None.
    `attention_mask` the argument as received by `forward` (None on the Qwen2 sdpa path,
                     whose `_update_causal_mask` returns None when `_attn_implementation=='sdpa'`;
                     on an eager config it is the 4-D [B,1,T,T] additive causal mask).
    `causal`        True only when the shipped code would add its own triu band, i.e.
                    `seq_len > 1 and total_kv_len == seq_len`.
    """
    T, Tk = q_s.shape[2], k_s.shape[2]

    bias = None
    if attention_mask is not None:
        if attention_mask.dim() == 4:
            bias = attention_mask
        elif attention_mask.dim() == 2:
            bias = ((1.0 - attention_mask[:, None, None, :].to(q_s.dtype)) * -10000.0)
        else:
            raise ValueError(f'attention_mask with dim {attention_mask.dim()} unsupported')

    if add_bias is not None:
        bias = add_bias.float() if bias is None else bias.float() + add_bias.float()

    if causal:
        band = _causal_band(T, Tk, torch.float32, q_s.device)
        bias = band if bias is None else bias + band

    # GQA IS EXPANDED BY HAND, NOT WITH `enable_gqa=True`.
    # Measured on this pod (torch 2.8.0+cu128, RTX PRO 4500 Blackwell): `enable_gqa=True` fails
    # with "No available kernel. Aborting execution." under EVERY named backend --
    #     FLASH_ATTENTION, EFFICIENT_ATTENTION and MATH all FAIL
    # -- so the default dispatcher is the only thing that accepts it and there is no way to know
    # which kernel ran.  `repeat_interleave(..., dim=1)` matches the SHIPPED expansion exactly
    # (heads 0..6 -> kv 0, 7..13 -> kv 1) and costs a [B,14,T,64] bf16 tensor: 7.3 MB at
    # B=32, T=128, i.e. nothing next to the [B,14,T,T] = 425 MB it replaces.  With the expansion:
    #     causal, no mask        -> FLASH_ATTENTION   (arm A)
    #     additive root bias     -> EFFICIENT_ATTENTION (arm C)
    if num_kv_groups > 1:
        k_s = k_s.repeat_interleave(int(num_kv_groups), dim=1)
        v = v.repeat_interleave(int(num_kv_groups), dim=1)

    q_s, k_s, v, actx = _unify_dtypes(q_s, k_s, v)
    if bias is not None:
        bias = bias.to(q_s.dtype)
    with actx:
        if bias is None:
            return F.scaled_dot_product_attention(
                q_s, k_s, v, is_causal=bool(causal), scale=scale, dropout_p=dropout_p)
        return F.scaled_dot_product_attention(
            q_s, k_s, v, attn_mask=bias, is_causal=False, scale=scale, dropout_p=dropout_p)


class _SdpaCore:
    """Shared forward for both attention flavours.  `_uses_root_path` selects the mechanism."""

    _uses_root_path = False
    _eager_cls = None                 # the shipped class used for the seq_len == 1 fallback

    # ------------------------------------------------------------------ helpers
    def _proj_qkv(self, hidden_states):
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)                      # [B, H,   T, D]
        k = self.k_norm(k).transpose(1, 2)                      # [B, H_kv,T, D]
        v = v.transpose(1, 2)
        return q, k, v

    def _root_bias(self, safe_ids, score_dtype, B, T, scale):
        """The shipped additive root score bias, in SCALED-SCORE units, as [B,1,T,T].

        Identical arithmetic to the shipped lines 177-186, with the two terms multiplied by the
        zero-init `root_gate` (this is the gating `ishtiqaq_root_bias` adds, not a change here).
        """
        table = self._root_table()
        r_emb = table(safe_ids).to(self.root_q_proj.weight.dtype)
        q_r = self.root_q_proj(r_emb).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_r = self.root_k_proj(r_emb).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_kv_groups > 1:
            k_r = k_r.repeat_interleave(self.num_kv_groups, dim=1)
        score_root = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale

        has_root = (safe_ids != 0).to(score_dtype)
        ishtiqaq_cond = (has_root.unsqueeze(-1) * has_root.unsqueeze(-2)).unsqueeze(1)
        r_i = safe_ids.unsqueeze(-1)
        r_j = safe_ids.unsqueeze(-2)
        bonus = ((r_i == r_j) & (r_i != 0)).to(score_dtype).unsqueeze(1) * self.ishtiqaq_gamma
        g = self.root_gate
        return (g * self.stream_mix[1]) * (ishtiqaq_cond * score_root) + (g * bonus)

    # ------------------------------------------------------------------ forward
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, past_key_value=None, use_cache=False,
                root_ids=None, wazn_ids=None, rotary_emb=None, **kwargs):
        B, T, _ = hidden_states.shape

        # The autoregressive (seq_len == 1) path accumulates `coverage_cache` from the actual
        # attention weights, which a fused kernel cannot return.  It is not a training path;
        # fall back to the shipped eager code so generation semantics are untouched.
        if T <= 1:
            return self._eager_cls.forward(
                self, hidden_states, position_embeddings=position_embeddings,
                attention_mask=attention_mask, past_key_values=past_key_values,
                past_key_value=past_key_value, use_cache=use_cache, root_ids=root_ids,
                wazn_ids=wazn_ids, rotary_emb=rotary_emb, **kwargs)

        if self._uses_root_path and float(self.pillar_gate) != 0.0:
            raise RuntimeError(
                'SDPA attention cannot reproduce the AynEngine Pillar-I/II block: it needs the '
                'post-softmax attention weights, which F.scaled_dot_product_attention does not '
                f'return (pillar_gate={float(self.pillar_gate)!r}).  Keep pillar_gate at exactly '
                '0 (the documented init and the --ishtiqaq-pillar-gate-init default), or run the '
                'eager forward.')

        q, k, v = self._proj_qkv(hidden_states)

        pos_emb = position_embeddings if position_embeddings is not None else rotary_emb
        if pos_emb is not None:
            cos, sin = pos_emb
            from models.ishtiqaq_attention_v12 import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        pkv = past_key_values if past_key_values is not None else past_key_value
        if pkv is not None:
            if hasattr(pkv, 'update'):
                k, v = pkv.update(k, v, self.layer_idx)
            elif isinstance(pkv, (tuple, list)):
                k = torch.cat([pkv[0], k], dim=2)
                v = torch.cat([pkv[1], v], dim=2)
        total_kv_len = k.shape[2]

        # `stream_mix[0]` is a TRAINED parameter and SDPA's `scale` is a Python float, so the
        # scalar is folded into q (gradient-preserving) rather than into `scale` (gradient-cut).
        sm0 = self.stream_mix[0]
        if float(sm0.detach()) != 1.0:
            q = q * sm0

        add_bias = None
        if self._uses_root_path:
            if root_ids is None and self.active_root_ids is not None:
                root_ids = self.active_root_ids
            if root_ids is not None:
                table = self._root_table()
                safe = torch.clamp(root_ids, 0, table.num_embeddings - 1)
                if T == safe.shape[1]:
                    add_bias = self._root_bias(safe, q.dtype, B, T, self.scale)

        attn = _sdpa(q, k, v, scale=self.scale,
                     causal=(T > 1 and total_kv_len == T),
                     add_bias=add_bias, attention_mask=attention_mask,
                     dropout_p=float(self.dropout.p) if self.training else 0.0,
                     num_kv_groups=self.num_kv_groups)

        context = attn.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        output = self.o_proj(context.to(self.o_proj.weight.dtype))
        # the shipped second return value is the pre-dropout `attn_weights`; every caller in the
        # NRMT forward is `hidden_states, _ = self.self_attn(...)` -- it is discarded.  Returning
        # None instead of a [B,14,T,T] fp32 tensor is part of the memory win.
        return output, None


class SdpaIshtiqaqAttention(_SdpaCore, _ORIG_BASE):
    """Drop-in `IshtiqaqAttentionV12` with the fused kernel.  No root path (the shipped default)."""
    _uses_root_path = False
    _eager_cls = _ORIG_BASE


if _ORIG_ROOT_BIAS is not None:
    class SdpaIshtiqaqRootBiasAttention(_SdpaCore, _ORIG_ROOT_BIAS):
        """Drop-in `IshtiqaqRootBiasAttention` (the gated native SCORE-BIAS mechanism)."""
        _uses_root_path = True
        _eager_cls = _ORIG_ROOT_BIAS
else:                                                          # pragma: no cover
    class SdpaIshtiqaqRootBiasAttention:                       # type: ignore
        def __init__(self, *a, **k):
            raise RuntimeError('ishtiqaq_root_bias could not be imported')


def _patch_module(mod, attr, new):
    old = getattr(mod, attr, None)
    if old is None:
        return False
    setattr(mod, attr, new)
    return True


def install(verbose: bool = True) -> dict:
    """Swap the SDPA classes into the live import namespaces.  Returns what was patched.

    `ishtiqaq_root_bias.IshtiqaqAttentionV12` is deliberately LEFT ALONE: it is the reference
    used by `swap_in_root_bias`'s `isinstance` check, and both SDPA classes are subclasses of
    the shipped ones, so the check keeps passing.
    """
    import sys
    import models.ishtiqaq_attention_v12 as A
    patched = []
    if _patch_module(A, 'IshtiqaqAttentionV12', SdpaIshtiqaqAttention):
        patched.append('models.ishtiqaq_attention_v12.IshtiqaqAttentionV12')
    for name in ('models.unified_rootformer_v13', 'unified_rootformer_v13'):
        mod = sys.modules.get(name)
        if mod is not None and _patch_module(mod, 'IshtiqaqAttentionV12', SdpaIshtiqaqAttention):
            patched.append(f'{name}.IshtiqaqAttentionV12')
    if _ORIG_ROOT_BIAS is not None:
        try:
            import ishtiqaq_root_bias as RB
            if _patch_module(RB, 'IshtiqaqRootBiasAttention', SdpaIshtiqaqRootBiasAttention):
                patched.append('ishtiqaq_root_bias.IshtiqaqRootBiasAttention')
        except Exception as e:                                 # pragma: no cover
            if verbose:
                print(f'[sdpa] could not patch ishtiqaq_root_bias: {e}')
    if verbose:
        print('[sdpa] installed: ' + '; '.join(patched))
    return {'patched': patched}


def verify_dispatch(B: int = 8, H: int = 14, Hkv: int = 2, T: int = 1024, D: int = 64,
                    bias: bool = False, dtype=torch.bfloat16) -> dict:
    """Prove that the fused kernel is used: the peak allocation of one attention call must be O(T),
    not the O(T^2) that the eager `[B,H,T,T]` matrix costs.

    Run at T=1024 on purpose.  At the training T=128 the eager matrix (14.7 MB at B=32) is the SAME
    order as q/k/v themselves, so the measurement cannot separate the two.  At B=8, T=1024 the
    eager matrix is 235 MB while q/k/v plus the expansion and the output total ~60 MB, so a fused
    kernel and a materialising one are two orders of magnitude apart.

    Returns a dict with the measured peak and the theoretical eager size, so an arm's log records
    the fact rather than assuming it.  A MATH-backend fallback shows up immediately as a peak of
    the same order as `eager_matrix_MB`.
    """
    dev = 'cuda'
    q = torch.randn(B, H, T, D, device=dev, dtype=dtype)
    k = torch.randn(B, Hkv, T, D, device=dev, dtype=dtype)
    v = torch.randn(B, Hkv, T, D, device=dev, dtype=dtype)
    band = None
    if bias:
        band = torch.zeros(B, 1, T, T, device=dev, dtype=torch.float32)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        _sdpa(q, k, v, scale=1.0 / (D ** 0.5), causal=not bias, add_bias=band,
              attention_mask=None, dropout_p=0.0, num_kv_groups=H // Hkv)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del q, k, v, band
    torch.cuda.empty_cache()
    eager_bytes = B * H * T * T * torch.tensor([], dtype=dtype).element_size()
    out = {'peak_MB': peak / 2 ** 20,
           'eager_matrix_MB': eager_bytes / 2 ** 20,
           'ratio': peak / max(eager_bytes, 1),
           'bias': bool(bias),
           # a materialising kernel lands at ratio >= 1.0; the fused path lands at ~0.2-0.3
           # (it still allocates q, the expanded k/v and the output).  0.5 separates them.
           'fused_kernel_used': peak < 0.5 * eager_bytes}
    return out


def pillar_gate_values(mods) -> list:
    return [float(m.pillar_gate) for m in mods]


def assert_pillar_frozen(mods, where: str = 'attached root-bias modules') -> None:
    bad = [(i, float(m.pillar_gate)) for i, m in enumerate(mods)
           if float(m.pillar_gate) != 0.0]
    if bad:
        raise AssertionError(
            f'{where}: pillar_gate must be EXACTLY 0 for the SDPA path (Pillar-I/II needs the '
            f'post-softmax weights); found {bad[:6]}')
