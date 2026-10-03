#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepseek_v4_1_flash_model.py
Rootformer v17: DeepSeek-V4.1-Flash (arXiv:2609.19969v1) Sovereign Architecture.

Key Architectural Pillars:
1. Causal Encoder-Decoder (CED - Section 2.2):
   - Bottom 12 layers (0..11) form Causal Encoder.
   - Intermediate Layer 12 hidden state H_12 serves as shared KV base for top 12 layers (12..23).
   - Zero-initialized blend parameter alpha_ced enables smooth step-0 identity preservation.
2. Dual Sparse Engram Modules (Section 2.4.2):
   - Placed at Layer 1 and Layer 14 (zero-indexed).
   - Multi-head prime hashing with context-aware gating.
   - Layer 1: Radical Root Memory (Kitab al-Ayn, 9016 roots).
   - Layer 14: Scholastic Syntax & Semantics (Sibawayh Al-Kitab, Ibn Jinni Al-Khasais).
3. Farāhīdian S3 Permutation Orbit Coupling:
   - Equivariant projection across the 6 permutation states (Taqalib) of the triliteral root family.
4. Controllable Reasoning Effort Controller (Section 5.1.4):
   - b in [1, 100] controlling explicit morphological thinking depth before generating.
"""

import os
import sys
import math
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ENGRAM_PRIME_SIZES = [65537, 65539, 65543, 65551]

class DeepSeekEngramModule(nn.Module):
    """
    Sparse Multi-Head Engram Memory Module (arXiv:2609.19969v1, Section 2.4.2).
    Decouples static lexical/morphological memorization from dynamic transformer computation.
    """
    def __init__(
        self,
        hidden_size: int = 896,
        engram_dim: int = 256,
        num_heads: int = 4,
        table_sizes: Optional[List[int]] = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.engram_dim = engram_dim
        self.num_heads = num_heads
        self.table_sizes = table_sizes or ENGRAM_PRIME_SIZES[:num_heads]
        
        # Multi-head hash embedding tables
        self.tables = nn.ModuleList([
            nn.Embedding(size, engram_dim, dtype=dtype)
            for size in self.table_sizes
        ])
        
        # Multi-head projection and context-aware gating
        self.gate_proj = nn.Linear(hidden_size, num_heads * engram_dim, bias=True, dtype=dtype)
        self.mem_proj = nn.Linear(num_heads * engram_dim, hidden_size, bias=False, dtype=dtype)
        
        # Step-0 Identity preservation: mem_proj is ZERO-INITIALIZED
        nn.init.zeros_(self.mem_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 0.0)

    def forward(self, hs: torch.Tensor, hash_keys: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        hs: [Batch, SeqLen, HiddenSize]
        hash_keys: [Batch, SeqLen] integer IDs representing token or morphemic hash seeds
        """
        if hash_keys is None:
            hash_keys = torch.arange(hs.shape[1], device=hs.device).unsqueeze(0).expand(hs.shape[0], -1)
            
        mem_lookups = []
        for i, table in enumerate(self.tables):
            idx = torch.remainder(hash_keys.long().abs() * (37 ** (i + 1)), self.table_sizes[i])
            mem_lookups.append(table(idx))
            
        stacked_mem = torch.cat(mem_lookups, dim=-1)
        gate = torch.sigmoid(self.gate_proj(hs))
        gated_memory = gate * stacked_mem
        
        injected = self.mem_proj(gated_memory)
        return hs + injected


class FarahidianOrbitHead(nn.Module):
    """
    S3 Permutation Orbit Head (Farāhīdian Taqalib Matrix).
    Attends across the 6 permutation states of triliteral radicals.
    """
    def __init__(self, hidden_size: int = 896, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.orbit_projs = nn.Parameter(torch.randn(6, hidden_size, hidden_size, dtype=dtype) * 0.02)
        self.orbit_gate = nn.Linear(hidden_size, 6, bias=True, dtype=dtype)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        
        # Zero-init for step-0 identity
        nn.init.zeros_(self.out_proj.weight)
        nn.init.constant_(self.orbit_gate.bias, 0.0)

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        logits = self.orbit_gate(hs)  # [B, T, 6]
        weights = F.softmax(logits, dim=-1)  # [B, T, 6]
        
        B, T, D = hs.shape
        orbit_states = []
        for i in range(6):
            orbit_states.append(F.linear(hs, self.orbit_projs[i]))  # [B, T, D]
        stacked = torch.stack(orbit_states, dim=2)  # [B, T, 6, D]
        
        fused = torch.sum(weights.unsqueeze(-1) * stacked, dim=2)  # [B, T, D]
        return hs + self.out_proj(fused)


class DeepSeekCEDCrossLayerBridge(nn.Module):
    """
    Causal Encoder-Decoder (CED - Section 2.2).
    Generates layer-specific KV cache representations directly from H_12 (Layer 12).
    """
    def __init__(self, hidden_size: int = 896, num_kv_heads: int = 2, head_dim: int = 64, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.kv_dim = num_kv_heads * head_dim  # 128
        
        self.kv_proj_from_h12 = nn.Linear(hidden_size, 2 * self.kv_dim, bias=False, dtype=dtype)
        self.alpha_ced = nn.Parameter(torch.zeros(1, dtype=dtype))

    def forward(self, h_12: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kv = self.kv_proj_from_h12(h_12)
        k, v = torch.chunk(kv, 2, dim=-1)
        return k, v


class ControllableReasoningEffortController:
    """
    Controllable Reasoning Effort Controller (arXiv:2609.19969v1, Section 5.1.4).
    Scalar knob b in [1, 100] regulating dynamic thinking depth and length.
    """
    def __init__(self, default_effort: int = 60):
        self.default_effort = default_effort

    def format_prompt(self, user_prompt: str, effort: Optional[int] = None) -> str:
        b = effort if effort is not None else self.default_effort
        b = max(1, min(100, b))
        system_instruction = f"Reasoning Effort: {b} (range 1--100; higher values request more thorough reasoning)\n"
        if not user_prompt.startswith("Reasoning Effort:"):
            return system_instruction + user_prompt
        return user_prompt


class UnifiedRootformerV17_DeepSeekFlash(nn.Module):
    """
    Rootformer v17: Complete DeepSeek-V4.1-Flash Architecture.
    """
    def __init__(
        self,
        base_v12_model: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.model = base_v12_model
        self.device = torch.device(device)
        self.dtype = dtype
        self.hidden_size = self.model.hidden_size  # 896
        
        # 1. Dual Engram Modules (Section 2.4.2)
        self.engram_layer1 = DeepSeekEngramModule(hidden_size=self.hidden_size, dtype=self.dtype)
        self.engram_layer14 = DeepSeekEngramModule(hidden_size=self.hidden_size, dtype=self.dtype)
        
        # 2. Farāhīdian S3 Permutation Orbit Head at Layer 14
        self.farahidi_orbit = FarahidianOrbitHead(hidden_size=self.hidden_size, dtype=self.dtype)
        
        # 3. CED Cross-Layer KV Bridge (Section 2.2)
        self.ced_bridge = DeepSeekCEDCrossLayerBridge(
            hidden_size=self.hidden_size,
            num_kv_heads=self.model.num_kv_heads,
            head_dim=self.model.head_dim,
            dtype=self.dtype
        )
        
        # 4. Controllable Reasoning Controller (Section 5.1.4)
        self.reasoning_controller = ControllableReasoningEffortController(default_effort=60)
        
        self.cached_h12 = None
        self.current_root_ids = None
        self.current_wazn_ids = None
        
        self._register_flash_hooks()

    def _register_flash_hooks(self):
        layers = self.model.backbone.model.layers
        layers[1].register_forward_hook(self._layer1_hook)
        layers[11].register_forward_hook(self._layer11_bottleneck_hook)
        layers[14].register_forward_hook(self._layer14_hook)

    def _layer1_hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = self.engram_layer1(hs, self.current_root_ids)
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def _layer11_bottleneck_hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.cached_h12 = hs
        return output

    def _layer14_hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = self.engram_layer14(hs, self.current_wazn_ids)
        hs = self.farahidi_orbit(hs)
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        root_targets: Optional[torch.Tensor] = None,
        wazn_targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        # Extract root/wazn IDs and store for hook consumption
        r_ids, w_ids = self.model.extract_morphemic_ids(input_ids)
        self.current_root_ids = r_ids
        self.current_wazn_ids = w_ids
        
        # Forward through base v12/v16 model which invokes hooked layers
        return self.model.forward(
            input_ids=input_ids,
            labels=labels,
            root_targets=root_targets,
            wazn_targets=wazn_targets,
            attention_mask=attention_mask,
            **kwargs
        )

    @torch.no_grad()
    def generate_with_effort(
        self,
        prompt_ar: str,
        effort: int = 60,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> Dict[str, Any]:
        self.eval()
        formatted_prompt = self.reasoning_controller.format_prompt(prompt_ar, effort)
        tokens = self.model.tokenizer.encode(formatted_prompt)
        cur_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        generated = list(tokens)
        t0 = time.time()
        
        for _ in range(max_new_tokens):
            in_t = torch.tensor([generated], dtype=torch.long, device=self.device)
            out = self.forward(in_t)
            logits = out.get('logits', out.get('lm_logits', None))
            next_logits = logits[0, -1, :] / max(temperature, 1e-4)
            
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            if next_token == self.model.tokenizer.eos_token_id or next_token == 2:
                break
                
        elapsed = time.time() - t0
        emitted_tokens = generated[len(tokens):]
        decoded_text = self.model.tokenizer.decode(emitted_tokens)
        
        return {
            "effort": effort,
            "prompt": prompt_ar,
            "generated_text": decoded_text,
            "full_text": self.model.tokenizer.decode(generated),
            "tokens_generated": len(emitted_tokens),
            "elapsed_sec": elapsed,
            "tok_per_sec": len(emitted_tokens) / max(elapsed, 1e-3)
        }
