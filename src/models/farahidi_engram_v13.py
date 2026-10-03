#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
farahidi_engram_v13.py
Farāhīdian Engram & Causal Encoder-Decoder (CED) Module.
Inspired by DeepSeek-V4.1-Flash (arXiv:2609.19969v1):
- Decouples static 9,015-root Farāhīdian lexical memory from dynamic syntactic computation.
- Sparse O(1) multi-head memory lookup with Context-Aware Gating at the cross-lingual bottleneck (Layer 12).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class FarahidiEngramModule(nn.Module):
    def __init__(
        self,
        hidden_size: int = 896,
        engram_dim: int = 256,
        num_roots: int = 9016,
        num_awzan: int = 128,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.engram_dim = engram_dim
        
        # 1. Static Farāhīdian Root & Wazn Memory Tables (Decoupled Lexicon)
        self.root_engram_table = nn.Embedding(num_roots, engram_dim, dtype=dtype)
        self.wazn_engram_table = nn.Embedding(num_awzan, engram_dim, dtype=dtype)
        
        # 2. Context-Aware Gating Projections (DeepSeek-V4.1-Flash Eq. 2)
        self.gate_proj = nn.Linear(hidden_size, engram_dim, bias=True, dtype=dtype)
        self.mem_proj = nn.Linear(engram_dim, hidden_size, bias=False, dtype=dtype)
        self.norm = nn.LayerNorm(hidden_size, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        root_ids: torch.Tensor,
        wazn_ids: torch.Tensor
    ) -> torch.Tensor:
        # Retrieve static Farāhīdian root and morphological invariants in O(1)
        safe_roots = torch.clamp(root_ids, 0, 9015)
        safe_awzan = torch.clamp(wazn_ids, 0, 127)
        r_mem = self.root_engram_table(safe_roots)
        w_mem = self.wazn_engram_table(safe_awzan)
        combined_mem = r_mem + w_mem
        
        # Context-Aware Gate (decides how much root invariant to infuse)
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        
        # Inject sparse memory into the Transformer hidden state
        injected = self.mem_proj(gate * combined_mem)
        return self.norm(hidden_states + injected)
