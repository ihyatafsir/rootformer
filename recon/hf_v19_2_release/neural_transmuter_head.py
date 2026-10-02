#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neural_transmuter_head.py
The Neural Farāhīdian Transmutation Head (الرأس العصبي للإحالة الدلالية).

Upgraded Deep 8-Layer Architecture with Multi-Stage Heritage Guidance:
- Layers 0-1: Layer 8 Cross-Attention (Morphology & Root Template Orbits)
- Layers 2-5: Layer 14 Cross-Attention (Sībawayh Syntactic Governance & Case Dependency)
- Layers 6-7: Layer 24 Cross-Attention (Macro-Semantic Scholastic Cohesion)

Anchored by 177,256 Classical Root Priors across all 9,195 roots of Lisān al-ʿArab.
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

class ConceptCrossAttentionBlock(nn.Module):
    def __init__(self, d_concept: int = 512, d_model: int = 896, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_concept = d_concept
        self.self_attn = nn.MultiheadAttention(embed_dim=d_concept, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_concept)

        # Cross-attention projecting Arabic memory states into English concept space
        self.k_proj = nn.Linear(d_model, d_concept)
        self.v_proj = nn.Linear(d_model, d_concept)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_concept, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_concept)

        self.ffn = nn.Sequential(
            nn.Linear(d_concept, d_concept * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_concept * 4, d_concept)
        )
        self.norm3 = nn.LayerNorm(d_concept)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 1. Causal Self-Attention over English Concepts
        sa_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + sa_out)

        # 2. Cross-Attention over Arabic representations
        k_mem = self.k_proj(memory)
        v_mem = self.v_proj(memory)
        ca_out, _ = self.cross_attn(x, k_mem, v_mem, key_padding_mask=memory_mask)
        x = self.norm2(x + ca_out)

        # 3. FFN
        x = self.norm3(x + self.ffn(x))
        return x


class NeuralFarahidianTransmuterHead(nn.Module):
    def __init__(
        self,
        d_model: int = 896,
        d_concept: int = 512,
        concept_vocab_size: int = 16384,
        num_roots: int = 9114,
        guide_layer_idx: int = 14,
        num_decoder_layers: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_concept = d_concept
        self.concept_vocab_size = concept_vocab_size
        self.num_roots = num_roots
        self.guide_layer_idx = guide_layer_idx
        self.num_decoder_layers = num_decoder_layers

        # 1. English Concept Embeddings (Whole Scholastic Lemmas)
        self.concept_embeddings = nn.Embedding(concept_vocab_size, d_concept)
        nn.init.normal_(self.concept_embeddings.weight, mean=0.0, std=0.02)

        # 2. Deep Guided Cross-Attention Decoder Blocks (8 Layers)
        self.decoder_layers = nn.ModuleList([
            ConceptCrossAttentionBlock(d_concept=d_concept, d_model=d_model, n_heads=8, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.final_norm = nn.LayerNorm(d_concept)

        # 3. Direct Prediction Projection (Tied with Concept Embeddings)
        self.output_bias = nn.Parameter(torch.zeros(concept_vocab_size))

        # 4. Learnable Root-Conditioned Semantic Prior Gate (The "What" Gate)
        self.root_gate_embed = nn.Embedding(num_roots, d_concept)
        nn.init.normal_(self.root_gate_embed.weight, mean=0.0, std=0.02)

        # 5. Syntactic Role Classifier (The "When" Gate)
        self.role_classifier = nn.Linear(d_concept, 8)

        # 6. Soft Prior Tensor (Loaded from root_concept_prior.json)
        self.register_buffer('root_prior_bias', None, persistent=False)

    def load_root_prior_matrix(self, prior_json_path: Path, token_to_id: Dict[str, int], scale: float = 1.5):
        """Loads the root prior table into the prior bias tensor."""
        if not prior_json_path.exists():
            print(f"[Transmuter] Warning: Prior file {prior_json_path} not found.")
            return

        print(f"[Transmuter] Compiling Root Prior Tensor from {prior_json_path.name}...", flush=True)
        with open(prior_json_path, 'r', encoding='utf-8') as f:
            priors = json.load(f)

        prior_tensor = torch.zeros(self.num_roots, self.concept_vocab_size, dtype=self.concept_embeddings.weight.dtype)
        root_map_file = prior_json_path.parent / 'nrmp_vocab.json'
        root2id = {}
        if root_map_file.exists():
            with open(root_map_file, 'r', encoding='utf-8') as f:
                nrmp_data = json.load(f)
                roots_list = nrmp_data.get('roots', [])
                if isinstance(roots_list, list) and roots_list:
                    root2id = {r: idx for idx, r in enumerate(roots_list)}
                else:
                    root2id = nrmp_data.get('root_to_id', {})

        applied_count = 0
        for root_str, concept_list in priors.items():
            r_id = root2id.get(root_str, None)
            if r_id is None:
                norm_r = root_str.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ء', 'ا')
                r_id = root2id.get(norm_r, None)
            if r_id is None or r_id >= self.num_roots:
                continue
            for item in concept_list:
                cid = item['concept_id']
                if cid < self.concept_vocab_size:
                    prior_tensor[r_id, cid] = scale
                    applied_count += 1

        self.root_prior_bias = prior_tensor.to(device=self.concept_embeddings.weight.device, dtype=self.concept_embeddings.weight.dtype)
        print(f"[Transmuter] Successfully anchored {applied_count:,} root-concept priors into Multi-Stage guidance.", flush=True)

    def get_layer_memory(self, hidden_states: Any, layer_idx: int) -> torch.Tensor:
        """Extracts appropriate layer memory for multi-stage guidance."""
        if isinstance(hidden_states, (tuple, list)):
            n_layers = len(hidden_states)
            if self.num_decoder_layers >= 8 and n_layers >= 25:
                if layer_idx < 2:
                    return hidden_states[8]
                elif layer_idx < 6:
                    return hidden_states[14]
                else:
                    return hidden_states[24]
            idx = min(self.guide_layer_idx, n_layers - 1)
            return hidden_states[idx]
        return hidden_states

    def extract_layer14_states(self, hidden_states: Any) -> torch.Tensor:
        """Extracts Layer 14 hidden states from backbone output."""
        if isinstance(hidden_states, (tuple, list)):
            idx = min(self.guide_layer_idx, len(hidden_states) - 1)
            return hidden_states[idx]
        return hidden_states

    def forward(
        self,
        hidden_states: Any,
        target_concept_ids: torch.Tensor,
        active_root_ids: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        B, M = target_concept_ids.shape
        device = target_concept_ids.device

        input_ids = target_concept_ids[:, :-1]
        targets = target_concept_ids[:, 1:].contiguous()
        seq_len = input_ids.shape[1]

        x = self.concept_embeddings(input_ids)

        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=x.dtype), diagonal=1)

        # Multi-stage cross-attention pass through 8 deep decoder layers
        for layer_idx, layer in enumerate(self.decoder_layers):
            mem = self.get_layer_memory(hidden_states, layer_idx)
            x = layer(x, mem, causal_mask=causal_mask, memory_mask=memory_mask)
        x = self.final_norm(x)

        logits = F.linear(x, self.concept_embeddings.weight, self.output_bias)

        if active_root_ids is not None and self.root_prior_bias is not None:
            safe_roots = active_root_ids.clamp(0, self.num_roots - 1)
            p_bias = self.root_prior_bias[safe_roots].max(dim=1).values
            logits = logits + 0.3 * p_bias.unsqueeze(1).to(logits.dtype)

        role_logits = self.role_classifier(x)

        loss = None
        if targets is not None:
            active_loss = (targets > 0).view(-1)
            active_logits = logits.view(-1, self.concept_vocab_size)[active_loss]
            active_targets = targets.view(-1)[active_loss]
            if active_targets.numel() > 0:
                loss = F.cross_entropy(active_logits, active_targets, label_smoothing=0.05)

        return {
            "logits": logits,
            "role_logits": role_logits,
            "latent_states": x,
            "loss": loss
        }

    @torch.no_grad()
    def transmute(
        self,
        hidden_states: Any,
        active_root_ids: Optional[torch.Tensor] = None,
        id2concept: Optional[Dict[int, str]] = None,
        max_length: int = 40,
        temperature: float = 0.0,
        repetition_penalty: float = 1.25,
        bos_id: int = 2,
        eos_id: int = 3
    ) -> List[str]:
        self.eval()
        device = next(self.parameters()).device
        curr_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        generated_ids = []

        p_bias = None
        if active_root_ids is not None and self.root_prior_bias is not None:
            safe_roots = active_root_ids.clamp(0, self.num_roots - 1)
            p_bias = self.root_prior_bias[safe_roots].max(dim=1).values

        for step in range(max_length):
            seq_len = curr_ids.shape[1]
            x = self.concept_embeddings(curr_ids)
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=x.dtype), diagonal=1)

            for layer_idx, layer in enumerate(self.decoder_layers):
                mem = self.get_layer_memory(hidden_states, layer_idx)
                x = layer(x, mem, causal_mask=causal_mask)
            x = self.final_norm(x)

            next_logits = F.linear(x[:, -1, :], self.concept_embeddings.weight, self.output_bias)

            if p_bias is not None:
                next_logits = next_logits + 0.3 * p_bias

            if repetition_penalty != 1.0 and generated_ids:
                for prev_id in set(generated_ids):
                    if next_logits[0, prev_id] > 0:
                        next_logits[0, prev_id] /= repetition_penalty
                    else:
                        next_logits[0, prev_id] *= repetition_penalty

            if temperature > 0.0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = torch.argmax(next_logits, dim=-1).item()

            if next_id == eos_id and step > 0:
                break

            generated_ids.append(next_id)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_id]], device=device)], dim=1)

        if id2concept:
            words = []
            for cid in generated_ids:
                w = id2concept.get(cid, f"<c_{cid}>")
                if not w.startswith('<'):
                    words.append(w)
            return words
        return [str(cid) for cid in generated_ids]
