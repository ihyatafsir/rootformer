#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_rootformer_v12.py
Rootformer v12: Pure Sovereign Classical Arabic Rootformer.
Backbone: Clean Qwen2.5-0.5B (24 layers, 896 dim) + 9,856 Pure Arabic Vocab + IshtiqaqAttention.
"""

import os
import sys
import time
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from models.ishtiqaq_attention_v12 import IshtiqaqAttentionV12
from models.morphemic_tokenizer_v12_arabic import PureArabicMorphemicTokenizerV12

class DisjointMorphologicalHeadsV12(nn.Module):
    def __init__(self, hidden_size: int = 896, num_roots: int = 9015, num_awzan: int = 128, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.root_head = nn.Linear(hidden_size, num_roots, bias=False, dtype=dtype)
        self.wazn_head = nn.Linear(hidden_size, num_awzan, bias=False, dtype=dtype)

    def forward(self, morph_hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.root_head(morph_hidden_state), self.wazn_head(morph_hidden_state)

class UnifiedRootformerV12(nn.Module):
    def __init__(
        self,
        blueprint_path: Union[str, Path] = '/workspace/rootformer_v12/data/rootformer_v12_arabic_blueprint.json',
        base_model_name: str = 'Qwen/Qwen2.5-0.5B',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        is_training: bool = False
    ):
        super().__init__()
        self.blueprint_path = Path(blueprint_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.is_training = is_training

        # 1. Pure Arabic Tokenizer & Vocab
        self.tokenizer = PureArabicMorphemicTokenizerV12(self.blueprint_path)
        self.vocab_size = self.tokenizer.vocab_size  # 9,856

        # 2. Config & Backbone
        print(f'[Rootformer v12] Initializing Qwen2.5-0.5B config from {base_model_name}...', flush=True)
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.config.vocab_size = self.vocab_size
        self.hidden_size = self.config.hidden_size  # 896
        self.num_layers = self.config.num_hidden_layers  # 24
        self.num_heads = self.config.num_attention_heads  # 14
        self.num_kv_heads = self.config.num_key_value_heads  # 2
        self.head_dim = self.hidden_size // self.num_heads  # 64
        self.morph_layer_idx = 14  # Stratum for morphological supervision (60% depth)

        # 3. Backbone with IshtiqaqAttention across all 24 layers
        print(f'[Rootformer v12] Instantiating 24-layer backbone with IshtiqaqAttention...', flush=True)
        self.backbone = AutoModelForCausalLM.from_config(self.config, dtype=self.dtype)

        for layer_idx, layer in enumerate(self.backbone.model.layers):
            layer.self_attn = IshtiqaqAttentionV12(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                num_roots=9015,
                num_awzan=128,
                ishtiqaq_init_strength=0.25,
                dropout=0.0,
                layer_idx=layer_idx,
                dtype=self.dtype
            )

        # 4. Decoupled Morphological Auxiliary Heads at Layer 14
        self.morphemic_heads = DisjointMorphologicalHeadsV12(
            hidden_size=self.hidden_size,
            num_roots=9015,
            num_awzan=128,
            dtype=self.dtype
        )

        # 5. Metadata Mask for Text Generation (-inf on all <wazn_...> & <root_...>)
        self.metadata_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        for token_id in range(self.vocab_size):
            token_str = self.tokenizer.id_to_token.get(token_id, '')
            if token_str.startswith('<wazn_') or token_str.startswith('<root_') or token_str.startswith('<extra_id_'):
                self.metadata_mask[token_id] = True

        # 6. Radical Mapping Tables
        self.id_to_root_table = torch.zeros(self.vocab_size, dtype=torch.long, device=self.device)
        for token_id in range(self.vocab_size):
            token_str = self.tokenizer.id_to_token.get(token_id, '')
            if token_str.startswith('<root_'):
                # Root index from 1 to 9014
                root_part = token_id - 349  # Offset to root block
                self.id_to_root_table[token_id] = max(1, min(9014, root_part))

        self.to(self.device)
        if not self.is_training:
            self.eval()

        print(f'[Rootformer v12] Initialized! Total Layers: {self.num_layers} | Vocab: {self.vocab_size} on {self.device}. [OK]', flush=True)

    def extract_morphemic_ids(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        safe_inputs = torch.clamp(input_ids, 0, self.vocab_size - 1)
        root_ids = self.id_to_root_table[safe_inputs]
        wazn_ids = torch.zeros_like(safe_inputs)  # Optional wazn tracking
        return root_ids, wazn_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        root_targets: Optional[torch.Tensor] = None,
        wazn_targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        root_ids, wazn_ids = self.extract_morphemic_ids(input_ids)

        for layer in self.backbone.model.layers:
            layer.self_attn.active_root_ids = root_ids
            layer.self_attn.active_wazn_ids = wazn_ids

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        all_hidden_states = outputs.hidden_states
        h_morph = all_hidden_states[self.morph_layer_idx]
        lm_logits = outputs.logits

        # Auxiliary Morphological Heads at Layer 14
        root_logits, wazn_logits = self.morphemic_heads(h_morph)

        total_loss = None
        loss_lm = None
        loss_root = None
        loss_wazn = None

        if labels is not None:
            # 1. Causal LM Loss across 9,856 morphemes
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_lm = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            total_loss = loss_lm

            # 2. Auxiliary Morphological Supervision
            if root_targets is not None:
                loss_root = F.cross_entropy(
                    root_logits.view(-1, 9015),
                    root_targets.view(-1),
                    ignore_index=-100
                )
                total_loss = total_loss + 0.05 * loss_root

            if wazn_targets is not None:
                loss_wazn = F.cross_entropy(
                    wazn_logits.view(-1, 128),
                    wazn_targets.view(-1),
                    ignore_index=-100
                )
                total_loss = total_loss + 0.05 * loss_wazn

        return {
            'loss': total_loss,
            'loss_lm': loss_lm,
            'loss_root': loss_root,
            'loss_wazn': loss_wazn,
            'logits': lm_logits,
            'root_logits': root_logits,
            'wazn_logits': wazn_logits
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.4,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.20
    ) -> Dict[str, Any]:
        self.eval()
        input_ids_list = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)

        generated = list(input_ids_list)
        t0 = time.time()
        tokens_generated = 0

        for step in range(max_new_tokens):
            cur_tensor = torch.tensor([generated], dtype=torch.long, device=self.device)
            r_ids, w_ids = self.extract_morphemic_ids(cur_tensor)
            for layer in self.backbone.model.layers:
                layer.self_attn.active_root_ids = r_ids
                layer.self_attn.active_wazn_ids = w_ids

            outputs = self.backbone(cur_tensor, return_dict=True)
            next_logits = outputs.logits[0, -1, :].clone()

            # Permanent metadata mask: -inf on <wazn_...> & <root_...>
            next_logits[self.metadata_mask] = -float('Inf')

            # Repetition Penalty
            if repetition_penalty != 1.0 and len(generated) > 0:
                for token_id in set(generated[-80:]):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty

            if temperature <= 0.01:
                next_token = torch.argmax(next_logits).item()
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = -float('Inf')
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)
            tokens_generated += 1

            if next_token == 2 or next_token == self.tokenizer.token_to_id.get('<eos>', 2):
                break

        elapsed = time.time() - t0
        tps = tokens_generated / max(elapsed, 1e-4)

        full_text = self.tokenizer.decode(generated)
        prompt_text = self.tokenizer.decode(input_ids_list)
        new_text = self.tokenizer.decode(generated[len(input_ids_list):])

        return {
            'full_text': full_text,
            'new_text': new_text.strip(),
            'tokens_generated': tokens_generated,
            'elapsed_seconds': elapsed,
            'tokens_per_sec': tps
        }

    def transmute_from_clean_qwen2_5(self, base_model_name: str = 'Qwen/Qwen2.5-0.5B'):
        print(f'[Transmutation] Downloading clean pretrained weights from {base_model_name}...', flush=True)
        pretrained = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=self.dtype)
        p_state = pretrained.state_dict()

        b_state = self.backbone.state_dict()
        adapted_count = 0

        for k, v in b_state.items():
            if k in p_state:
                if v.shape == p_state[k].shape:
                    b_state[k].copy_(p_state[k])
                    adapted_count += 1
                elif 'embed_tokens.weight' in k or 'lm_head.weight' in k:
                    # Slice vocabulary from 151936 -> 9856
                    min_v = min(v.shape[0], p_state[k].shape[0])
                    b_state[k][:min_v].copy_(p_state[k][:min_v])
                    adapted_count += 1
                    print(f'  [Transmutation] Sliced {k}: {p_state[k].shape} -> {v.shape}')

        self.backbone.load_state_dict(b_state)
        print(f'[Transmutation Complete] Adapted {adapted_count} layers directly from clean {base_model_name}!')
        del pretrained
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate_bilingual(
        self,
        arabic_text: str,
        max_chars: int = 150,
        cov_w: float = 1.2,
        gov_s: float = 1.0,
        alpha_rep: float = 1.15,
        beta_eos: float = 2.0
    ) -> Dict[str, any]:
        """
        Generates English translation from Arabic input under AynEngine's Production Decoding Ladder:
        1. Pure Arabic Closed Vocabulary: English emitted char-by-char in SIMD slots 9366..9409.
        2. Order-4 Word N-gram Blocker (dissolves the 'statement of the statement' loop).
        3. Character Repetition Penalty (alpha=1.15 over 64 chars).
        4. Length-aware Dynamic EOS Boost (beta=2.0).
        """
        import string
        import time

        t0 = time.time()
        latin_chars = list(string.ascii_lowercase) + [' ', '.', ',', '-', ':', ';', '?', '!'] + [str(i) for i in range(10)]
        latin_to_id = {c: 9366 + idx for idx, c in enumerate(latin_chars)}
        id_to_latin = {v: k for k, v in latin_to_id.items()}
        trans_token_id = 9410
        eos_token_id = 2

        # Activate attention governance
        for layer in self.backbone.model.layers:
            layer.self_attn.coverage_weight.data.fill_(cov_w)
            layer.self_attn.governance_strength.data.fill_(gov_s)
            layer.self_attn.reset_coverage()

        ar_ids = self.tokenizer.encode(arabic_text)
        prompt_ids = ar_ids + [trans_token_id]
        cur = list(prompt_ids)
        emitted = []

        for step in range(max_chars):
            cur_t = torch.tensor([cur], dtype=torch.long, device=self.device)
            r_ids, w_ids = self.extract_morphemic_ids(cur_t)
            for layer in self.backbone.model.layers:
                layer.self_attn.active_root_ids = r_ids
                layer.self_attn.active_wazn_ids = w_ids

            out = self.backbone(cur_t, return_dict=True)
            logits = out.logits[0, -1, :].clone()

            # Constrain to Latin SIMD slots + EOS
            allowed = torch.zeros_like(logits, dtype=torch.bool)
            for tid in list(latin_to_id.values()) + [eos_token_id]:
                allowed[tid] = True
            logits[~allowed] = -float('inf')

            # 1. Order-4 N-Gram Word Blocker
            text_so_far = ''.join(emitted)
            words = text_so_far.split()
            if len(words) >= 4:
                trigram = tuple(words[-3:])
                all_trigrams = [tuple(words[i:i+3]) for i in range(len(words)-3)]
                if trigram in all_trigrams:
                    idx = all_trigrams.index(trigram)
                    if idx + 3 < len(words):
                        blocked_next_word = words[idx + 3]
                        if blocked_next_word and blocked_next_word[0] in latin_to_id:
                            logits[latin_to_id[blocked_next_word[0]]] -= 12.0

            # 2. Repetition Penalty on recent characters
            recent_chars = emitted[-64:]
            char_counts = {}
            for c in recent_chars:
                char_counts[c] = char_counts.get(c, 0) + 1
            for c, count in char_counts.items():
                if c in latin_to_id and count > 1:
                    logits[latin_to_id[c]] -= (count - 1) * alpha_rep

            # 3. Dynamic EOS Boost
            if len(words) >= 12:
                progress = min(1.0, (len(words) - 12) / 10.0)
                logits[eos_token_id] += beta_eos * progress

            next_id = torch.argmax(logits).item()
            if next_id == eos_token_id:
                break
            if next_id in id_to_latin:
                emitted.append(id_to_latin[next_id])
            cur.append(next_id)

        elapsed = time.time() - t0
        english_text = ''.join(emitted).strip()

        return {
            'arabic_input': arabic_text,
            'english_translation': english_text,
            'tokens_generated': len(emitted),
            'elapsed_seconds': elapsed,
            'chars_per_sec': len(emitted) / max(elapsed, 1e-4)
        }

