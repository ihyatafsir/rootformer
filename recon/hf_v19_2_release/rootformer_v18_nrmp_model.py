#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rootformer_v18_nrmp_model.py
The Farāhīdian Next-Root/Morph Prediction (NRMP) Foundation Model.
Evolves out of Next-Token Prediction (NTP) to word-level morphological factorization:
P(Word_{t+1}) = P(Root_{t+1}) * P(Wazn_{t+1} | Root) * P(Affixes_{t+1} | Root, Wazn)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Any

class FarahidianWordEmbedding(nn.Module):
    """
    Multi-Component Morphological Embedding:
    e(W) = [E_root(r), E_wazn(w), E_prefix(p), E_suffix(s)]
    """
    def __init__(self, num_roots: int, num_awzan: int, num_prefixes: int, num_suffixes: int, d_model: int = 896):
        super().__init__()
        self.d_model = d_model
        self.d_root = d_model // 2         # 448
        self.d_wazn = d_model // 4         # 224
        self.d_prefix = d_model // 8       # 112
        self.d_suffix = d_model - (self.d_root + self.d_wazn + self.d_prefix) # 112
        
        self.root_embed = nn.Embedding(num_roots, self.d_root)
        self.wazn_embed = nn.Embedding(num_awzan, self.d_wazn)
        self.prefix_embed = nn.Embedding(num_prefixes, self.d_prefix)
        self.suffix_embed = nn.Embedding(num_suffixes, self.d_suffix)
        
        # Scaling factor
        self.scale = math.sqrt(d_model)

    def forward(self, p_ids: torch.Tensor, r_ids: torch.Tensor, w_ids: torch.Tensor, s_ids: torch.Tensor) -> torch.Tensor:
        """
        Input: [Batch, SeqLen] integer tensors
        Output: [Batch, SeqLen, d_model] continuous word representation
        """
        e_r = self.root_embed(r_ids)
        e_w = self.wazn_embed(w_ids)
        e_p = self.prefix_embed(p_ids)
        e_s = self.suffix_embed(s_ids)
        
        # Concatenate into full model dimension
        word_emb = torch.cat([e_r, e_w, e_p, e_s], dim=-1) * self.scale
        return word_emb

class FarahidianNRMPHead(nn.Module):
    """
    Hierarchical Factorized Prediction Head:
    1. Root Head: P(R_{t+1} | Context)
    2. Wazn Head: P(W_{t+1} | R_{t+1}, Context)
    3. Prefix & Suffix Heads: P(P_{t+1}, S_{t+1} | R_{t+1}, Context)
    """
    def __init__(self, d_model: int, num_roots: int, num_awzan: int, num_prefixes: int, num_suffixes: int, d_root: int):
        super().__init__()
        self.d_model = d_model
        self.d_root = d_root
        
        # 1. Primary Ontological Root Head
        self.root_head = nn.Linear(d_model, num_roots, bias=False)
        
        # 2. Conditioned Morphological Projector
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model + d_root, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 3. Wazn Head
        self.wazn_head = nn.Linear(d_model, num_awzan, bias=False)
        
        # 4. Affix Heads
        self.prefix_head = nn.Linear(d_model, num_prefixes, bias=False)
        self.suffix_head = nn.Linear(d_model, num_suffixes, bias=False)

    def forward(self, h: torch.Tensor, root_embed_layer: nn.Embedding, target_roots: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        h: [Batch, SeqLen, d_model] -> Backbone hidden states
        root_embed_layer: embedding table to project chosen roots
        target_roots: ground truth roots during training [Batch, SeqLen]
        """
        # Step 1: Root Prediction (Ontological Substance)
        root_logits = self.root_head(h) # [B, T, num_roots]
        
        # Determine root representation for conditioning
        if self.training and target_roots is not None:
            chosen_root_ids = target_roots
        else:
            chosen_root_ids = torch.argmax(root_logits, dim=-1)
            
        e_chosen_root = root_embed_layer(chosen_root_ids) # [B, T, d_root]
        
        # Step 2: Morphological Conditioning
        cond_input = torch.cat([h, e_chosen_root], dim=-1) # [B, T, d_model + d_root]
        h_cond = self.cond_proj(cond_input) # [B, T, d_model]
        
        # Step 3: Template & Affix Prediction
        wazn_logits = self.wazn_head(h_cond)       # [B, T, num_awzan]
        prefix_logits = self.prefix_head(h_cond)   # [B, T, num_prefixes]
        suffix_logits = self.suffix_head(h_cond)   # [B, T, num_suffixes]
        
        return {
            "root_logits": root_logits,
            "wazn_logits": wazn_logits,
            "prefix_logits": prefix_logits,
            "suffix_logits": suffix_logits
        }

class RootformerV18_NRMP(nn.Module):
    """
    Rootformer v18: The Farāhīdian Next-Root/Morph Prediction Foundation Model.
    Employs the 24-layer DeepSeek-V4.1-Flash backbone with IshtiqaqAttention
    and factorized morphemic generation heads.
    """
    def __init__(self, base_v17_model, vocab, device: str = 'cuda', dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.vocab = vocab
        
        # Reference backbone from v17
        if hasattr(base_v17_model, 'model') and hasattr(base_v17_model.model, 'backbone'):
            self.backbone = base_v17_model.model.backbone.model
        elif hasattr(base_v17_model, 'backbone'):
            self.backbone = base_v17_model.backbone.model if hasattr(base_v17_model.backbone, 'model') else base_v17_model.backbone
        else:
            self.backbone = base_v17_model
            
        self.d_model = getattr(self.backbone.config, 'hidden_size', 896)
        
        # 1. Morphological Embedding Layer
        self.morphemic_embed = FarahidianWordEmbedding(
            num_roots=vocab.num_roots,
            num_awzan=vocab.num_awzan,
            num_prefixes=vocab.num_prefixes,
            num_suffixes=vocab.num_suffixes,
            d_model=self.d_model
        ).to(device=device, dtype=dtype)
        
        # 2. Factorized Prediction Head
        self.nrmp_head = FarahidianNRMPHead(
            d_model=self.d_model,
            num_roots=vocab.num_roots,
            num_awzan=vocab.num_awzan,
            num_prefixes=vocab.num_prefixes,
            num_suffixes=vocab.num_suffixes,
            d_root=self.morphemic_embed.d_root
        ).to(device=device, dtype=dtype)
        
        # Final LayerNorm
        self.final_norm = nn.LayerNorm(self.d_model).to(device=device, dtype=dtype)
        
        # Dual-Actuator English Transmutation Engine
        try:
            from farahidian_transmutation_engine import FarahidianEnglishTransmuter
            self.transmuter = FarahidianEnglishTransmuter(self.vocab)
        except Exception:
            self.transmuter = None

        # Sībawayh & Al-Khalīl Hard Grammatical Exclusion Engine
        try:
            from sibawayh_governance_engine import SibawayhNRMPGovernance
            self.sibawayh_gov = SibawayhNRMPGovernance(self.vocab)
        except Exception:
            self.sibawayh_gov = None

    def forward(
        self,
        p_ids: torch.Tensor,
        r_ids: torch.Tensor,
        w_ids: torch.Tensor,
        s_ids: torch.Tensor,
        target_roots: Optional[torch.Tensor] = None,
        target_awzan: Optional[torch.Tensor] = None,
        target_prefixes: Optional[torch.Tensor] = None,
        target_suffixes: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass over morphemic word-event sequence.
        """
        # Step 1: Composite Morphological Embedding
        inputs_embeds = self.morphemic_embed(p_ids, r_ids, w_ids, s_ids)
        
        # Step 2: 24-Layer Transformer Backbone
        outputs = self.backbone(inputs_embeds=inputs_embeds)
        hidden_states = self.final_norm(outputs.last_hidden_state)
        
        # Step 3: Factorized NRMP Prediction Heads
        head_outputs = self.nrmp_head(
            h=hidden_states,
            root_embed_layer=self.morphemic_embed.root_embed,
            target_roots=target_roots
        )
        
        loss = None
        loss_dict = {}
        if target_roots is not None and target_awzan is not None:
            # Shift predictions and targets for Next-Word Event Prediction
            shift_root_logits = head_outputs['root_logits'][:, :-1, :].contiguous()
            shift_wazn_logits = head_outputs['wazn_logits'][:, :-1, :].contiguous()
            shift_prefix_logits = head_outputs['prefix_logits'][:, :-1, :].contiguous()
            shift_suffix_logits = head_outputs['suffix_logits'][:, :-1, :].contiguous()
            
            shift_target_roots = target_roots[:, 1:].contiguous()
            shift_target_awzan = target_awzan[:, 1:].contiguous()
            shift_target_prefixes = target_prefixes[:, 1:].contiguous()
            shift_target_suffixes = target_suffixes[:, 1:].contiguous()
            
            # Mask out PAD tokens (PAD_ROOT = 0)
            pad_mask = (shift_target_roots != self.vocab.PAD_ROOT)
            
            loss_root = F.cross_entropy(
                shift_root_logits.view(-1, self.vocab.num_roots),
                shift_target_roots.view(-1),
                ignore_index=self.vocab.PAD_ROOT
            )
            
            loss_wazn = F.cross_entropy(
                shift_wazn_logits.view(-1, self.vocab.num_awzan),
                shift_target_awzan.view(-1),
                ignore_index=self.vocab.PAD_WAZN
            )
            
            loss_prefix = F.cross_entropy(
                shift_prefix_logits.view(-1, self.vocab.num_prefixes),
                shift_target_prefixes.view(-1),
                ignore_index=self.vocab.PAD_PREFIX
            )
            
            loss_suffix = F.cross_entropy(
                shift_suffix_logits.view(-1, self.vocab.num_suffixes),
                shift_target_suffixes.view(-1),
                ignore_index=self.vocab.PAD_SUFFIX
            )
            
            # Weighted Farāhīdian Multi-Task Objective
            loss = loss_root + 0.5 * loss_wazn + 0.25 * loss_prefix + 0.25 * loss_suffix
            loss_dict = {
                "total_loss": loss.item(),
                "loss_root": loss_root.item(),
                "loss_wazn": loss_wazn.item(),
                "loss_prefix": loss_prefix.item(),
                "loss_suffix": loss_suffix.item(),
                "root_ppl": math.exp(min(loss_root.item(), 20.0))
            }
            
        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "root_logits": head_outputs['root_logits'],
            "wazn_logits": head_outputs['wazn_logits'],
            "prefix_logits": head_outputs['prefix_logits'],
            "suffix_logits": head_outputs['suffix_logits'],
            "hidden_states": hidden_states
        }

    def generate_words(self, prompt_text: str, max_new_words: int = 20, min_new_words: int = 5) -> Dict[str, Any]:
        """
        Autoregressively generates next words by predicting (Root, Wazn, Prefix, Suffix)
        and deterministically realizing surface Arabic text in 1 forward pass per word!
        """
        self.eval()
        encoded = self.vocab.encode_sentence(prompt_text)
        if not encoded:
            encoded = [(self.vocab.NONE_PREFIX, self.vocab.BOS_ROOT, self.vocab.NONE_WAZN, self.vocab.NONE_SUFFIX)]
            
        cur_p = [t[0] for t in encoded]
        cur_r = [t[1] for t in encoded]
        cur_w = [t[2] for t in encoded]
        cur_s = [t[3] for t in encoded]
        
        generated_tuples = []
        with torch.no_grad():
            for step in range(max_new_words):
                in_p = torch.tensor([cur_p], dtype=torch.long, device=self.device)
                in_r = torch.tensor([cur_r], dtype=torch.long, device=self.device)
                in_w = torch.tensor([cur_w], dtype=torch.long, device=self.device)
                in_s = torch.tensor([cur_s], dtype=torch.long, device=self.device)
                
                inputs_embeds = self.morphemic_embed(in_p, in_r, in_w, in_s)
                outputs = self.backbone(inputs_embeds=inputs_embeds)
                h = self.final_norm(outputs.last_hidden_state)[:, -1:, :]
                
                # Sībawayh Operator Tracking
                prev_tuple = (cur_p[-1], cur_r[-1], cur_w[-1], cur_s[-1])
                op_state = self.sibawayh_gov.get_operator_state(prev_tuple) if self.sibawayh_gov else "NONE"
                
                # 1. Root Prediction: P(Root_{t+1} | Context) with Sibawayh & Al-Khalīl exclusion
                r_logits = self.nrmp_head.root_head(h)[0, 0, :].clone()
                if self.sibawayh_gov:
                    r_logits = self.sibawayh_gov.apply_root_exclusion_mask(
                        r_logits, op_state, cur_r[-6:], step, min_new_words
                    )
                else:
                    r_logits[self.vocab.PAD_ROOT] = -1e9
                    r_logits[self.vocab.BOS_ROOT] = -1e9
                    r_logits[self.vocab.root2id['<UNK>']] = -1e9
                    if step < min_new_words:
                        r_logits[self.vocab.EOS_ROOT] = -1e9
                    recent_roots = cur_r[-6:]
                    for r_item in set(recent_roots):
                        count = recent_roots.count(r_item)
                        r_logits[r_item] -= count * 2.5
                        
                next_r = torch.argmax(r_logits).item()
                if next_r == self.vocab.EOS_ROOT:
                    break
                    
                root_str = self.vocab.id2root.get(next_r, "")
                is_particle = root_str.startswith('<P:') or root_str in ['<PARTICLE>', '<UNK>', '<PAD>']
                
                # 2. Causal Morphological Conditioning on Chosen Root R_{t+1}
                r_tensor = torch.tensor([[next_r]], dtype=torch.long, device=self.device)
                e_chosen_root = self.morphemic_embed.root_embed(r_tensor)
                cond_input = torch.cat([h, e_chosen_root], dim=-1)
                h_cond = self.nrmp_head.cond_proj(cond_input)
                
                # 3. Wazn / Template Prediction: P(Wazn_{t+1} | Root_{t+1}, Context)
                if is_particle:
                    next_w = self.vocab.NONE_WAZN
                else:
                    w_logits = self.nrmp_head.wazn_head(h_cond)[0, 0, :].clone()
                    if self.sibawayh_gov:
                        w_logits = self.sibawayh_gov.apply_wazn_exclusion_mask(
                            w_logits, op_state, self.vocab.NONE_PREFIX, is_radical_root=True
                        )
                    else:
                        w_logits[self.vocab.NONE_WAZN] = -1e9
                        w_logits[self.vocab.PAD_WAZN] = -1e9
                        w_logits[self.vocab.wazn2id['<UNK>']] = -1e9
                    next_w = torch.argmax(w_logits).item()
                    
                # 4. Affix Prediction: P(Prefix, Suffix | Root_{t+1}, Wazn_{t+1}, Context)
                p_logits = self.nrmp_head.prefix_head(h_cond)[0, 0, :].clone()
                if self.sibawayh_gov:
                    p_logits = self.sibawayh_gov.apply_prefix_exclusion_mask(p_logits, op_state)
                else:
                    p_logits[self.vocab.PAD_PREFIX] = -1e9
                next_p = torch.argmax(p_logits).item()
                
                s_logits = self.nrmp_head.suffix_head(h_cond)[0, 0, :].clone()
                if self.sibawayh_gov:
                    s_logits = self.sibawayh_gov.apply_suffix_exclusion_mask(s_logits, next_w)
                else:
                    s_logits[self.vocab.PAD_SUFFIX] = -1e9
                next_s = torch.argmax(s_logits).item()
                
                cur_p.append(next_p)
                cur_r.append(next_r)
                cur_w.append(next_w)
                cur_s.append(next_s)
                generated_tuples.append((next_p, next_r, next_w, next_s))
                
        # Two-Tier Farāhīdian Realization:
        # Tier 1 (Al-Ṣarf: Morphological Derivation) + Tier 2 (Al-Naḥw: Basran Syntactic Governance)
        surface_text = self.vocab.decode_sentence(generated_tuples, apply_syntax=True)
        
        # Dual-Actuator English Transmutation
        transmuted_english = ""
        if self.transmuter:
            try:
                transmuted_english = self.transmuter.transmute_tuples(generated_tuples)
            except Exception:
                transmuted_english = ""

        return {
            "generated_text": surface_text,
            "transmuted_english": transmuted_english,
            "generated_tuples": generated_tuples,
            "num_words": len(generated_tuples)
        }

