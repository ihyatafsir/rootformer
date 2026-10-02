#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rootformer_v18_nrmp_model.py
The Farāhīdian Next-Root/Morph Prediction (NRMP) Foundation Model.
Evolves out of Next-Token Prediction (NTP) to word-level morphological factorization:
P(Word_{t+1}) = P(Root_{t+1}) * P(Wazn_{t+1} | Root) * P(Affixes_{t+1} | Root, Wazn)
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Any


# ---------------------------------------------------------------------------------------
# VISIBILITY (pillar 3).  This model REPORTS its grammar: `governance`/`andalusian` come out
# of the decode call.  Every handler below used to swallow its failure, so a broken governor
# or realizer produced an EMPTY report that a caller reads as "no violations".  Each handler
# keeps its previous default and reports the first failure of its site on stderr.
# ---------------------------------------------------------------------------------------
_WARNED_SITES = set()


def _warn_once(site, message):
    if site not in _WARNED_SITES:
        _WARNED_SITES.add(site)
        print(f'[rootformer_v18_nrmp_model] {message}', file=sys.stderr)

# Sibawayh's valency stack and his persistent ʿāmil, as one object the decoder calls.
# constituent_stack.ConstituentStack is the (already written, previously uncalled) valency
# stack: analyze(words) -> [(category, case, role, reason)], 88.5% role coverage.
try:
    from constituent_stack import ConstituentStack
except Exception as exc:
    # VISIBILITY: both modules ship with this release, so a failed import is a broken install --
    # and the decode path would silently lose the valency stack / persistent ʿāmil entirely.
    # Defaults None are unchanged (the decode code None-checks them).
    print(f'[rootformer_v18_nrmp_model] constituent_stack import failed: {exc!r} -- the valency '
          f'stack is DISABLED for this process', file=sys.stderr)
    ConstituentStack = None
try:
    from sibawayh_governor import SibawayhGovernor
except Exception as exc:
    # VISIBILITY: as above -- without the governor the decode reports no ʿamal at all.
    print(f'[rootformer_v18_nrmp_model] sibawayh_governor import failed: {exc!r} -- the '
          f'persistent ʿāmil is DISABLED for this process', file=sys.stderr)
    SibawayhGovernor = None

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
        except Exception as exc:
            # VISIBILITY: transmuter=None silently drops transmuted_english from every decode.
            print(f'[rootformer_v18_nrmp_model] English transmuter unavailable: {exc!r} -- '
                  f'transmuted_english will be empty', file=sys.stderr)
            self.transmuter = None

        # Sībawayh & Al-Khalīl Hard Grammatical Exclusion Engine
        try:
            from sibawayh_governance_engine import SibawayhNRMPGovernance
            self.sibawayh_gov = SibawayhNRMPGovernance(self.vocab)
        except Exception as exc:
            # VISIBILITY: the hard exclusion mask is not applied when this is None; the decode
            # still returns a sentence, so the missing constraint must not be silent.
            print(f'[rootformer_v18_nrmp_model] SibawayhNRMPGovernance unavailable: {exc!r} -- '
                  f'the hard exclusion mask is NOT applied', file=sys.stderr)
            self.sibawayh_gov = None

        # Sībawayh's constituent/valency stack (the ʿāmil's demands, walked word by word)
        # and the governor that keeps the ʿāmil in force across an intervening noun.
        #   al-Kitab 1/421: «فصار النعت مجرورا مثل المنعوت لأنهما كالاسم الواحد»
        self.constituent_stack = None
        if ConstituentStack is not None:
            try:
                self.constituent_stack = ConstituentStack(
                    self.vocab, getattr(self.vocab, 'blueprint_path', None))
            except Exception as exc:
                # VISIBILITY: a construction failure silently removes the valency stack from the
                # decode path, so the governance report comes back empty.  Default None unchanged.
                print(f'[rootformer_v18_nrmp_model] ConstituentStack construction failed: '
                      f'{exc!r} -- valency roles will not be reported', file=sys.stderr)
                self.constituent_stack = None
        blueprint = getattr(self.vocab, 'blueprint_path', None)
        self.sibawayh_governor = None
        if SibawayhGovernor is not None:
            try:
                self.sibawayh_governor = SibawayhGovernor(
                    self.vocab, self.constituent_stack, blueprint)
            except Exception as exc:
                # VISIBILITY: as above -- no governor means no persistent ʿāmil in the report.
                print(f'[rootformer_v18_nrmp_model] SibawayhGovernor construction failed: '
                      f'{exc!r} -- the persistent ʿāmil will not be reported', file=sys.stderr)
                self.sibawayh_governor = None

        # The Andalusian realisation layer: i'rab + the diptote (Ibn 'Usfur), al-iktifa'
        # (clitic saturation), the jussive apocope and al-Shatibi's waw, wrapped once in
        # andalusian_realizer.AndalusianRealizer.  install() only attaches the object to
        # the vocab as `vocab.andalusian_realizer`; no existing behaviour is touched.
        self.andalusian_realizer = None
        try:
            from andalusian_realizer import AndalusianRealizer
            self.andalusian_realizer = AndalusianRealizer(self.vocab).install(self.vocab)
        except Exception as exc:
            # VISIBILITY: this layer is additive to the output; if it fails to install, the model
            # silently stops reporting iʿrāb.  Default None is unchanged.
            print(f'[rootformer_v18_nrmp_model] AndalusianRealizer install failed: {exc!r} -- '
                  f'the andalusian report will be empty', file=sys.stderr)
            self.andalusian_realizer = None

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
        
        # ---- Sībawayh's ʿāmil over the PROMPT ------------------------------------------
        # The operator chain is seeded once and then advanced by every generated word, so
        # the jarr / naṣb / jazm of a word does not die after one step (al-Kitab 1/421).
        gov = self.sibawayh_governor
        if gov is not None:
            gov.reset()
            prompt_words = prompt_text.split()
            for i, wd in enumerate(prompt_words):
                gov.observe_word(wd, prompt_words[i + 1] if i + 1 < len(prompt_words) else None)
        else:
            prompt_words = []
        # the valency stack is run over the prompt too (real call, decode-time role seed)
        prompt_valency = []
        if self.constituent_stack is not None and prompt_words:
            try:
                prompt_valency = self.constituent_stack.analyze(prompt_words)
            except Exception as exc:
                # VISIBILITY: the prompt's valency seed silently became [] -- the decode proceeds
                # with no role seed and reports as if the stack had agreed.  Default [] unchanged.
                _warn_once('prompt_valency',
                           f'constituent_stack.analyze(prompt) raised: {exc!r} -- the prompt '
                           f'valency seed is empty')
                prompt_valency = []

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
                
                # Sībawayh Operator Tracking -- PERSISTENT, not recomputed from t-1
                prev_tuple = (cur_p[-1], cur_r[-1], cur_w[-1], cur_s[-1])
                coord_between = False
                if gov is not None:
                    op_state = gov.state_for_next()          # the ʿāmil in force (al-Kitab 1/421)
                elif self.sibawayh_gov:
                    op_state = self.sibawayh_gov.get_operator_state(prev_tuple)
                else:
                    op_state = "NONE"
                # the previous word's own operator, for the class-transition test below
                prev_class = (gov.class_of_tuple(prev_tuple) if gov is not None
                              else 'NONE')
                # the ʿaṭf coordinator (و/ف) is carried in the prefix slot (Alfiyyah:
                # «وعطفك الفعل على الفعل يصح» is admitted only through it)
                coord_between = bool(
                    prev_class == 'FIL' and gov is not None
                    and gov.coordinator_from_prefix(
                        self.vocab.id2prefix.get(cur_p[-1], '')))
                
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
                    if gov is not None:
                        # DECODE-TIME TRANSITION TABLE (Ibn Malik's POS automaton under
                        # Sibawayh's classes): a wazn whose class the previous class cannot
                        # reach is masked.  Fiʿl -> Fiʿl survives exactly when a coordinator
                        # intervened (Alfiyyah, bāb al-ʿaṭf).
                        w_logits = gov.mask_wazn_logits(w_logits, prev_class, coord_between)
                    if not self.sibawayh_gov and gov is None:
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
                if gov is not None:
                    # advance the ʿāmil chain with the word just generated, so its government
                    # carries to the next step instead of dying (al-Kitab 1/421)
                    gov.observe_tuple((next_p, next_r, next_w, next_s))
                
        # Two-Tier Farāhīdian Realization:
        # Tier 1 (Al-Ṣarf: Morphological Derivation) + Tier 2 (Al-Naḥw: Basran Syntactic Governance)
        surface_text = self.vocab.decode_sentence(generated_tuples, apply_syntax=True)
        
        # Dual-Actuator English Transmutation
        transmuted_english = ""
        if self.transmuter:
            try:
                transmuted_english = self.transmuter.transmute_tuples(generated_tuples)
            except Exception as exc:
                # VISIBILITY: an empty transmutation looks exactly like a successful one that
                # produced nothing.  Default "" is unchanged.
                _warn_once('transmute',
                           f'transmute_tuples raised: {exc!r} -- transmuted_english is empty')
                transmuted_english = ""

        # ---- the constituent/valency stack over the generated hypothesis --------------
        # (Sibawayh's ʿamal: the case comes from the persistent operator, the role from
        # valency x morphology.)  Exposed so a caller can audit the decode, and so the stack
        # is genuinely on the decode path rather than in an uncalled module.
        governance = []
        if self.sibawayh_governor is not None and generated_tuples:
            try:
                governance = self.sibawayh_governor.analyze(generated_tuples)
            except Exception as exc:
                # VISIBILITY: `governance: []` is precisely how "this sentence violates nothing"
                # is reported, so a governor crash was indistinguishable from a clean decode.
                _warn_once('governance',
                           f'sibawayh_governor.analyze raised: {exc!r} -- the governance report '
                           f'is EMPTY, not clean')
                governance = []
        elif self.constituent_stack is not None and generated_tuples:
            try:
                words = [self.vocab.decode_word(*t) for t in generated_tuples]
                governance = self.constituent_stack.analyze(words)
            except Exception as exc:
                # VISIBILITY: as above -- an empty report used to stand in for a clean one.
                _warn_once('governance_stack',
                           f'constituent_stack.analyze raised on the generated hypothesis: '
                           f'{exc!r} -- the governance report is EMPTY, not clean')
                governance = []

        # ---- ANDALUSIAN REALISATION (the decode hook) ----------------------------------
        # One call per generated word: the model's (prefix, root, wazn, suffix) event is
        # turned into its i'rab-realised surface by the cited Andalusian rules -- Ibn
        # 'Usfur's diptote, Sibawayh's jussive apocope, al-iktifa' (which valency slot the
        # clitic saturates, and whether that closed the host's slot) and al-Shatibi's waw.
        # The case comes from the governor's persistent 'amil (governance[i][1]); the hook
        # is ADDITIVE -- it writes the new keys below and changes no existing output.
        andalusian = []
        realizer = (self.andalusian_realizer
                    or getattr(self.vocab, 'andalusian_realizer', None))
        if realizer is not None and generated_tuples:
            for i, t in enumerate(generated_tuples):
                case = None
                if i < len(governance):
                    g = governance[i]
                    if isinstance(g, (tuple, list)) and len(g) > 1:
                        case = g[1]
                    else:
                        case = getattr(g, 'case', None)
                try:
                    andalusian.append(realizer.analyze_tuple(
                        self.vocab, t[0], t[1], t[2], t[3], case=case))
                except Exception as exc:
                    # VISIBILITY: a None entry used to be appended silently, so a systematic
                    # realizer failure produced a report full of holes that reads as "no iʿrāb
                    # applicable".  The None entry is unchanged.
                    _warn_once('andalusian',
                               f'andalusian_realizer.analyze_tuple raised: {exc!r} -- None '
                               f'entries are being reported for the affected words')
                    andalusian.append(None)

        return {
            "generated_text": surface_text,
            "transmuted_english": transmuted_english,
            "generated_tuples": generated_tuples,
            "num_words": len(generated_tuples),
            "governance": governance,
            "prompt_governance": prompt_valency,
            "valency_stack_used": bool(self.constituent_stack is not None
                                       or self.sibawayh_governor is not None),
            "andalusian": andalusian,
            "andalusian_realizer_used": bool(andalusian),
            "realised_text": ' '.join(
                (a.get('irab') or a.get('surface') or '') for a in andalusian if a),
        }

