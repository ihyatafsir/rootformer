#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_rootformer_v13.py -- the CORRECTED transmutation target.

`models/unified_rootformer_v12.py` is READ-ONLY and is not edited: this file is generated as a
corrected sibling so the diff is auditable and the shipped file keeps its md5.

THE DIFF vs unified_rootformer_v12.py (md5 68646a897e7037e5219dfd0463fdcc3d)
--------------------------------------------------------------------------
  v12 line 26   DisjointMorphologicalHeadsV12(num_roots=9015, num_awzan=128)
  v13           num_roots=LIVE_ROOTS (9490), num_awzan=LIVE_AWZAN (142)
  v12 line 51   self.vocab_size = self.tokenizer.vocab_size  # 9,856   (comment AND value stale)
  v13           asserted == LIVE_VOCAB (10052)
  v12 line 80   IshtiqaqAttentionV12(num_roots=9015)
  v13           num_roots=LIVE_ROOTS
  v12 line 81   num_awzan=128
  v13           num_awzan=LIVE_AWZAN
  v12 line 91   DisjointMorphologicalHeadsV12(num_roots=9015)
  v13           num_roots=LIVE_ROOTS
  v12 line 92   num_awzan=128
  v13           num_awzan=LIVE_AWZAN
  v12 line 109  root_part = token_id - 349 ; clamp(1, 9014)          <- offset arithmetic
  v13           build_token_root_table(vocab)  -- by String through the live root2id, with an
                injectivity assertion
  v12 line 121  wazn_ids = torch.zeros_like(safe_inputs)             <- DEAD STUB
  v13           build_token_wazn_table(vocab)  -- the real wazn id
  v12 line 172  root_logits.view(-1, 9015)
  v13           root_logits.view(-1, LIVE_ROOTS)
  v12 line 180  wazn_logits.view(-1, 128)
  v13           wazn_logits.view(-1, LIVE_AWZAN)
  v12 line 277  AutoModelForCausalLM.from_pretrained(base)  <- needs the NETWORK
  v13           transmute_from_clean_qwen2_5() loads the SEALED LOCAL config+weights, resizes
                embed_tokens/lm_head 151936 -> 10052, and asserts all 290 real Qwen tensors.

TWO ROOT SOURCES, ON PURPOSE
----------------------------
  * the per-layer `IshtiqaqAttentionV12.root_embed` is fixed to 9490x64 so it AGREES with the
    live space (no 475 missing rows, no 476-way alias).  It stays DEAD in the NRMT forward unless
    the score-bias pathway is attached (proved in liveness_report.json).
  * every NEW pathway reads the RELEASED, synthesis-trained, 448-dim
    `morphemic_embed.root_embed` (9490 x 448).  The stale 9015x64 table is never used as a
    source: 475 ids alias onto row 9014 there.
"""
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from root_space import (LIVE_ROOTS, LIVE_AWZAN, LIVE_VOCAB, SpaceViolation,
                        assert_one_root_space, assert_no_aliasing,
                        build_token_root_table, build_token_wazn_table)

from models.ishtiqaq_attention_v12 import IshtiqaqAttentionV12
from models.morphemic_tokenizer_v12_arabic import PureArabicMorphemicTokenizerV12


def find_qwen_weights(candidates=None) -> Optional[str]:
    """The largest real (non-symlink-resolved-to-nothing) file in the Qwen blob dirs.

    HF content-addressed blobs carry no suffix, so SIZE is the discriminator: the weights are
    ~988 MB, the config blob is 681 B.  A dangling symlink is skipped by os.path.isfile().
    """
    import glob
    pats = candidates or [
        '/workspace/alt_base/cache/hub/models--Qwen--Qwen2.5-0.5B/blobs/*',
        '/workspace/.hf_home/hub/models--Qwen--Qwen2.5-0.5B/blobs/*',
        str(Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/blobs/*'),
    ]
    hits = []
    for pat in pats:
        for f in glob.glob(pat):
            try:
                if os.path.isfile(f) and os.path.getsize(f) > 500_000_000:
                    hits.append((os.path.getsize(f), f))
            except OSError:
                continue
    if not hits:
        return None
    hits.sort(reverse=True)
    return hits[0][1]


class DisjointMorphologicalHeadsV13(nn.Module):
    def __init__(self, hidden_size: int = 896, num_roots: int = LIVE_ROOTS,
                 num_awzan: int = LIVE_AWZAN, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        if (num_roots, num_awzan) != (LIVE_ROOTS, LIVE_AWZAN):
            raise SpaceViolation(f'num_roots/num_awzan must be ({LIVE_ROOTS},{LIVE_AWZAN}), '
                                 f'got ({num_roots},{num_awzan})')
        self.root_head = nn.Linear(hidden_size, num_roots, bias=False, dtype=dtype)
        self.wazn_head = nn.Linear(hidden_size, num_awzan, bias=False, dtype=dtype)

    def forward(self, morph_hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.root_head(morph_hidden_state), self.wazn_head(morph_hidden_state)


class UnifiedRootformerV13(nn.Module):
    """The corrected transmutation target.  Only the constants, the two id tables and the
    transmutation differ from v12; the module set is identical so a released checkpoint loads
    with exactly the stale 48 keys dropped (asserted)."""

    def __init__(
        self,
        blueprint_path: Union[str, Path] = '/workspace/rootformer_v12/data/rootformer_v12_arabic_blueprint.json',
        base_model_name: str = 'Qwen/Qwen2.5-0.5B',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available()
        and torch.cuda.is_bf16_supported() else torch.float32,
        is_training: bool = False,
        vocab=None,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.blueprint_path = Path(blueprint_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.is_training = is_training

        # 1. Pure Arabic Tokenizer & Vocab
        self.tokenizer = PureArabicMorphemicTokenizerV12(self.blueprint_path)
        self.vocab_size = int(self.tokenizer.vocab_size)
        if self.vocab_size != LIVE_VOCAB:
            raise SpaceViolation(f'tokenizer vocab_size={self.vocab_size}, spec={LIVE_VOCAB}')

        # 1b. the live NRMP vocab (root/awzan id space).  Built here so the two id tables below
        # are derived from the SAME object the trainer uses, not from a literal.
        if vocab is None:
            import nrmp_vocab as _nv
            _V = next(v for k, v in vars(_nv).items()
                      if isinstance(v, type) and 'MorphemicVocab' in k)
            vocab = _V(str(self.blueprint_path))
        self.vocab = vocab
        if (int(vocab.num_roots), int(vocab.num_awzan)) != (LIVE_ROOTS, LIVE_AWZAN):
            raise SpaceViolation(f'live vocab is ({vocab.num_roots},{vocab.num_awzan}), spec is '
                                 f'({LIVE_ROOTS},{LIVE_AWZAN}) -- refusing to transmute')

        # 2. Config & Backbone
        print(f'[Rootformer v13] Initializing Qwen2.5-0.5B config from {base_model_name}...',
              flush=True)
        try:
            from transformers import AutoConfig
            self.config = AutoConfig.from_pretrained(base_model_name,
                                                     local_files_only=local_files_only)
        except Exception as exc:
            raise RuntimeError(
                f'cannot read the sealed base config for {base_model_name!r} with '
                f'local_files_only={local_files_only}: {type(exc).__name__}: {exc}') from exc
        self.config.vocab_size = self.vocab_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.morph_layer_idx = 14

        # 3. Backbone with IshtiqaqAttention across all 24 layers, IN THE LIVE SPACE
        print('[Rootformer v13] Instantiating backbone with IshtiqaqAttention in the LIVE root '
              f'space (roots={LIVE_ROOTS}, awzan={LIVE_AWZAN})...', flush=True)
        from transformers import AutoModelForCausalLM
        self.backbone = AutoModelForCausalLM.from_config(self.config, dtype=self.dtype)

        for layer_idx, layer in enumerate(self.backbone.model.layers):
            layer.self_attn = IshtiqaqAttentionV12(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                num_roots=LIVE_ROOTS,          # was 9015
                num_awzan=LIVE_AWZAN,          # was 128
                ishtiqaq_init_strength=0.25,
                dropout=0.0,
                layer_idx=layer_idx,
                dtype=self.dtype
            )

        # 4. Decoupled Morphological Auxiliary Heads at Layer 14
        self.morphemic_heads = DisjointMorphologicalHeadsV13(
            hidden_size=self.hidden_size, num_roots=LIVE_ROOTS, num_awzan=LIVE_AWZAN,
            dtype=self.dtype)

        # 5. Metadata Mask for Text Generation
        self.metadata_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        for token_id in range(self.vocab_size):
            token_str = self.tokenizer.id_to_token.get(token_id, '')
            if token_str.startswith('<wazn_') or token_str.startswith('<root_') \
                    or token_str.startswith('<extra_id_'):
                self.metadata_mask[token_id] = True

        # 6. Radical / Wazn Mapping Tables -- BY STRING, INJECTIVE, ONE SPACE
        root_list, self.root_table_info = build_token_root_table(vocab, self.vocab_size, fill=0)
        wazn_list, self.wazn_table_info = build_token_wazn_table(vocab, self.vocab_size, fill=0)
        self.id_to_root_table = torch.tensor(root_list, dtype=torch.long, device=self.device)
        self.id_to_wazn_table = torch.tensor(wazn_list, dtype=torch.long, device=self.device)

        # one-space assertions at construction time
        self.space_census = assert_one_root_space(self.backbone, vocab)
        mx = int(self.id_to_root_table.max())
        if mx > LIVE_ROOTS - 1:
            raise SpaceViolation(f'id_to_root_table max={mx} exceeds the live root space')
        print(f'[Rootformer v13] token->root table: {self.root_table_info["tokens"]} <root_X> '
              f'tokens -> {self.root_table_info["distinct"]} distinct live root ids '
              f'(injective=True, max={mx})')
        print(f'[Rootformer v13] token->wazn table: {self.wazn_table_info["tokens"]} <wazn_X> '
              f'tokens -> {self.wazn_table_info["distinct"]} distinct live wazn ids')

        self.to(self.device)
        if not self.is_training:
            self.eval()
        print(f'[Rootformer v13] Initialized! layers={self.num_layers} vocab={self.vocab_size} '
              f'roots={LIVE_ROOTS} awzan={LIVE_AWZAN} on {self.device}. [OK]', flush=True)

    # ------------------------------------------------------------------ ids
    def extract_morphemic_ids(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Real (root_ids, wazn_ids) from token ids.  No clamp, no offset.

        The v12 `wazn_ids = torch.zeros_like(...)` stub is gone: it returned all-zero wazn ids
        and was mistaken for the architecture even after the dead code was found.
        """
        safe = torch.clamp(input_ids, 0, self.vocab_size - 1)
        return self.id_to_root_table[safe], self.id_to_wazn_table[safe]

    # ------------------------------------------------------------------ forward
    def forward(self, input_ids=None, labels=None, root_targets=None, wazn_targets=None,
                attention_mask=None, inputs_embeds=None, **kwargs) -> Dict[str, Any]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError('forward needs input_ids or inputs_embeds')
            root_ids, wazn_ids = self.extract_morphemic_ids(input_ids)
            self.set_active_morphemic_ids(root_ids, wazn_ids)
        outputs = self.backbone(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True)
        all_hidden_states = outputs.hidden_states
        h_morph = all_hidden_states[self.morph_layer_idx]
        lm_logits = outputs.logits
        root_logits, wazn_logits = self.morphemic_heads(h_morph)

        total_loss = loss_lm = loss_root = loss_wazn = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_lm = F.cross_entropy(shift_logits.view(-1, self.vocab_size),
                                      shift_labels.view(-1), ignore_index=-100)
            total_loss = loss_lm
            if root_targets is not None:
                loss_root = F.cross_entropy(root_logits.view(-1, LIVE_ROOTS),   # was 9015
                                            root_targets.view(-1), ignore_index=-100)
                total_loss = total_loss + 0.05 * loss_root
            if wazn_targets is not None:
                loss_wazn = F.cross_entropy(wazn_logits.view(-1, LIVE_AWZAN),   # was 128
                                            wazn_targets.view(-1), ignore_index=-100)
                total_loss = total_loss + 0.05 * loss_wazn
        return {'loss': total_loss, 'loss_lm': loss_lm, 'loss_root': loss_root,
                'loss_wazn': loss_wazn, 'logits': lm_logits, 'root_logits': root_logits,
                'wazn_logits': wazn_logits}

    def set_active_morphemic_ids(self, root_ids, wazn_ids):
        for layer in self.backbone.model.layers:
            layer.self_attn.active_root_ids = root_ids
            layer.self_attn.active_wazn_ids = wazn_ids

    # ------------------------------------------------------------------ checkpoint
    def load_released_checkpoint(self, path, drop_stale_tables: bool = True) -> Dict[str, Any]:
        """Load the BACKBONE half of a released v19.2 checkpoint, remapping the key prefix.

        A released checkpoint is a `RootformerNRMT` state dict, so its trunk keys are
        `backbone.layers.N.*` / `backbone.embed_tokens` / `backbone.norm` -- because the wrapper
        holds the Qwen2Model directly.  THIS module holds the CausalLM, so its own keys are
        `backbone.model.layers.N.*` / `backbone.model.embed_tokens` / `backbone.model.norm`.
        The prefix is remapped explicitly; a naive `load_state_dict` finds 555 "missing" keys and
        520 "unexpected" ones and silently loads nothing (that is exactly what happened on the
        first run of transmute_v13_check.py, and it is why this remap is written down).

        `morphemic_embed.*`, `final_norm.*` and `nrmp_head.*` belong to the NRMT wrapper, not to
        this module; they are returned under `not_owned` so the caller can overlay them on the
        wrapper with the same call the trainer makes.

        The released file also carries 24 x `self_attn.root_embed.weight` (9015,64) and 24 x
        `self_attn.wazn_embed.weight` (128,64).  Those tables are dead code in the NRMT forward
        (0/24 attention calls carry root ids -- liveness_report.json) AND they are in the wrong
        space, so they cannot load into a 9490/142 module.  Dropping exactly those 48 is asserted
        by the caller; `n_dropped` is returned.
        """
        from safetensors.torch import load_file
        sd = load_file(str(path))
        stale = [k for k in sd if k.endswith('self_attn.root_embed.weight')
                 or k.endswith('self_attn.wazn_embed.weight')]
        dropped = {}
        if drop_stale_tables:
            for k in stale:
                dropped[k] = tuple(sd.pop(k).shape)
        mine, not_owned = {}, {}
        for k, v in sd.items():
            if k.startswith('backbone.layers.') or k == 'backbone.embed_tokens.weight' \
                    or k == 'backbone.norm.weight':
                mine['backbone.model.' + k[len('backbone.'):]] = v
            else:
                not_owned[k] = tuple(v.shape)
        missing, unexpected = self.load_state_dict(mine, strict=False)
        return {'dropped_stale': dropped, 'n_dropped': len(dropped),
                'backbone_keys_mapped': len(mine), 'missing': list(missing),
                'unexpected': list(unexpected), 'not_owned': not_owned}

    # ------------------------------------------------------------------ transmutation
    def transmute_from_clean_qwen2_5(self, base_model_name: str = 'Qwen/Qwen2.5-0.5B',
                                     weights_path: Optional[str] = None,
                                     local_files_only: bool = True) -> Dict[str, Any]:
        """Copy the REAL Qwen tensors into the corrected, resized target.

        The v12 version did `b_state = self.backbone.state_dict()` and copied by NAME where shapes
        happened to match, and for embed_tokens/lm_head it copied the FIRST min(rows) rows of the
        raw Qwen table.  Two defects: (a) `self.backbone` is the CausalLM whose keys are
        `model.layers.N.*`, while the released checkpoint's keys are `layers.N.*`, so the name
        match is against the wrong prefix; (b) slicing rows [0:10052] of Qwen's embed_tokens maps
        Qwen's first 10052 BPE token ids (Latin!) onto the morphemic Arabic ids -- every row is
        the wrong token.

        The weights are read from a SEALED SAFETENSORS FILE, not through the HF hub: on this pod
        the default cache holds only the 681-byte config blob, `/workspace/.hf_home` holds a
        DANGLING symlink for the weights, and the only real 988,097,824-byte copy lives in
        another agent's read-only cache.  Resolving through the hub silently picks the broken
        link; the file is addressed directly instead.

        The real Qwen2.5-0.5B carries 290 tensors that this architecture consumes:
          24 layers x 12 (q/k/v weight+bias, o, gate, up, down, 2 layernorms) = 288
          + embed_tokens + final norm                                            = 2
        lm_head is TIED to embed_tokens in this checkpoint and is not a separate tensor.
        """
        from safetensors.torch import load_file
        if weights_path is None:
            weights_path = find_qwen_weights()
        if weights_path is None:
            raise FileNotFoundError(
                'no real Qwen2.5-0.5B safetensors found.  The HF cache on this pod carries only '
                'the config blob; the weights are a 988 MB file under another agent\'s cache.  '
                'Pass weights_path=... explicitly.  Refusing to fall back to random init: that '
                'is the failure the 290-tensor assertion exists to catch.')
        print(f'[Transmute v13] loading sealed base weights {weights_path}', flush=True)
        raw = load_file(str(weights_path))
        print(f'[Transmute v13] raw base tensors = {len(raw)}')
        src_rows = int(raw['model.embed_tokens.weight'].shape[0])
        print(f'[Transmute v13] raw base embed_tokens rows = {src_rows} -> target '
              f'{self.vocab_size}')

        # resize the target table (it is already 10052 by construction; this is the assertion)
        assert int(self.backbone.model.embed_tokens.weight.shape[0]) == self.vocab_size

        # map the target Qwen2Model names onto the source CausalLM names
        target = {}
        for k, v in self.backbone.model.state_dict().items():
            for cand in ('model.' + k, k):
                if cand in raw:
                    target[k] = raw[cand]
                    break
        adapted, resized, missed = [], [], []
        with torch.no_grad():
            for k, v in target.items():
                dst = self.backbone.model.state_dict()[k]
                if dst.shape == v.shape:
                    dst.copy_(v)
                    adapted.append(k)
                elif k == 'embed_tokens.weight' and dst.shape[0] == self.vocab_size:
                    # RESIZE, explicitly, and record it: rows [0:10052] of the RAW table.  This
                    # is an id-to-id copy of the first 10052 BPE ids, which is NOT a morphemic
                    # embedding; embed_tokens is DEAD in the NRMT path (grad is None, proved in
                    # liveness_report.json), so the corrected construction leaves it as the
                    # honest resize and does not pretend it is an Arabic embedding.
                    dst.copy_(v[:self.vocab_size])
                    resized.append((k, tuple(v.shape), tuple(dst.shape)))
                else:
                    missed.append((k, tuple(v.shape), tuple(dst.shape)))

        info = {'qwen_state_keys': len(raw), 'weights_path': str(weights_path),
                'adapted': len(adapted),
                'resized': resized, 'missed': missed,
                'layer_tensors': None,
                'embed_tokens_rows': src_rows, 'target_rows': self.vocab_size}
        print(f'[Transmute v13] adapted {len(adapted)} tensors + {len(resized)} resized, '
              f'{len(missed)} missed')
        for r in resized:
            print(f'    resized {r[0]}: {r[1]} -> {r[2]}')
        if missed:
            print(f'    MISSED: {missed}')

        # the "290 real Qwen tensors" assertion
        layer_tensors = sum(1 for k in adapted if k.startswith('layers.'))
        consumed = len(adapted) + len(resized)
        if layer_tensors != 288:
            raise SpaceViolation(f'expected 288 real Qwen layer tensors (24 x 12), '
                                 f'adapted {layer_tensors}')
        if 'norm.weight' not in adapted:
            raise SpaceViolation('the final Qwen RMSNorm was not adapted')
        if consumed != 290:
            raise SpaceViolation(f'consumed {consumed} Qwen tensors, expected 290 '
                                 f'(288 layer + embed_tokens + norm)')
        if missed:
            raise SpaceViolation(f'{len(missed)} target tensors had no matching Qwen source: '
                                 f'{missed}')
        info['layer_tensors'] = layer_tensors
        info['consumed_qwen_tensors'] = consumed
        print(f'[Transmute v13] {layer_tensors} layer tensors + embed_tokens(resized) + norm = '
              f'{consumed}/290 real Qwen tensors [OK]')
        del raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return info

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.4,
                 top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.20):
        self.eval()
        input_ids_list = self.tokenizer.encode(prompt)
        generated = list(input_ids_list)
        t0 = time.time()
        n = 0
        for _ in range(max_new_tokens):
            cur = torch.tensor([generated], dtype=torch.long, device=self.device)
            r_ids, w_ids = self.extract_morphemic_ids(cur)
            self.set_active_morphemic_ids(r_ids, w_ids)
            out = self.backbone(cur, return_dict=True)
            lg = out.logits[0, -1, :].clone()
            lg[self.metadata_mask] = -float('Inf')
            if repetition_penalty != 1.0 and generated:
                for tid in set(generated[-80:]):
                    lg[tid] = lg[tid] / repetition_penalty if lg[tid] > 0 \
                        else lg[tid] * repetition_penalty
            if temperature <= 0.01:
                nxt = int(torch.argmax(lg))
            else:
                lg = lg / temperature
                if top_k > 0:
                    rm = lg < torch.topk(lg, top_k)[0][..., -1, None]
                    lg[rm] = -float('Inf')
                if top_p < 1.0:
                    sl, si = torch.sort(lg, descending=True)
                    cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    rm = cum > top_p
                    rm[..., 1:] = rm[..., :-1].clone()
                    rm[..., 0] = 0
                    lg[si[rm]] = -float('Inf')
                nxt = int(torch.multinomial(F.softmax(lg, dim=-1), 1))
            generated.append(nxt)
            n += 1
            if nxt in (2, self.tokenizer.token_to_id.get('<eos>', 2)):
                break
        el = time.time() - t0
        full = self.tokenizer.decode(generated)
        return {'full_text': full, 'tokens_generated': n, 'elapsed_seconds': el,
                'tokens_per_sec': n / max(el, 1e-4)}
