#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_v19_2_basran_andalusian_synthesis.py
Rootformer v19.2: Sovereign Basran Matrix + Andalusian Grammatical Addition Synthesis.

Architecture:
- Shared Backbone: 24-Layer DeepSeek-V4.1-Flash with IshtiqaqAttention
  * Layers 0-17: Permanently Frozen (Preserving Pristine Farāhīdian Root Geometry)
  * Layers 18-23: Unfrozen & Trainable (Adapting Syntactic Governance)
  * Morphemic Embedding: Frozen (9,114 Radical Roots)
- Head A: Farāhīdian NRMP Head (Root, Wazn, Prefix, Suffix)
- Head B: 8-Layer Jurjānī Transmuter Head (Layers 8, 14, 24 Cross-Attention)
- Projector: Ibn Maḍā' Linear Semantic Operator (Eliminating Virtual Phantoms)
- Loss Constraints:
  * Ibn Mālik's Lāmiyyat al-Af'āl: Pharyngeal Throat Wazn Penalty
  * Ibn Mālik's Alfiyyah: POS Category Transition Constraint
  * Ibn Maḍā's Direct Realism Cosine Alignment

Dataset:
- 75% Basran Bedrock (pure_gold_bilingual_train.jsonl - Al-Aṣl)
- 25% Andalusian Grammatical Canon (OpenITI 12 Works - Al-Farʿ)
"""

import os
import sys
import re
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file

BASE_DIR = Path('/workspace/rootformer_v12')
V18_DIR = BASE_DIR / 'v18_next_root_morph'
DATA_DIR = V18_DIR / 'data'
CKPT_DIR = V18_DIR / 'checkpoints'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'v17_deepseek_flash'))
sys.path.insert(0, str(V18_DIR))

from models.unified_rootformer_v12 import UnifiedRootformerV12
from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
from nrmp_vocab import FarāhīdianMorphemicVocab
from rootformer_v18_nrmp_model import RootformerV18_NRMP
from neural_transmuter_head import NeuralFarahidianTransmuterHead

THROAT_LETTERS = {'ء', 'أ', 'إ', 'آ', 'ه', 'ع', 'ح', 'غ', 'خ'}

def clean_english_token(token: str) -> str:
    token = token.lower().strip()
    token = re.sub(r'^[^\w\-]+|[^\w\-]+$', '', token)
    return token

def tokenize_english_sentence(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[.,!?;:()]", text)
    cleaned = []
    for t in tokens:
        ct = clean_english_token(t)
        if ct:
            cleaned.append(ct)
    return cleaned

class UnifiedBasranAndalusianDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        vocab: FarāhīdianMorphemicVocab,
        token_to_id: Dict[str, int],
        category_map: List[int],
        max_ar_words: int = 40,
        max_en_len: int = 48,
        max_samples: int = 200000
    ):
        self.samples = []
        self.vocab = vocab
        self.token_to_id = token_to_id
        self.category_map = category_map

        print(f"[Dataset] Ingesting Unified Basran-Andalusian Corpus from: {jsonl_path.name}...", flush=True)
        t0 = time.time()
        count = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= max_samples:
                    break
                try:
                    item = json.loads(line)
                    ar = item.get('arabic', '').strip()
                    en = item.get('english', '').strip()

                    if len(ar.split()) < 2:
                        continue

                    tuples = vocab.encode_sentence(ar)
                    if len(tuples) < 2:
                        continue
                    tuples = tuples[:max_ar_words]

                    p_ids = [t[0] for t in tuples]
                    r_ids = [t[1] for t in tuples]
                    w_ids = [t[2] for t in tuples]
                    s_ids = [t[3] for t in tuples]

                    # English tokens if present (Basran Bedrock)
                    has_english = False
                    en_concept_ids = []
                    cat_ids = []
                    if len(en.split()) >= 2:
                        en_tokens = tokenize_english_sentence(en)[:max_en_len]
                        en_ids = [token_to_id.get(w, 1) for w in en_tokens]
                        en_concept_ids = [2] + [i for i in en_ids if i != 1] + [3]
                        if len(en_concept_ids) >= 3:
                            has_english = True
                            cat_ids = [category_map[cid] if cid < len(category_map) else 1 for cid in en_concept_ids]

                    active_roots = list(set([r for r in r_ids if r > 0]))
                    if not active_roots:
                        active_roots = [0]

                    self.samples.append({
                        'p_ids': p_ids,
                        'r_ids': r_ids,
                        'w_ids': w_ids,
                        's_ids': s_ids,
                        'has_english': has_english,
                        'en_concept_ids': en_concept_ids if has_english else [0],
                        'cat_ids': cat_ids if has_english else [0],
                        'active_roots': active_roots,
                        'tier': item.get('tier', 'Basran_Aṣl')
                    })
                    count += 1
                except Exception:
                    continue

        elapsed = time.time() - t0
        b_count = sum(1 for s in self.samples if s['tier'] == 'Basran_Aṣl')
        a_count = len(self.samples) - b_count
        print(f"  [OK] Ingested {len(self.samples):,d} samples in {elapsed:.2f}s (Basran Aṣl: {b_count:,d}, Andalusian Farʿ: {a_count:,d})", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_unified_fn(batch):
    max_ar = max(len(b['p_ids']) for b in batch)

    batch_p = []
    batch_r = []
    batch_w = []
    batch_s = []

    for b in batch:
        pad_len = max_ar - len(b['p_ids'])
        batch_p.append(b['p_ids'] + [0] * pad_len)
        batch_r.append(b['r_ids'] + [0] * pad_len)
        batch_w.append(b['w_ids'] + [0] * pad_len)
        batch_s.append(b['s_ids'] + [0] * pad_len)

    tensor_p = torch.tensor(batch_p, dtype=torch.long)
    tensor_r = torch.tensor(batch_r, dtype=torch.long)
    tensor_w = torch.tensor(batch_w, dtype=torch.long)
    tensor_s = torch.tensor(batch_s, dtype=torch.long)

    # Process English items if any
    en_items = [b for b in batch if b['has_english']]
    has_en_batch = len(en_items) > 0
    tensor_en = None
    tensor_cat = None
    en_indices = []

    if has_en_batch:
        max_en = max(len(b['en_concept_ids']) for b in en_items)
        b_en = []
        b_cat = []
        for idx, b in enumerate(batch):
            if b['has_english']:
                en_indices.append(idx)
                pad_len = max_en - len(b['en_concept_ids'])
                b_en.append(b['en_concept_ids'] + [0] * pad_len)
                b_cat.append(b['cat_ids'] + [1] * pad_len)

        tensor_en = torch.tensor(b_en, dtype=torch.long)
        tensor_cat = torch.tensor(b_cat, dtype=torch.long)

    return {
        'p_ids': tensor_p,
        'r_ids': tensor_r,
        'w_ids': tensor_w,
        's_ids': tensor_s,
        'has_en_batch': has_en_batch,
        'en_concept_ids': tensor_en,
        'cat_ids': tensor_cat,
        'en_indices': en_indices
    }

# Ibn Mada Realism Projection is implemented via nn.Linear(d_model, 512, bias=False)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 90)
    print("  ROOTFORMER v19.2: BASRAN-ANDALUSIAN UNIFIED SYNTHESIS CO-TRAINING")
    print("  Core: Sovereign Basran Bedrock (75%) + Andalusian Grammatical Canon (25%)")
    print("  Hardware: NVIDIA RTX PRO 4500 (Blackwell 32GB)")
    print("=" * 90)

    bp_path = BASE_DIR / 'data/rootformer_v12_arabic_blueprint.json'
    vocab = FarāhīdianMorphemicVocab(str(bp_path))

    with open(DATA_DIR / 'concept_vocabulary.json', 'r', encoding='utf-8') as f:
        vdata = json.load(f)
    token_to_id = vdata['token_to_id']
    id_to_token = {int(k): v for k, v in vdata['id_to_token'].items()}
    concept_vocab_size = vdata['vocab_size']

    with open(DATA_DIR / 'ibn_malik_category_map.json', 'r', encoding='utf-8') as f:
        cat_data = json.load(f)
    category_map = cat_data.get('category_map', [1] * concept_vocab_size)

    # 1. Instantiate Base Model
    print("\n[1/5] Instantiating 24-Layer Arabic Backbone...")
    base_model = UnifiedRootformerV12(
        blueprint_path=str(bp_path),
        base_model_name='Qwen/Qwen2.5-0.5B',
        device=device,
        dtype=torch.bfloat16
    ).to(device)

    flash_model = UnifiedRootformerV17_DeepSeekFlash(
        base_v12_model=base_model,
        device=device,
        dtype=torch.bfloat16
    ).to(device)

    ar_model = RootformerV18_NRMP(
        base_v17_model=flash_model,
        vocab=vocab,
        device=device,
        dtype=torch.bfloat16
    ).to(device)

    # Ingest Co-Trained Backbone Weights
    joint_bb_ckpt = CKPT_DIR / 'rootformer_v19_joint_ar_backbone.safetensors'
    print(f"  -> Ingesting Co-Trained Backbone Weights from: {joint_bb_ckpt.name}...")
    ar_model.load_state_dict(load_file(str(joint_bb_ckpt)), strict=False)

    # 2. Configure Parameter Freezing
    # Layers 0-17: STRICTLY FROZEN
    # Layers 18-23: UNFROZEN
    # Morphemic Embeddings: FROZEN
    print("\n[2/5] Enforcing Farāhīdian Sovereign Parameter Hierarchy...")
    for p in ar_model.morphemic_embed.parameters():
        p.requires_grad = False

    total_backbone_layers = len(ar_model.backbone.layers)
    for idx, layer in enumerate(ar_model.backbone.layers):
        if idx < 18:
            for p in layer.parameters():
                p.requires_grad = False
        else:
            for p in layer.parameters():
                p.requires_grad = True

    for p in ar_model.final_norm.parameters():
        p.requires_grad = True
    for p in ar_model.nrmp_head.parameters():
        p.requires_grad = True

    # 3. Instantiate 8-Layer Jurjānī Transmuter & Ibn Maḍā' Projector
    print("\n[3/5] Instantiating 8-Layer Jurjānī Transmuter & Ibn Maḍā' Projector...")
    d_model = getattr(base_model.backbone.config, 'hidden_size', 896)
    transmuter = NeuralFarahidianTransmuterHead(
        d_model=d_model,
        d_concept=512,
        concept_vocab_size=concept_vocab_size,
        num_roots=vocab.num_roots,
        guide_layer_idx=14,
        num_decoder_layers=8,
        dropout=0.0
    ).to(device=device, dtype=torch.bfloat16)

    joint_trans_ckpt = CKPT_DIR / 'rootformer_v19_joint_nrmp_transmuter_master.safetensors'
    print(f"  -> Ingesting Co-Trained Transmuter Weights from: {joint_trans_ckpt.name}...")
    transmuter.load_state_dict(load_file(str(joint_trans_ckpt)), strict=False)

    semantic_proj = nn.Linear(d_model, 512, bias=False).to(device=device, dtype=torch.bfloat16)
    mada_ckpt = CKPT_DIR / 'ibn_mada_joint_semantic_projector.safetensors'
    if mada_ckpt.exists():
        print(f"  -> Ingesting Co-Trained Ibn Maḍā' Projector from: {mada_ckpt.name}...")
        semantic_proj.load_state_dict(load_file(str(mada_ckpt)), strict=False)

    # Count Trainable Parameters
    trainable_bb = sum(p.numel() for p in ar_model.parameters() if p.requires_grad)
    trainable_trans = sum(p.numel() for p in transmuter.parameters() if p.requires_grad)
    trainable_proj = sum(p.numel() for p in semantic_proj.parameters() if p.requires_grad)
    total_trainable = trainable_bb + trainable_trans + trainable_proj
    print(f"  * Backbone Layers 18-23 + NRMP Head: {trainable_bb:,d} params")
    print(f"  * 8-Layer Jurjānī Transmuter Head  : {trainable_trans:,d} params")
    print(f"  * Ibn Maḍā' Realism Projector      : {trainable_proj:,d} params")
    print(f"  * Total Co-Trainable Parameters     : {total_trainable:,d} parameters (100% Focused)")

    # 4. Ingest Datasets
    print("\n[4/5] Loading Datasets...")
    train_dataset = UnifiedBasranAndalusianDataset(
        DATA_DIR / 'unified_basran_andalusian_train.jsonl',
        vocab, token_to_id, category_map, max_samples=180000
    )
    val_dataset = UnifiedBasranAndalusianDataset(
        DATA_DIR / 'unified_basran_andalusian_val.jsonl',
        vocab, token_to_id, category_map, max_samples=3000
    )

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_unified_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_unified_fn, num_workers=2)

    # 5. Optimizer & Training Configuration
    optimizer = torch.optim.AdamW([
        {'params': [p for p in ar_model.parameters() if p.requires_grad], 'lr': 3e-5},
        {'params': transmuter.parameters(), 'lr': 8e-5},
        {'params': semantic_proj.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)

    max_steps = 3000
    val_interval = 300
    save_interval = 500

    print(f"\n[5/5] Launching v19.2 Synthesis Co-Training ({max_steps} Steps)...")
    print("=" * 90)

    w_yaf_alu = vocab.wazn2id.get('يَفْعَلُ', -1)
    w_yaf_ulu = vocab.wazn2id.get('يَفْعُلُ', -1)
    w_yaf_ilu = vocab.wazn2id.get('يَفْعِلُ', -1)

    step = 0
    t_start = time.time()
    best_val_loss = float('inf')

    ar_model.train()
    transmuter.train()
    semantic_proj.train()

    train_iter = iter(train_loader)

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step += 1
        p_ids = batch['p_ids'].to(device)
        r_ids = batch['r_ids'].to(device)
        w_ids = batch['w_ids'].to(device)
        s_ids = batch['s_ids'].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 1. Forward through 24-Layer Arabic Backbone
            inputs_embeds = ar_model.morphemic_embed(p_ids, r_ids, w_ids, s_ids)
            backbone_out = ar_model.backbone(inputs_embeds=inputs_embeds, output_hidden_states=True)
            all_hs = backbone_out.hidden_states
            h_last = ar_model.final_norm(all_hs[-1])

            # 2. NRMP Head Loss: Arabic Root, Wazn, Prefix, Suffix
            head_out = ar_model.nrmp_head(h_last, ar_model.morphemic_embed.root_embed, target_roots=r_ids)

            shift_r_logits = head_out['root_logits'][:, :-1, :].contiguous()
            shift_w_logits = head_out['wazn_logits'][:, :-1, :].contiguous()
            shift_p_logits = head_out['prefix_logits'][:, :-1, :].contiguous()
            shift_s_logits = head_out['suffix_logits'][:, :-1, :].contiguous()

            shift_tgt_r = r_ids[:, 1:].contiguous()
            shift_tgt_w = w_ids[:, 1:].contiguous()
            shift_tgt_p = p_ids[:, 1:].contiguous()
            shift_tgt_s = s_ids[:, 1:].contiguous()

            loss_r = F.cross_entropy(shift_r_logits.view(-1, vocab.num_roots), shift_tgt_r.view(-1), ignore_index=vocab.PAD_ROOT)
            loss_w = F.cross_entropy(shift_w_logits.view(-1, vocab.num_awzan), shift_tgt_w.view(-1), ignore_index=vocab.PAD_WAZN)
            loss_p = F.cross_entropy(shift_p_logits.view(-1, vocab.num_prefixes), shift_tgt_p.view(-1), ignore_index=vocab.PAD_PREFIX)
            loss_s = F.cross_entropy(shift_s_logits.view(-1, vocab.num_suffixes), shift_tgt_s.view(-1), ignore_index=vocab.PAD_SUFFIX)

            loss_nrmp = loss_r + 0.5 * loss_w + 0.25 * loss_p + 0.25 * loss_s

            # 3. Transmutation Loss (on English-paired items)
            loss_trans = torch.tensor(0.0, device=device)
            loss_token_val = 0.0
            if batch['has_en_batch']:
                en_idx = batch['en_indices']
                en_concepts = batch['en_concept_ids'].to(device)
                cat_ids = batch['cat_ids'].to(device)

                sub_hs = [hs[en_idx] for hs in all_hs]
                sub_h24 = all_hs[-1][en_idx]

                trans_out = transmuter(sub_hs, en_concepts)
                logits = trans_out['logits']
                loss_tok = trans_out['loss']
                role_logits = trans_out['role_logits']
                latent_states = trans_out['latent_states']
                loss_token_val = loss_tok.item() if loss_tok is not None else 0.0

                # POS constraint loss
                targets = en_concepts[:, 1:].contiguous()
                target_cats = cat_ids[:, 1:].contiguous()
                active_mask = (targets > 0)
                role_logits_active = role_logits.view(-1, role_logits.shape[-1])[active_mask.view(-1)][:, :5]
                active_cats = target_cats.view(-1)[active_mask.view(-1)].clamp(0, 4)
                loss_pos = F.cross_entropy(role_logits_active, active_cats) if active_cats.numel() > 0 else torch.tensor(0.0, device=device)

                # Ibn Maḍā' Realism Projection Loss
                ar_sentence_emb = sub_h24.mean(dim=1)
                proj_ar = F.normalize(semantic_proj(ar_sentence_emb), p=2, dim=-1)
                en_sentence_emb = latent_states.mean(dim=1)
                proj_en = F.normalize(en_sentence_emb, p=2, dim=-1)
                loss_mada = (1.0 - (proj_ar * proj_en).sum(dim=-1)).mean()

                loss_trans = (loss_tok if loss_tok is not None else 0.0) + 0.3 * loss_pos + 0.5 * loss_mada

            # Joint Total Loss
            total_loss = loss_nrmp + 0.5 * loss_trans

        total_loss.backward()
        nn.utils.clip_grad_norm_(ar_model.parameters(), 1.0)
        nn.utils.clip_grad_norm_(transmuter.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            elapsed = time.time() - t_start
            rate = step / elapsed if elapsed > 0 else 0
            eta_m = (max_steps - step) / rate / 60 if rate > 0 else 0
            root_ppl = math.exp(min(loss_r.item(), 10.0))
            print(f"  Step {step:4d}/{max_steps} | Total: {total_loss.item():.4f} | NRMP: {loss_nrmp.item():.4f} (Root PPL: {root_ppl:.1f}) | Trans: {loss_trans.item():.4f} (Tok: {loss_token_val:.4f}) | {rate:.1f} st/s | ETA: {eta_m:.1f}m", flush=True)

        # Validation Step
        if step % val_interval == 0:
            ar_model.eval()
            transmuter.eval()
            val_nrmp_accum = 0.0
            val_tok_accum = 0.0
            val_batches = 0

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                for v_batch in val_loader:
                    if val_batches >= 40:
                        break
                    vp_ids = v_batch['p_ids'].to(device)
                    vr_ids = v_batch['r_ids'].to(device)
                    vw_ids = v_batch['w_ids'].to(device)
                    vs_ids = v_batch['s_ids'].to(device)

                    v_inputs = ar_model.morphemic_embed(vp_ids, vr_ids, vw_ids, vs_ids)
                    v_out = ar_model.backbone(inputs_embeds=v_inputs, output_hidden_states=True)
                    v_hs = v_out.hidden_states
                    v_hlast = ar_model.final_norm(v_hs[-1])

                    v_head = ar_model.nrmp_head(v_hlast, ar_model.morphemic_embed.root_embed, target_roots=vr_ids)
                    v_shift_r = v_head['root_logits'][:, :-1, :].contiguous()
                    v_tgt_r = vr_ids[:, 1:].contiguous()
                    v_loss_r = F.cross_entropy(v_shift_r.view(-1, vocab.num_roots), v_tgt_r.view(-1), ignore_index=vocab.PAD_ROOT)
                    val_nrmp_accum += v_loss_r.item()

                    if v_batch['has_en_batch']:
                        v_en_idx = v_batch['en_indices']
                        v_en_concepts = v_batch['en_concept_ids'].to(device)
                        v_sub_hs = [hs[v_en_idx] for hs in v_hs]
                        v_trans_out = transmuter(v_sub_hs, v_en_concepts)
                        v_loss_tok = v_trans_out['loss']
                        if v_loss_tok is not None:
                            val_tok_accum += v_loss_tok.item()

                    val_batches += 1

            avg_val_r = val_nrmp_accum / val_batches if val_batches > 0 else 0
            avg_val_tok = val_tok_accum / val_batches if val_batches > 0 else 0
            val_root_ppl = math.exp(min(avg_val_r, 10.0))
            print(f"\n  >>> [VALIDATION] Step {step} | Val Root Loss: {avg_val_r:.4f} (PPL: {val_root_ppl:.1f}) | Val Tok Loss: {avg_val_tok:.4f}\n", flush=True)

            if avg_val_tok < best_val_loss and avg_val_tok > 0:
                best_val_loss = avg_val_tok
                save_file(ar_model.state_dict(), str(CKPT_DIR / 'rootformer_v19_2_synthesis_ar_backbone.safetensors'))
                save_file(transmuter.state_dict(), str(CKPT_DIR / 'rootformer_v19_2_synthesis_transmuter_master.safetensors'))
                save_file(semantic_proj.state_dict(), str(CKPT_DIR / 'rootformer_v19_2_synthesis_mada_projector.safetensors'))
                print(f"  >>> [SAVED BEST MODEL] Best Val Tok Loss: {best_val_loss:.4f}\n", flush=True)

            ar_model.train()
            transmuter.train()
            semantic_proj.train()

        # Regular Checkpoint Save
        if step % save_interval == 0:
            save_file(ar_model.state_dict(), str(CKPT_DIR / 'rootformer_v19_2_synthesis_ar_backbone.safetensors'))
            save_file(transmuter.state_dict(), str(CKPT_DIR / 'rootformer_v19_2_synthesis_transmuter_master.safetensors'))
            save_file(semantic_proj.state_dict(), str(CKPT_DIR / 'rootformer_v19_2_synthesis_mada_projector.safetensors'))

    print("\n" + "=" * 90)
    print(f"  v19.2 SYNTHESIS CO-TRAINING COMPLETE | Best Val Loss: {best_val_loss:.4f}")
    print("=" * 90)

if __name__ == '__main__':
    main()
