#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transmute_quickstart.py
========================================================================================
ROOTFORMER v19.2: SOVEREIGN ARABIC-TO-ENGLISH TRANSMUTATION ENGINE
Basran Bedrock (Al-Aṣl) + Andalusian Grammatical Addition (Al-Farʿ) Synthesis
========================================================================================

Quickstart Usage:
    # 1. Interactive proposition transmutation:
    python transmute_quickstart.py "العلم نور يضيء العقل ويهدي إلى الحق"

    # 2. Run built-in classical heritage benchmark:
    python transmute_quickstart.py --demo
"""

import sys
import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from models.unified_rootformer_v12 import UnifiedRootformerV12
from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
from nrmp_vocab import FarāhīdianMorphemicVocab
from rootformer_v18_nrmp_model import RootformerV18_NRMP
from neural_transmuter_head import NeuralFarahidianTransmuterHead
from khalil_students_andalusian_master_engine import MasterSovereignTransmuter

import json

def load_v19_2_engine(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[*] Initializing Rootformer v19.2 Synthesis Engine on {device}...")
    bp_path = ROOT_DIR / 'data/rootformer_v12_arabic_blueprint.json'
    vocab = FarāhīdianMorphemicVocab(str(bp_path))

    with open(ROOT_DIR / 'data/concept_vocabulary.json', 'r', encoding='utf-8') as f:
        vdata = json.load(f)
    token_to_id = vdata['token_to_id']
    id_to_token = {int(k): v for k, v in vdata['id_to_token'].items()}

    # Arabic 24-Layer Backbone (Layers 18-23 Co-Trained)
    base_model = UnifiedRootformerV12(str(bp_path), 'Qwen/Qwen2.5-0.5B', device, torch.bfloat16).to(device)
    flash_model = UnifiedRootformerV17_DeepSeekFlash(base_model, device, torch.bfloat16).to(device)
    ar_model = RootformerV18_NRMP(flash_model, vocab, device, torch.bfloat16).to(device)

    bb_ckpt = ROOT_DIR / 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors'
    ar_model.load_state_dict(load_file(str(bb_ckpt)), strict=False)
    ar_model.eval()

    # 8-Layer Jurjānī Deep Transmuter Head
    d_model = getattr(base_model.backbone.config, 'hidden_size', 896)
    transmuter = NeuralFarahidianTransmuterHead(
        d_model=d_model,
        d_concept=512,
        concept_vocab_size=vdata['vocab_size'],
        num_roots=vocab.num_roots,
        guide_layer_idx=14,
        num_decoder_layers=8,
        dropout=0.0
    ).to(device=device, dtype=torch.bfloat16)

    trans_ckpt = ROOT_DIR / 'checkpoints/rootformer_v19_2_synthesis_transmuter_master.safetensors'
    transmuter.load_state_dict(load_file(str(trans_ckpt)), strict=False)
    transmuter.eval()

    engine = MasterSovereignTransmuter(transmuter, vocab, token_to_id, id_to_token)
    return ar_model, engine, vocab, device

def transmute(text: str, ar_model, engine, vocab, device):
    tuples = vocab.encode_sentence(text)
    p_ids = torch.tensor([[t[0] for t in tuples]], dtype=torch.long, device=device)
    r_ids = torch.tensor([[t[1] for t in tuples]], dtype=torch.long, device=device)
    w_ids = torch.tensor([[t[2] for t in tuples]], dtype=torch.long, device=device)
    s_ids = torch.tensor([[t[3] for t in tuples]], dtype=torch.long, device=device)

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        inputs_embeds = ar_model.morphemic_embed(p_ids, r_ids, w_ids, s_ids)
        backbone_out = ar_model.backbone(inputs_embeds=inputs_embeds, output_hidden_states=True)
        all_hs = backbone_out.hidden_states
        output = engine.transmute_proposition(all_hs, text)
    return output

def main():
    parser = argparse.ArgumentParser(description="Rootformer v19.2 Quickstart CLI")
    parser.add_argument("text", nargs="?", default=None, help="Arabic proposition to transmute")
    parser.add_argument("--demo", action="store_true", help="Run full classical benchmark demo")
    args = parser.parse_args()

    ar_model, engine, vocab, device = load_v19_2_engine()

    if args.demo or args.text is None:
        demo_sentences = [
            ("Al-Ghazali", "العلم نور يضيء العقل ويهدي إلى الحق"),
            ("Sibawayh & Maxim", "رأس الحكمة مخافة الله"),
            ("Mecelle (Ottoman Law)", "اليقين لا يزول بالشك"),
            ("Ibn Malik (Alfiyyah)", "كلامنا لفظ مفيد كاستقم واسم وفعل ثم حرف الكلم"),
            ("Ibn Mada al-Qurtubi", "إنما العمل من النصب والرفع للمتكلم نفسه"),
            ("Ibn Khaldun", "اللسان ملكة صناعية يحصل بالممارسة وتكرار الكلام العربي"),
            ("Al-Jurjani", "ليس النظم إلا توخي معاني النحو فيما بين الكلم")
        ]
        print("\n" + "=" * 80)
        print("  ROOTFORMER v19.2: CLASSICAL HERITAGE TRANSMUTATION BENCHMARK DEMO")
        print("=" * 80)
        for author, sent in demo_sentences:
            res = transmute(sent, ar_model, engine, vocab, device)
            print(f"\n[{author}]")
            print(f"  Arabic   : {sent}")
            print(f"  Transmute: {res}")
        print("\n" + "=" * 80)
    else:
        res = transmute(args.text, ar_model, engine, vocab, device)
        print(f"\nInput Arabic : {args.text}")
        print(f"Transmutation: {res}\n")

if __name__ == '__main__':
    main()
