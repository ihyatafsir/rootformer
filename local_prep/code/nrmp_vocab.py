#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmp_vocab.py
Farāhīdian Next-Root/Morph Vocabulary and Disentangled Representation.
Maps Arabic words to structured tuples: (prefix, root, wazn, suffix)
and provides deterministic morphological realization.
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

class FarāhīdianMorphemicVocab:
    def __init__(self, blueprint_path: str):
        self.blueprint_path = blueprint_path
        
        # Ensure parent models path is accessible
        models_dir = str(Path(blueprint_path).parent.parent / 'models')
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
            
        from morphemic_tokenizer_v12_arabic import PureArabicMorphemicTokenizerV12
        self.base_tok = PureArabicMorphemicTokenizerV12(blueprint_path)

        # Basran Syntactic Realizer Engine (محرك الضبط النحوي والإعرابي)
        try:
            from farahidian_syntactic_realizer import BasranSyntacticRealizer
            self.syntactic_realizer = BasranSyntacticRealizer(self)
        except ImportError:
            self.syntactic_realizer = None
        
        # Build Special & Particle Lists
        self.COMMON_PARTICLES = [
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'حتى', 'منذ', 'مذ',
            'إن', 'أن', 'كأن', 'لكن', 'ليت', 'لعل', 'لا', 'ما', 'لم', 'لن',
            'ليس', 'كان', 'هو', 'هي', 'هما', 'هم', 'هن', 'أنت', 'أنا', 'نحن',
            'هذا', 'هذه', 'هؤلاء', 'ذلك', 'تلك', 'أولئك', 'الذي', 'التي', 'الذين',
            'ثم', 'أو', 'بل', 'أم', 'إما', 'إذا', 'إذ', 'كل', 'بعض', 'غير',
            'سوى', 'عند', 'بين', 'فوق', 'تحت', 'قبل', 'بعد', 'حيث', 'كيف',
            'أين', 'متى', 'كم', 'أي', 'نعم', 'بلى', 'إذن', 'قد', 'سوف'
        ]
        
        # 1. ROOT VOCABULARY (~9,200)
        self.special_roots = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<PARTICLE>']
        raw_roots = sorted(list(self.base_tok.roots_set))
        
        # Dedicated particles in root slot
        particle_roots = [f'<P:{p}>' for p in self.COMMON_PARTICLES]
        self.roots_list = self.special_roots + particle_roots + raw_roots
        self.root2id = {r: i for i, r in enumerate(self.roots_list)}
        self.id2root = {i: r for i, r in enumerate(self.roots_list)}
        
        # 2. WAZN VOCABULARY (~130)
        self.special_awzan = ['<PAD>', '<NONE>', '<UNK>']
        raw_awzan = sorted(list(self.base_tok.awzan_set))
        self.awzan_list = self.special_awzan + raw_awzan
        self.wazn2id = {w: i for i, w in enumerate(self.awzan_list)}
        self.id2wazn = {i: w for i, w in enumerate(self.awzan_list)}
        
        # 3. PREFIX VOCABULARY (~26)
        self.canonical_prefixes = [
            '<PAD>', '<NONE>', '<UNK>', 'ال', 'و', 'وال', 'ف', 'فال',
            'ب', 'بال', 'ل', 'لل', 'ك', 'كال', 'كل', 'في', 'من', 'على',
            'عن', 'إلى', 'أن', 'إن', 'كأن', 'ليس', 'مع', 'س'
        ]
        self.prefix2id = {p: i for i, p in enumerate(self.canonical_prefixes)}
        self.id2prefix = {i: p for i, p in enumerate(self.canonical_prefixes)}
        
        # 4. SUFFIX VOCABULARY (~24)
        self.canonical_suffixes = [
            '<PAD>', '<NONE>', '<UNK>', 'ه', 'ها', 'هم', 'هما', 'هن',
            'ك', 'كم', 'كن', 'نا', 'ي', 'ت', 'تم', 'ا', 'ون', 'ين',
            'ان', 'ات', 'ة', 'ية', 'هم'
        ]
        # Deduplicate while preserving order
        seen = set()
        dedup_suffixes = []
        for s in self.canonical_suffixes:
            if s not in seen:
                seen.add(s)
                dedup_suffixes.append(s)
        self.canonical_suffixes = dedup_suffixes
        self.suffix2id = {s: i for i, s in enumerate(self.canonical_suffixes)}
        self.id2suffix = {i: s for i, s in enumerate(self.canonical_suffixes)}
        
        # Constants
        self.PAD_ROOT = self.root2id['<PAD>']
        self.BOS_ROOT = self.root2id['<BOS>']
        self.EOS_ROOT = self.root2id['<EOS>']
        self.UNK_ROOT = self.root2id['<UNK>']
        
        self.PAD_WAZN = self.wazn2id['<PAD>']
        self.NONE_WAZN = self.wazn2id['<NONE>']
        
        self.PAD_PREFIX = self.prefix2id['<PAD>']
        self.NONE_PREFIX = self.prefix2id['<NONE>']
        
        self.PAD_SUFFIX = self.suffix2id['<PAD>']
        self.NONE_SUFFIX = self.suffix2id['<NONE>']

    @property
    def num_roots(self) -> int:
        return len(self.roots_list)
        
    @property
    def num_awzan(self) -> int:
        return len(self.awzan_list)
        
    @property
    def num_prefixes(self) -> int:
        return len(self.canonical_prefixes)
        
    @property
    def num_suffixes(self) -> int:
        return len(self.canonical_suffixes)

    def encode_word(self, word: str) -> Tuple[int, int, int, int]:
        """
        Decomposes a single Arabic word into a 4-tuple of categorical integer IDs:
        (prefix_id, root_id, wazn_id, suffix_id)
        """
        clean_w = word.strip()
        if not clean_w:
            return (self.NONE_PREFIX, self.UNK_ROOT, self.NONE_WAZN, self.NONE_SUFFIX)
            
        # Check if entire word is a direct particle
        particle_tag = f'<P:{clean_w}>'
        if particle_tag in self.root2id:
            return (self.NONE_PREFIX, self.root2id[particle_tag], self.NONE_WAZN, self.NONE_SUFFIX)
            
        # Decompose using morphological analyzer
        p, r, wz, s = self.base_tok.decompose_arabic_word(clean_w)
        
        # Map Prefix
        if p is None or p == '':
            p_id = self.NONE_PREFIX
        else:
            p_id = self.prefix2id.get(p, self.prefix2id['<UNK>'])
            
        # Map Root
        if r is None or r == '':
            if clean_w in self.COMMON_PARTICLES or f'<P:{clean_w}>' in self.root2id:
                r_id = self.root2id.get(f'<P:{clean_w}>', self.root2id['<PARTICLE>'])
            else:
                r_id = self.root2id['<PARTICLE>']
        else:
            particle_cand = f'<P:{r}>'
            if particle_cand in self.root2id:
                r_id = self.root2id[particle_cand]
            else:
                r_id = self.root2id.get(r, self.root2id['<UNK>'])
                
        # Map Wazn
        if wz is None or wz == '':
            w_id = self.NONE_WAZN
        else:
            w_id = self.wazn2id.get(wz, self.wazn2id['<UNK>'])
            
        # Map Suffix
        if s is None or s == '':
            s_id = self.NONE_SUFFIX
        else:
            s_id = self.suffix2id.get(s, self.suffix2id['<UNK>'])
            
        return (p_id, r_id, w_id, s_id)

    def decode_word(self, p_id: int, r_id: int, w_id: int, s_id: int) -> str:
        """
        Deterministically realizes a (prefix, root, wazn, suffix) tuple back into
        a surface Arabic word string.
        """
        # Check special tokens
        if r_id == self.PAD_ROOT or r_id == self.BOS_ROOT or r_id == self.EOS_ROOT:
            return ''
            
        root_str = self.id2root.get(r_id, '<UNK>')
        wazn_str = self.id2wazn.get(w_id, '<NONE>')
        prefix_str = self.id2prefix.get(p_id, '<NONE>')
        suffix_str = self.id2suffix.get(s_id, '<NONE>')
        
        p_text = '' if prefix_str in ['<NONE>', '<PAD>', '<UNK>'] else prefix_str
        s_text = '' if suffix_str in ['<NONE>', '<PAD>', '<UNK>'] else suffix_str
        
        # Check particle tag
        if root_str.startswith('<P:'):
            particle_core = root_str[3:-1]
            return p_text + particle_core + s_text
            
        if root_str in ['<UNK>', '<PARTICLE>', '<PAD>']:
            return p_text + root_str + s_text
            
        # Canonical Morphological Realization
        if wazn_str not in ['<NONE>', '<PAD>', '<UNK>']:
            stem = self.base_tok.realize_root_and_wazn(root_str, wazn_str)
        else:
            stem = root_str
            
        return p_text + stem + s_text

    def encode_sentence(self, sentence: str) -> List[Tuple[int, int, int, int]]:
        """Encodes full space-delimited Arabic text into sequence of word-event tuples."""
        words = sentence.strip().split()
        return [self.encode_word(w) for w in words]

    def decode_sentence(self, word_tuples: List[Tuple[int, int, int, int]], apply_syntax: bool = True) -> str:
        """
        Decodes sequence of word-event tuples into space-delimited text.
        When apply_syntax=True, invokes the Basran Syntactic Realizer (محرك الضبط النحوي والإعرابي)
        to enforce jussive elisions, annexation nūn drop, hamza case-harmony, etc.
        """
        if apply_syntax and hasattr(self, 'syntactic_realizer') and self.syntactic_realizer is not None:
            return self.syntactic_realizer.realize_sentence_from_tuples(word_tuples)
        words = [self.decode_word(*t) for t in word_tuples]
        return ' '.join([w for w in words if w])

    def save_vocab(self, output_path: str):
        data = {
            "roots": self.roots_list,
            "awzan": self.awzan_list,
            "prefixes": self.canonical_prefixes,
            "suffixes": self.canonical_suffixes,
            "counts": {
                "roots": self.num_roots,
                "awzan": self.num_awzan,
                "prefixes": self.num_prefixes,
                "suffixes": self.num_suffixes
            }
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Farāhīdian Vocab] Saved vocabulary registry to: {output_path}")

if __name__ == '__main__':
    bp = '/home/absolut7/.gemini/antigravity-ide/scratch/rootformer/v12/data/rootformer_v12_arabic_blueprint.json'
    vocab = FarāhīdianMorphemicVocab(bp)
    print(f"Initialized Vocab: {vocab.num_roots} roots | {vocab.num_awzan} awzan | {vocab.num_prefixes} prefixes | {vocab.num_suffixes} suffixes")
    
    test_sent = 'العالم مكون ومحدث بالعلة الفاعلة في الوجود'
    encoded = vocab.encode_sentence(test_sent)
    print(f"\nSource Sentence: «{test_sent}»")
    print(f"Encoded Events: {len(encoded)} words")
    for w, t in zip(test_sent.split(), encoded):
        print(f"  {w:10} -> p:{t[0]:2} ({vocab.id2prefix[t[0]]}), r:{t[1]:4} ({vocab.id2root[t[1]]}), w:{t[2]:3} ({vocab.id2wazn[t[2]]}), s:{t[3]:2} ({vocab.id2suffix[t[3]]})")
        
    decoded = vocab.decode_sentence(encoded)
    print(f"Decoded Realization: «{decoded}»")
    print(f"Exact Reconstruction Match: {test_sent == decoded}")
    
    out_vocab = '/home/absolut7/.gemini/antigravity-ide/scratch/rootformer/v12/v18_next_root_morph/data/nrmp_vocab.json'
    vocab.save_vocab(out_vocab)
