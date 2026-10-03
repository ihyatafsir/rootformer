#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
universal_farahidi_transduction.py
Universal Cross-Lingual Transduction Bridge based on Sībawayh's Articulatory Space
and Al-Khalīl's Combinatoric Phonetics.
Enables 'Translating Without Translating':
Bridges continuous Latin character phonetic trajectories to discrete Arabic consonantal roots
and vice-versa through minimal articulatory energy paths in 17D space.
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np

from sibawayh_phonetics import (
    get_vector, sibawayh_distance,
    find_nearest_arabic_neighbor, find_nearest_latin_neighbor,
    SIBAWAYH_ARABIC_VECTORS, SIBAWAYH_LATIN_VECTORS
)
from khalil_combinatorics import evaluate_farahidi_phonotactics, FARAHIDI_RANK
from ibn_jinni_ishtiqaq import IbnJinniIshtiqaqEngine

class UniversalFarahidiTransductionBridge:
    def __init__(self):
        self.ishtiqaq_engine = IbnJinniIshtiqaqEngine()
        
    def word_to_phonetic_trajectory(self, text: str) -> np.ndarray:
        """Converts an English or Arabic word into a sequence of 17D vectors."""
        vecs = []
        for ch in text.lower():
            v = get_vector(ch)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return np.zeros((1, 17))
        return np.array(vecs)
        
    def compute_trajectory_energy(self, traj: np.ndarray) -> float:
        """Computes the internal kinetic energy (effort) of an articulatory trajectory."""
        if len(traj) <= 1:
            return 0.0
        diffs = traj[1:] - traj[:-1]
        energy = np.sum(diffs ** 2)
        return float(energy)

    def transduce_latin_to_farahidi_skeleton(self, latin_word: str) -> Dict[str, any]:
        """
        Transduces a Latin word into its underlying Farāhīdian phonetic skeleton:
        Maps each consonant to its nearest Sībawayh articulatory locus,
        filters non-consonantal glides/vowels to isolate the consonantal radicals,
        and evaluates phonotactic compatibility.
        """
        raw_chars = [c for c in latin_word.lower() if c.isalpha()]
        mapped_ar = []
        consonants = []
        vowels = {'a', 'e', 'i', 'o', 'u'}
        
        for c in raw_chars:
            ar_c, dist = find_nearest_arabic_neighbor(c)
            mapped_ar.append((c, ar_c, dist))
            if c not in vowels:
                consonants.append(ar_c)
                
        # Consolidate adjacent geminates
        dedup_radicals = []
        for c in consonants:
            if not dedup_radicals or dedup_radicals[-1] != c:
                dedup_radicals.append(c)
                
        # Candidate 3-radical or 4-radical root
        candidate_radicals = tuple(dedup_radicals[:4] if len(dedup_radicals) >= 4 else dedup_radicals[:3])
        score, viols, classification = evaluate_farahidi_phonotactics(candidate_radicals) if candidate_radicals else (0.0, [], "mumtani")
        
        return {
            "latin_input": latin_word,
            "phonetic_transcription": "".join(m[1] for m in mapped_ar),
            "extracted_consonants": "".join(dedup_radicals),
            "candidate_farahidi_root": "".join(candidate_radicals),
            "phonotactic_score": score,
            "violations": viols,
            "character_mappings": mapped_ar
        }

    def project_root_to_latin_phonetics(self, arabic_root: str) -> Dict[str, any]:
        """
        Projects an Arabic root to its Latin phonetic realization in Sībawayh's space.
        """
        rads = list(arabic_root.strip())
        latin_proj = []
        for r in rads:
            lat_c, dist = find_nearest_latin_neighbor(r)
            latin_proj.append((r, lat_c, dist))
            
        return {
            "arabic_root": arabic_root,
            "latin_skeleton": "".join(m[1] for m in latin_proj),
            "details": latin_proj
        }

if __name__ == '__main__':
    print("=" * 75)
    print(">>> UNIVERSAL FARAHIDI TRANSDUCTION BRIDGE VERIFICATION <<<")
    print("=" * 75)
    
    bridge = UniversalFarahidiTransductionBridge()
    
    # Test 1: Projecting Latin concepts into Farahidian roots
    test_latin = ["beginning", "reason", "intellect", "substance", "accident", "peace", "necessary"]
    for word in test_latin:
        res = bridge.transduce_latin_to_farahidi_skeleton(word)
        print(f"Latin: '{word:12s}' -> Transcription: {res['phonetic_transcription']:15s} | Extracted Root: {res['candidate_farahidi_root']:5s} (Score: {res['phonotactic_score']:.2f})")
        
    # Test 2: Projecting Arabic roots into Latin skeletons
    print("\n--- Arabic Roots Projected to Latin Phonetics ---")
    test_roots = ["حدث", "عقل", "وجب", "نظر", "سلم", "جوهر", "عرض"]
    for r in test_roots:
        res = bridge.project_root_to_latin_phonetics(r)
        print(f"Arabic Root: [{r}] -> Latin Phonetic Skeleton: '{res['latin_skeleton']}'")
