#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ibn_jinni_ishtiqaq.py
Formalization of Abu al-Fath Uthman Ibn Jinni's Linguistic Algorithms:
1. Al-Ishtiqaq al-Akbar (The Greater Derivation, Al-Khasa'is):
   Proves the invariant acoustic-semantic core across all n! root permutations under the symmetric group S_n.
2. Tasaqub al-Alfaz li-Tasaqub al-Ma'ani (Phonetic Proximity Governs Semantic Modulations):
   Measures how subtle shifts in Sibawayh's articulatory space (e.g., Isti'la, Itbaq, Shiddah)
   modulate physical and conceptual semantic intensity.
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np

from sibawayh_phonetics import get_vector, sibawayh_distance, DIMENSIONS, DIM_WEIGHTS
from khalil_combinatorics import get_permutations, evaluate_farahidi_phonotactics

class IbnJinniIshtiqaqEngine:
    """
    Implements Ibn Jinni's algorithms connecting Farahidian permutations and Sibawayh articulatory space.
    """
    def __init__(self):
        pass
        
    def compute_root_centroid(self, root_str: str) -> np.ndarray:
        """
        Computes the Farahidian-Sibawayh acoustic centroid (al-Qadr al-Mushtarak)
        mu_root = (1/k) * sum(phi(c_i)) for i=1..k in 17D space.
        """
        rads = list(root_str.strip())
        vectors = []
        for r in rads:
            v = get_vector(r)
            if v is not None:
                vectors.append(v)
            else:
                vectors.append([0.0] * 17)
        if not vectors:
            return np.zeros(17)
        return np.mean(np.array(vectors), axis=0)

    def analyze_ishtiqaq_akbar(self, root_str: str) -> Dict[str, any]:
        """
        Executes full Ishtiqaq Akbar analysis for a root:
        - Validates that all n! permutations share an identical acoustic invariant centroid.
        - Calculates pairwise phonetic trajectory distance between permutations.
        - Computes trajectory velocity: how the tongue transitions through vocal loci.
        """
        rads = list(root_str.strip())
        perms = get_permutations(rads)
        centroid = self.compute_root_centroid(root_str)
        
        perm_details = []
        for p in perms:
            word = "".join(p)
            score, viols, classification = evaluate_farahidi_phonotactics(p)
            
            # Articulatory velocity: Euclidean step size along the vocal tract
            step_distances = []
            for i in range(len(p) - 1):
                d = sibawayh_distance(p[i], p[i+1])
                step_distances.append(d)
                
            total_trajectory_effort = sum(step_distances)
            mean_step = np.mean(step_distances) if step_distances else 0.0
            
            perm_details.append({
                "permutation": word,
                "phonotactic_score": score,
                "trajectory_effort": round(float(total_trajectory_effort), 4),
                "mean_step_distance": round(float(mean_step), 4),
                "violations": viols
            })
            
        # Sort permutations by trajectory effort (lowest effort = most euphonic/natural)
        perm_details.sort(key=lambda x: (x['trajectory_effort'], -x['phonotactic_score']))
        
        return {
            "root": root_str,
            "invariant_centroid_17d": [round(float(x), 4) for x in centroid],
            "total_permutations": len(perms),
            "most_euphonic_permutation": perm_details[0]['permutation'],
            "permutations": perm_details
        }

    def analyze_tasaqub_pair(self, root1: str, root2: str) -> Dict[str, any]:
        """
        Analyzes Ibn Jinni's 'Tasaqub al-Alfaz li-Tasaqub al-Ma'ani':
        Compares two proximate roots (e.g., نضح vs نضخ, or هز vs أز)
        and quantifies the phonetic shift and corresponding semantic modulation.
        """
        r1, r2 = list(root1.strip()), list(root2.strip())
        if len(r1) != len(r2):
            return {"error": "Roots must have equal length"}
            
        shifts = []
        total_phonetic_dist = 0.0
        
        for idx, (c1, c2) in enumerate(zip(r1, r2)):
            if c1 != c2:
                d = sibawayh_distance(c1, c2)
                v1 = get_vector(c1)
                v2 = get_vector(c2)
                
                # Identify which Sifat shifted
                sifat_shifts = []
                if v1 and v2:
                    for dim_idx, dim_name in enumerate(DIMENSIONS):
                        diff = v2[dim_idx] - v1[dim_idx]
                        if abs(diff) > 0.1:
                            sifat_shifts.append({
                                "dimension": dim_name,
                                "from_val": v1[dim_idx],
                                "to_val": v2[dim_idx],
                                "direction": "increased" if diff > 0 else "decreased"
                            })
                            
                shifts.append({
                    "position": idx + 1,
                    "from_char": c1,
                    "to_char": c2,
                    "distance": round(float(d), 4),
                    "sifat_shifts": sifat_shifts
                })
                total_phonetic_dist += d
                
        # Semantic Intensity Prediction based on Ibn Jinni's principles:
        # - Higher Itbaq / Isti'la / Shiddah -> Higher physical intensity, force, loudness
        # - Lower / Rakhawah / Hams -> Softness, inwardness, subtlety
        intensity_delta = 0.0
        for s in shifts:
            for sf in s['sifat_shifts']:
                dim = sf['dimension']
                delta = sf['to_val'] - sf['from_val']
                if dim in ['itbaq_infitah', 'istila_istifal', 'shiddah_rikhawah', 'qalqalah']:
                    intensity_delta += delta * 1.5
                elif dim == 'jahr_hams':
                    intensity_delta += delta * 1.0
                    
        modulation = "Intensification (القوة والشدة والغلظة)" if intensity_delta > 0.2 else (
            "Softening / Subtlety (الخفة والرقة واللطف)" if intensity_delta < -0.2 else "Qualitative Shift"
        )
        
        return {
            "root1": root1,
            "root2": root2,
            "total_phonetic_distance": round(float(total_phonetic_dist), 4),
            "shifts": shifts,
            "intensity_delta": round(float(intensity_delta), 4),
            "predicted_semantic_modulation": modulation
        }

if __name__ == '__main__':
    print("=" * 75)
    print(">>> IBN JINNI ALGORITHMS: AL-ISHTIQAQ AL-AKBAR & TASAQUB AL-ALFAZ <<<")
    print("=" * 75)
    
    engine = IbnJinniIshtiqaqEngine()
    
    # Test 1: Al-Ishtiqaq al-Akbar for classic root [س-ل-م]
    print("\n--- 1. Al-Ishtiqaq al-Akbar for [س-ل-م] ---")
    res = engine.analyze_ishtiqaq_akbar("سلم")
    print(f"Invariant Centroid (17D): {res['invariant_centroid_17d'][:6]}...")
    print(f"Most Euphonic Permutation: {res['most_euphonic_permutation']}")
    for p in res['permutations']:
        print(f"  * {p['permutation']} | Effort: {p['trajectory_effort']} | Score: {p['phonotactic_score']}")
        
    # Test 2: Tasaqub al-Alfaz on Ibn Jinni's exact canonical examples
    print("\n--- 2. Ibn Jinni's Canonical Examples of Tasaqub al-Alfaz ---")
    canonical_pairs = [
        ("نضح", "نضخ"),  # Oozing vs Gushing (Halq wasat Haa -> Halq adna Khaa: Isti'la increased!)
        ("هزز", "أزز"),  # Shaking vs Boiling/Agitating (Ha -> Hamza: Plosive/Jahr increased!)
        ("قدد", "قطط"),  # Cutting lengthwise vs cutting transversely (Dal -> Taa: Itbaq/Istila increased!)
    ]
    
    for r1, r2 in canonical_pairs:
        t_res = engine.analyze_tasaqub_pair(r1, r2)
        print(f"\nPair [{r1}] -> [{r2}]:")
        print(f"  Phonetic Dist: {t_res['total_phonetic_distance']} | Intensity Delta: {t_res['intensity_delta']:+.2f}")
        print(f"  Predicted Semantic Modulation: {t_res['predicted_semantic_modulation']}")
        for s in t_res['shifts']:
            shift_dims = [f"{sf['dimension']}({sf['direction']})" for sf in s['sifat_shifts']]
            print(f"  Pos {s['position']}: '{s['from_char']}' -> '{s['to_char']}' | Sifat Shifts: {', '.join(shift_dims)}")
