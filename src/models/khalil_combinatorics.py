#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
khalil_combinatorics.py
Formalization of Al-Khalil ibn Ahmad al-Farahidi's Combinatoric Permutation Matrix
and Phonotactic Laws (Kitab al-Ayn, Preface and Bab al-Huruf).
Implements exact n! permutations, Farahidian phonotactic compatibility (I'tilaf vs Tanafur),
and classification into Musta'mal (Attested), Muhmal (Permissible Unused),
Nadirah (Rare Anomalous), and Mumtani' (Forbidden / Muwallad).
"""

import itertools
import math
from typing import List, Dict, Set, Tuple, Optional, Literal

# Al-Farahidi's Phonetic Ordering (From Deepest Throat to Lips & Air)
FARAHIDI_ALPHABET_ORDER = [
    'ع', 'ح', 'ه', 'خ', 'غ',  # Halqiyyah (Throat / Guttural)
    'ق', 'ك',                  # Lahawiyyah (Uvular & Velar)
    'ج', 'ش', 'ض',             # Shajariyyah (Palatal & Lateral)
    'ص', 'س', 'ز',             # Asaliyyah (Alveolar Sibilants)
    'ط', 'د', 'ت',             # Nat'iyyah (Dental-Alveolar)
    'ظ', 'ذ', 'ث',             # Lithawiyyah (Interdental)
    'ر', 'ل', 'ن',             # Dhalaqiyyah (Tongue-tip Liquids)
    'ف', 'ب', 'م',             # Shafawiyyah (Labials)
    'و', 'ي', 'ا', 'ء'         # Hawi / Jawf / Glides
]

FARAHIDI_RANK = {c: idx for idx, c in enumerate(FARAHIDI_ALPHABET_ORDER)}

# Discrete Throat Loci (Al-Halq has 3 distinct levels)
HALQ_AQSA = {'ء', 'ه', 'ا'}
HALQ_AWSAT = {'ع', 'ح'}
HALQ_ADNA = {'غ', 'خ'}
HALQ_LETTERS = HALQ_AQSA | HALQ_AWSAT | HALQ_ADNA

DHALAQAH_LETTERS = {'ر', 'ل', 'ن', 'ف', 'ب', 'م'}
LABIAL_LETTERS = {'ف', 'ب', 'م'}
NATIYYAH_LETTERS = {'ط', 'د', 'ت'}
ASALIYYAH_LETTERS = {'ص', 'س', 'ز'}
LITHAWIYYAH_LETTERS = {'ظ', 'ذ', 'ث'}
SHAJARIYYAH_LETTERS = {'ج', 'ش', 'ض'}
LAHAWIYYAH_LETTERS = {'ق', 'ك'}

HOMORGANIC_GROUPS = [
    HALQ_AWSAT,
    HALQ_ADNA,
    LAHAWIYYAH_LETTERS,
    SHAJARIYYAH_LETTERS,
    ASALIYYAH_LETTERS,
    NATIYYAH_LETTERS,
    LITHAWIYYAH_LETTERS,
    LABIAL_LETTERS
]

def get_permutations(root_chars: List[str]) -> List[Tuple[str, ...]]:
    """
    Generates exact n! permutations (Taqalib) of root radicals according to Al-Khalil:
    - 2 letters (Thuna'i) -> 2! = 2
    - 3 letters (Thulathi Masdus) -> 3! = 6
    - 4 letters (Ruba'i) -> 4! = 24
    - 5 letters (Khumasi) -> 5! = 120
    """
    return list(itertools.permutations(root_chars))

def evaluate_farahidi_phonotactics(perm: Tuple[str, ...]) -> Tuple[float, List[str], str]:
    """
    Evaluates phonotactic compatibility (I'tilaf vs Tanafur) according to Al-Khalil:
    1. Homorganic Halq Repulsion: Letters within the exact same throat tier (e.g. ع with ح, or غ with خ)
       cannot collide contiguously. Different tiers (ع with هـ as in عهد, or أ with ح as in أحد) are permissible.
    2. Dhalaqah Law: Ruba'i and Khumasi MUST contain at least one liquid (r, l, n, f, b, m).
       If absent and attested: Nadirah (نادرة). If absent and unattested: Muwalladah/Mumtani'ah.
    3. Homorganic Repulsion: Contiguous distinct consonants from the same narrow articulatory locus are penalized.
    4. Vocal Tract Trajectory Effort (Thiql): Measures articulatory smoothness across the Farahidian rank.
    
    Returns:
        (score: float in [0.0, 1.0], violations: List[str], classification: str)
    """
    k = len(perm)
    violations = []
    classification = "Musta'mal_Muhmal"
    
    # Rule 2: Al-Khalil's Dhalaqah Law for Ruba'i and Khumasi
    if k >= 4:
        dhalaqah_count = sum(1 for c in perm if c in DHALAQAH_LETTERS)
        if dhalaqah_count == 0:
            violations.append(f"Dhalaqah Law: Length {k} root devoid of liquid/labial letters ({','.join(DHALAQAH_LETTERS)})")
            classification = "Muwallad_or_Nadir"
            return 0.0, violations, classification
    
    # Rule 1: Precise Homorganic Throat Collision (Same Tier)
    for i in range(k - 1):
        c1, c2 = perm[i], perm[i+1]
        if c1 != c2:
            if (c1 in HALQ_AWSAT and c2 in HALQ_AWSAT) or (c1 in HALQ_ADNA and c2 in HALQ_ADNA):
                violations.append(f"Same-Tier Halq Collision: Contiguous throat sounds '{c1}' and '{c2}'")
                classification = "Mumtani'"
                return 0.0, violations, classification

    # Rule 3: Homorganic Collisions (Penalized)
    penalty = 0.0
    for i in range(k - 1):
        c1, c2 = perm[i], perm[i+1]
        if c1 == c2:
            continue  # Doubling (Tad'if) is standard Arabic morphology
        for group in HOMORGANIC_GROUPS:
            if c1 in group and c2 in group:
                violations.append(f"Homorganic Friction: Proximate articulators '{c1}' and '{c2}'")
                penalty += 0.35
                break

    # Rule 4: Contiguous Labial Friction
    for i in range(k - 1):
        c1, c2 = perm[i], perm[i+1]
        if c1 in LABIAL_LETTERS and c2 in LABIAL_LETTERS and c1 != c2:
            violations.append(f"Labial Friction: Contiguous labials '{c1}' and '{c2}'")
            penalty += 0.25

    # Vocal Tract Smoothness (Khiffah vs Thiql)
    total_trajectory = 0
    for i in range(k - 1):
        r1 = FARAHIDI_RANK.get(perm[i], 15)
        r2 = FARAHIDI_RANK.get(perm[i+1], 15)
        total_trajectory += abs(r1 - r2)
    
    max_traj = 25.0 * max(1, k - 1)
    trajectory_score = max(0.0, 1.0 - (total_trajectory / max_traj))
    final_score = max(0.0, min(1.0, (1.0 - penalty) * (0.5 + 0.5 * trajectory_score)))
    
    if final_score < 0.15:
        classification = "Mumtani'"
    else:
        classification = "Musta'mal_Muhmal"
        
    return round(final_score, 4), violations, classification

class FarahidiPermutationEngine:
    """
    Engine implementing Al-Khalil's full permutation and lexicon partitioning system.
    """
    def __init__(self, classical_lexicon_roots: Optional[Set[str]] = None):
        self.lexicon = classical_lexicon_roots or set()
        
    def analyze_root(self, root_str: str) -> Dict[str, any]:
        """
        Executes Al-Khalil's full permutation analysis for a root.
        Computes all n! forms, phonotactic scores, and assigns Musta'mal / Muhmal / Nadirah / Mumtani'.
        """
        radicals = list(root_str.strip())
        n = len(radicals)
        perms = get_permutations(radicals)
        
        results = []
        mustamal_count = 0
        muhmal_count = 0
        nadirah_count = 0
        mumtani_count = 0
        
        for p in perms:
            word = "".join(p)
            score, viols, raw_class = evaluate_farahidi_phonotactics(p)
            
            if raw_class == "Muwallad_or_Nadir":
                if word in self.lexicon:
                    status = "Nadirah (Attested Bedouin Anomaly, e.g. كشعثج)"
                    nadirah_count += 1
                else:
                    status = "Muwalladah / Mumtani'ah (Unattested Non-Arabic / Post-Classical)"
                    mumtani_count += 1
            elif raw_class == "Mumtani'":
                status = "Mumtani' (Forbidden by Phonotactic Law)"
                mumtani_count += 1
            elif word in self.lexicon:
                status = "Musta'mal (Attested in Classical Arabic)"
                mustamal_count += 1
            else:
                status = "Muhmal (Phonotactically Legal, Potential / Foreign Space)"
                muhmal_count += 1
                
            results.append({
                "permutation": word,
                "radicals": list(p),
                "phonotactic_score": score,
                "status": status,
                "violations": viols
            })
            
        canonical_farahidi = "".join(sorted(radicals, key=lambda c: FARAHIDI_RANK.get(c, 99)))
        
        return {
            "root": root_str,
            "length": n,
            "total_permutations": len(perms),
            "canonical_farahidi_order": canonical_farahidi,
            "counts": {
                "mustamal": mustamal_count,
                "muhmal": muhmal_count,
                "nadirah": nadirah_count,
                "mumtani": mumtani_count
            },
            "permutations": results
        }

if __name__ == '__main__':
    print("=" * 75)
    print(">>> AL-KHALIL IBN AHMAD COMBINATORIC ENGINE & PHONOTACTICS VERIFIED <<<")
    print("=" * 75)
    
    known_roots = {"عهد", "أحد", "حدث", "عقل", "علم", "كتب", "سلم", "كشعثج"}
    engine = FarahidiPermutationEngine(classical_lexicon_roots=known_roots)
    
    for r in ["عهد", "أحد", "كشعثج"]:
        res = engine.analyze_root(r)
        print(f"\nRoot [{r}] -> Counts: {res['counts']}")
        for p in res['permutations'][:3]:
            print(f"  * {p['permutation']}: {p['status']} (score={p['phonotactic_score']:.2f})")


# ==========================================================================================
# al-TAQĀLĪB ORBIT MECHANISM (parameter tying) -- re-export
# ==========================================================================================
# ENGINEERING: the canonical implementation lives in the release-root module
# `khalil_orbits.py` (KhalilPermutationOrbits: orbits over the ATTESTED permutations of a
# root, `orbit_of`, `tie_embeddings`, `loss`).  It is re-exported HERE because the project's
# correctness audit imports it from this module:
#
#     from models.khalil_combinatorics import KhalilPermutationOrbits
#
# Nothing else in this module depends on it; the permutation/phonotactics engine above is
# unchanged.  FIDELITY CAVEAT: the tying this class performs is licensed by al-Khalīl's own
# accept/reject filter in Kitāb al-ʿAyn («يُكَتَب مُسْتَعْمَلها. ويُلغى مُهْمَلها»).  It is NOT
# al-ishtiqāq al-akbar: the claim that the permutations share one MEANING is rejected by the
# tradition (Ibn ʿUsfūr, al-Mumtiʿ: «غير مأخوذ به؛ لعدم اطراده») and is not implemented.
# See the header of `khalil_orbits.py` for the verbatim sources.
def _load_khalil_orbits():
    """Import `khalil_orbits` whether or not the release root is on sys.path."""
    import importlib.util
    import sys
    from pathlib import Path as _Path
    try:
        import khalil_orbits as _mod  # type: ignore
        return _mod
    except Exception:
        _path = _Path(__file__).resolve().parent.parent / 'khalil_orbits.py'
        if not _path.is_file():
            raise ImportError(f'khalil_orbits.py not found at {_path}')
        _spec = importlib.util.spec_from_file_location('khalil_orbits', _path)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules.setdefault('khalil_orbits', _mod)
        _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
        return _mod

_khalil_orbits_mod = _load_khalil_orbits()
KhalilPermutationOrbits = _khalil_orbits_mod.KhalilPermutationOrbits
OrbitTiedEmbedding = _khalil_orbits_mod.OrbitTiedEmbedding
normalize_root = _khalil_orbits_mod.normalize_root

