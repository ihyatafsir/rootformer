#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
morph_analyzer.py
Classical Arabic Morphological Analyzer (Al-Muḥallil al-Ṣarfī al-Farāhīdī).
Provides deep morphological analysis: root extraction, template (wazn) detection,
clitic factorization, and grammatical categorization.
Anchored in the 9,114 roots of Lisān al-ʿArab and Kitāb al-ʿAyn.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Union

# Classical Closed Particles and Prepositions
CLOSED_PARTICLES = {
    'إنما': 'particle_restrictive',
    'إلى': 'preposition',
    'عن': 'preposition',
    'على': 'preposition',
    'في': 'preposition',
    'من': 'preposition',
    'حتى': 'preposition_or_conjunction',
    'إن': 'particle_conditional_or_emphatic',
    'أن': 'particle_subordinating',
    'كأن': 'particle_simulative',
    'لكن': 'particle_adversative',
    'ليت': 'particle_optative',
    'لعل': 'particle_speculative',
    'إلا': 'particle_exceptive',
    'غير': 'noun_exceptive',
    'إذا': 'particle_temporal_conditional',
    'إذ': 'particle_temporal',
    'حيث': 'adverb_spatial',
    'ثم': 'conjunction',
    'أو': 'conjunction',
    'أم': 'conjunction_interrogative',
    'بل': 'particle_digressive',
    'لا': 'particle_negative',
    'ما': 'particle_negative_or_relative',
    'لم': 'particle_jussive_negative',
    'لن': 'particle_subjunctive_negative',
    'ليس': 'verb_negative',
    'كان': 'verb_incomplete',
    'هذا': 'demonstrative_masc_sing',
    'هذه': 'demonstrative_fem_sing',
    'ذلك': 'demonstrative_distal_masc',
    'تلك': 'demonstrative_distal_fem',
    'هؤلاء': 'demonstrative_plural',
    'أولئك': 'demonstrative_distal_plural',
    'الذي': 'relative_pronoun_masc',
    'التي': 'relative_pronoun_fem',
    'الذين': 'relative_pronoun_masc_pl',
    'اللواتي': 'relative_pronoun_fem_pl',
    'هو': 'pronoun_3rd_masc_sing',
    'هي': 'pronoun_3rd_fem_sing',
    'هما': 'pronoun_3rd_dual',
    'هم': 'pronoun_3rd_masc_pl',
    'هن': 'pronoun_3rd_fem_pl',
    'أنت': 'pronoun_2nd_masc_sing',
    'أنتم': 'pronoun_2nd_masc_pl',
    'أنا': 'pronoun_1st_sing',
    'نحن': 'pronoun_1st_pl',
    'قد': 'particle_verifying',
    'سوف': 'particle_future',
    'نعم': 'particle_affirmative',
    'بلى': 'particle_affirmative',
    'إذن': 'particle_consequential',
    'لو': 'particle_hypothetical',
    'لولا': 'particle_impossibility',
    'إما': 'particle_distributive',
    'هل': 'particle_interrogative',
    'لما': 'particle_temporal_or_jussive',
    'مع': 'preposition',
    'عند': 'preposition_spatial',
    'قبل': 'adverb_temporal',
    'بعد': 'adverb_temporal',
    'بين': 'adverb_spatial',
    'حيثما': 'particle_conditional_spatial',
    'كيف': 'interrogative_manner',
    'أين': 'interrogative_location',
    'متى': 'interrogative_time',
    'كم': 'interrogative_quantity',
    'أي': 'interrogative_relative'
}

COMPOUND_PARTICLES = {
    'بلا': ('ب', 'لا'), 'بغير': ('ب', 'غير'), 'ولما': ('و', 'لما'), 'فلما': ('ف', 'لما'),
    'كلما': ('كل', 'ما'), 'فهو': ('ف', 'هو'), 'فهي': ('ف', 'هي'), 'وهو': ('و', 'هو'),
    'وهي': ('و', 'هي'), 'فإن': ('ف', 'إن'), 'وإن': ('و', 'إن'), 'لأن': ('ل', 'أن'),
    'عنها': ('عن', 'ها'), 'عليها': ('على', 'ها'), 'منها': ('من', 'ها'), 'فيها': ('في', 'ها'),
    'به': ('ب', 'ه'), 'بها': ('ب', 'ها'), 'له': ('ل', 'ه'), 'لها': ('ل', 'ها'),
    'فله': ('ف', 'له'), 'وله': ('و', 'له'), 'أنها': ('أن', 'ها'), 'إنها': ('إن', 'ها'),
    'أنه': ('أن', 'ه'), 'إنه': ('إن', 'ه'), 'أنهم': ('أن', 'هم'), 'إنهم': ('إن', 'هم'),
    'ليست': ('ليس', 'ت'), 'مما': ('من', 'ما'), 'عما': ('عن', 'ما'), 'بما': ('ب', 'ما'),
    'ولا': ('و', 'لا'), 'لأنه': ('ل', 'أنه'), 'فلأنه': ('ف', 'لأنه'), 'وإما': ('و', 'إما'),
    'فإما': ('ف', 'إما'), 'بذاته': ('ب', 'ذات'), 'بذاتها': ('ب', 'ذات'), 'لذاته': ('ل', 'ذات'),
    'لذاتها': ('ل', 'ذات'), 'بغيره': ('ب', 'غير'), 'لغيره': ('ل', 'غير'), 'عنه': ('عن', 'ه'),
    'منه': ('من', 'ه'), 'فيه': ('في', 'ه'), 'عليه': ('على', 'ه'), 'إليه': ('إلى', 'ه'),
    'إليها': ('إلى', 'ها'), 'معه': ('مع', 'ه'), 'معها': ('مع', 'ها'), 'فإنه': ('ف', 'إنه'),
    'فإنها': ('ف', 'إنها'), 'وإنه': ('و', 'إنه'), 'وإنها': ('و', 'إنها'), 'كأنه': ('كأن', 'ه'),
    'كأنها': ('كأن', 'ها'), 'كأنهم': ('كأن', 'هم'), 'كأنما': ('كأن', 'ما'), 'منا': ('من', 'نا'),
    'منهم': ('من', 'هم'), 'منكم': ('من', 'كم'), 'فلم': ('ف', 'لم'), 'ولم': ('و', 'لم'),
    'فلا': ('ف', 'لا')
}

PROCLITICS = ['فال', 'وال', 'كال', 'بال', 'لل', 'ال', 'و', 'ف', 'ب', 'ل', 'ك', 'س']
ENCLITICS = [
    'اتهم', 'اتهن', 'اتنا', 'اتكم', 'اتكن',
    'هما', 'كما', 'ونا', 'ينه', 'ينها',
    'ات', 'ون', 'ين', 'ان', 'ية', 'هم', 'هن', 'كم', 'كن', 'نا', 'ها', 'وا',
    'ة', 'ي', 'ه', 'ك', 'ت', 'ا'
]

ROOT_CANONICAL_MAP = {
    'قدم': 'تقدم', 'جزأ': 'جزا', 'جزء': 'جزا', 'بدأ': 'بدا', 'بدء': 'بدا',
    'أله': 'اله', 'ءله': 'اله', 'موه': 'موه', 'قبل': 'قبا', 'نها': 'نهي',
    'أمر': 'امر', 'أحد': 'احد', 'يقن': 'يقن', 'شفي': 'شفي', 'شفء': 'شفي',
    'غيي': 'غيي', 'جني': 'جني', 'جنى': 'جني', 'أصل': 'اصل', 'اصل': 'اصل',
    'شيء': 'شيا', 'خلو': 'خلا', 'عين': 'عيا', 'فتح': 'تفح', 'أمل': 'امل',
    'امل': 'امل', 'عضو': 'عضا', 'عضي': 'عضا', 'حشو': 'حشا', 'حشي': 'حشا',
    'هوي': 'هوا'
}

AWZAN_PATTERNS = [
    # Form X: استفعال / استفعل / يستفعل / مستفعل
    (re.compile(r'^است([^\W\d_])([^\W\d_])ا([^\W\d_])$'), 'اِسْتِفْعَال', 'verbal_noun_form_x'),
    (re.compile(r'^(?:است|يست|نست|أست|تست)([^\W\d_])([^\W\d_])([^\W\d_])$'), 'اِسْتَفْعَلَ', 'verb_form_x'),
    (re.compile(r'^مست([^\W\d_])([^\W\d_])([^\W\d_])$'), 'مُسْتَفْعِل', 'active_participle_form_x'),
    (re.compile(r'^م([^\W\d_])([^\W\d_])ا([^\W\d_])$'), 'مِفْعَال', 'noun_of_instrument'),
    (re.compile(r'^م([^\W\d_])تا([^\W\d_])$'), 'مُفْتَعِل', 'active_participle_form_viii_hollow'),
    (re.compile(r'^[ايت]([^\W\d_])تا([^\W\d_])$'), 'اِفْتَعَلَ', 'verb_form_viii_hollow'),
    (re.compile(r'^ت([^\W\d_]{4})$'), 'تَفَعْلَلَ', 'verb_quad_form_ii'),
    (re.compile(r'^ي([^\W\d_]{4})$'), 'يُتَفَعْلَلُ', 'verb_quad_form_ii'),
    (re.compile(r'^م([^\W\d_]{4})$'), 'مُتَفَعْلِل', 'participle_quad_form_ii'),
    (re.compile(r'^ان([^\W\d_])([^\W\d_])ا([^\W\d_])$'), 'اِنْفِعَال', 'verbal_noun_form_vii'),
    (re.compile(r'^ان([^\W\d_])([^\W\d_])([^\W\d_])$'), 'اِنْفَعَلَ', 'verb_form_vii'),
    (re.compile(r'^ا([^\W\d_])ت([^\W\d_])ا([^\W\d_])$'), 'اِفْتِعَال', 'verbal_noun_form_viii'),
    (re.compile(r'^ا([^\W\d_])ت([^\W\d_])([^\W\d_])$'), 'اِفْتَعَلَ', 'verb_form_viii'),
    (re.compile(r'^م([^\W\d_])ت([^\W\d_])([^\W\d_])$'), 'مُفْتَعِل', 'active_participle_form_viii'),
    (re.compile(r'^[يتأ]([^\W\d_])ت([^\W\d_])([^\W\d_])$'), 'يَفْتَعِلُ', 'verb_imperfect_form_viii'),
    (re.compile(r'^[يتنأ]ت([^\W\d_])([^\W\d_])([^\W\d_])$'), 'يَتَفَعَّلُ', 'verb_imperfect_form_v'),
    (re.compile(r'^[يتنأ]ت([^\W\d_])ا([^\W\d_])([^\W\d_])$'), 'يَتَفَاعَلُ', 'verb_imperfect_form_vi'),
    (re.compile(r'^ت([^\W\d_])([^\W\d_])([^\W\d_])$'), 'تَفَعَّلَ', 'verb_form_v'),
    (re.compile(r'^ت([^\W\d_])ا([^\W\d_])([^\W\d_])$'), 'تَفَاعَلَ', 'verb_form_vi'),
    (re.compile(r'^مت([^\W\d_])([^\W\d_])([^\W\d_])$'), 'مُتَفَعِّل', 'participle_form_v'),
    (re.compile(r'^مت([^\W\d_])ا([^\W\d_])([^\W\d_])$'), 'مُتَفَاعِل', 'participle_form_vi'),
    (re.compile(r'^([^\W\d_])و[ا]ء?([^\W\d_])([^\W\d_])$'), 'مَفَاعِل', 'broken_plural_fawail'),
    (re.compile(r'^([^\W\d_])([^\W\d_])ائ([^\W\d_])$'), 'مَفَاعِل', 'broken_plural_faail'),
    (re.compile(r'^ت([^\W\d_])([^\W\d_])ي([^\W\d_])$'), 'تَفْعِيل', 'verbal_noun_form_ii'),
    (re.compile(r'^م([^\W\d_])([^\W\d_])و([^\W\d_])$'), 'مَفْعُول', 'passive_participle_form_i'),
    (re.compile(r'^([^\W\d_])ا([^\W\d_])([^\W\d_])$'), 'فَاعِل', 'active_participle_form_i'),
    (re.compile(r'^([^\W\d_])ا([^\W\d_])([^\W\d_])ة$'), 'فَاعِلَة', 'active_participle_fem_form_i'),
    (re.compile(r'^([^\W\d_])([^\W\d_])ي([^\W\d_])$'), 'فَعِيل', 'adjective_sifah_mushabbahah'),
    (re.compile(r'^([^\W\d_])([^\W\d_])ي([^\W\d_])ة$'), 'فَعِيلَة', 'noun_fem'),
    (re.compile(r'^([^\W\d_])([^\W\d_])و([^\W\d_])$'), 'فُعُول', 'verbal_noun_or_plural'),
    (re.compile(r'^([^\W\d_])([^\W\d_])ا([^\W\d_])$'), 'فِعَال', 'verbal_noun_or_plural'),
    (re.compile(r'^([^\W\d_])([^\W\d_])ا([^\W\d_])ة$'), 'فِعَالَة', 'noun_profession_or_fem'),
    (re.compile(r'^([^\W\d_])([^\W\d_])([^\W\d_])ان$'), 'فُعْلَان', 'verbal_noun_or_plural'),
    (re.compile(r'^أ([^\W\d_])([^\W\d_])([^\W\d_])$'), 'أَفْعَل', 'elative_afal_at-tafdeel'),
    (re.compile(r'^أ([^\W\d_])([^\W\d_])ا([^\W\d_])$'), 'أَفْعَال', 'broken_plural_afal'),
    (re.compile(r'^[إا]([^\W\d_])([^\W\d_])ا([^\W\d_])$'), 'إِفْعَال', 'verbal_noun_form_iv'),
    (re.compile(r'^[يتنأ]([^\W\d_])([^\W\d_])([^\W\d_])$'), 'يَفْعَلُ', 'verb_imperfect_form_i'),
    (re.compile(r'^م([^\W\d_])([^\W\d_])([^\W\d_])$'), 'مَفْعَل', 'noun_of_place_or_time'),
    (re.compile(r'^م([^\W\d_])([^\W\d_])([^\W\d_])ة$'), 'مَفْعَلَة', 'noun_fem_place'),
    (re.compile(r'^([^\W\d_])([^\W\d_])([^\W\d_])$'), 'فَعَلَ', 'verb_perfect_form_i')
]


class ArabicMorphAnalyzer:
    """
    Sovereign Classical Arabic Morphological Analyzer.
    Provides fast, deterministic lexical decomposition into:
    (Prefixes, Consonantal Radical Root, Morphological Template Wazn, Suffixes).
    """
    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab_path = vocab_path
        self.valid_roots: Set[str] = set()
        self.particles: Set[str] = set(CLOSED_PARTICLES.keys())
        self._load_vocabulary(vocab_path)

    def _load_vocabulary(self, vocab_path: Optional[str]):
        candidates = []
        if vocab_path and Path(vocab_path).exists():
            candidates.append(Path(vocab_path))

        this_dir = Path(__file__).parent.resolve()
        candidates.extend([
            this_dir / "nrmp_vocab.json",
            this_dir / "data" / "nrmp_vocab.json",
            this_dir / "vocab.json",
            Path("/workspace/rootformer_v12/v18_next_root_morph/data/nrmp_vocab.json"),
            Path("/home/absolut7/.gemini/antigravity-ide/scratch/rootformer/v12/v18_next_root_morph/data/nrmp_vocab.json")
        ])

        data = None
        for c in candidates:
            if c.exists():
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    break
                except Exception:
                    pass

        if data:
            if "roots" in data:
                for r in data["roots"]:
                    if r.startswith("<P:"):
                        self.particles.add(r[3:-1])
                    elif not r.startswith("<"):
                        self.valid_roots.add(r)
            elif isinstance(data, dict):
                for k in data.keys():
                    if not k.startswith("<"):
                        self.valid_roots.add(k)

        # Inject canonical aliases
        for r_orig, r_can in ROOT_CANONICAL_MAP.items():
            self.valid_roots.add(r_orig)
            self.valid_roots.add(r_can)

    @staticmethod
    def normalize_arabic(text: str) -> str:
        """Strips tashkeel diacritics, tatweel, and non-Arabic punctuation."""
        if not text:
            return ""
        t = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
        t = re.sub(r'[،؛؟.:!«»\(\)\[\]\{\}…"\']', ' ', t)
        return re.sub(r'\s+', ' ', t).strip()

    def analyze_word(self, word: str) -> Dict[str, Any]:
        """
        Decomposes an Arabic word into morphological coordinates:
        root, wazn, prefix, suffix, and grammatical category.
        """
        clean = self.normalize_arabic(word)
        if not clean:
            return {
                "word": word, "normalized": "", "type": "empty",
                "prefix": "", "root": None, "wazn": None, "suffix": "",
                "is_valid_root": False
            }

        # 1. Closed Particles Check (Never shred)
        if clean in CLOSED_PARTICLES:
            return {
                "word": word, "normalized": clean, "type": CLOSED_PARTICLES[clean],
                "prefix": "", "root": f"<P:{clean}>", "wazn": "<NONE>", "suffix": "",
                "is_valid_root": False
            }

        # 2. Compound Particles Check
        if clean in COMPOUND_PARTICLES:
            p, part = COMPOUND_PARTICLES[clean]
            return {
                "word": word, "normalized": clean, "type": "compound_particle",
                "prefix": p, "root": f"<P:{part}>", "wazn": "<NONE>", "suffix": "",
                "is_valid_root": False
            }

        # 3. Direct Trilateral / Quadrilateral Root
        if len(clean) == 3 and clean in self.valid_roots:
            return {
                "word": word, "normalized": clean, "type": "root_direct_triliteral",
                "prefix": "", "root": clean, "wazn": "فَعَلَ", "suffix": "",
                "is_valid_root": True
            }
        if len(clean) == 4 and clean in self.valid_roots:
            return {
                "word": word, "normalized": clean, "type": "root_direct_quadrilateral",
                "prefix": "", "root": clean, "wazn": "فَعْلَلَ", "suffix": "",
                "is_valid_root": True
            }

        # 4. Canonical Metaphysical / Philosophical Terms
        if clean in ['ماهية', 'الماهية']:
            p = 'ال' if clean.startswith('ال') else ''
            return {
                "word": word, "normalized": clean, "type": "philosophical_term",
                "prefix": p, "root": "موه", "wazn": "فَاعِلِيَّة", "suffix": "",
                "is_valid_root": True
            }
        if clean in ['برهان', 'البرهان']:
            p = 'ال' if clean.startswith('ال') else ''
            return {
                "word": word, "normalized": clean, "type": "philosophical_term",
                "prefix": p, "root": "برهن", "wazn": "فُعْلَان", "suffix": "",
                "is_valid_root": True
            }

        # 5. Clitic Stripping & Awzān Pattern Matching Cascade
        art_prefixes = [p for p in ['فال', 'وال', 'كال', 'بال', 'لل', 'ال'] if clean.startswith(p)]
        single_prefixes = [p for p in ['و', 'ف', 'ب', 'ل', 'ك', 'س'] if clean.startswith(p)]
        candidate_prefixes = art_prefixes + [''] + single_prefixes

        cand_suffixes = sorted([s for s in ENCLITICS if clean.endswith(s) and s != ''], key=len, reverse=True)
        candidate_suffixes = [''] + cand_suffixes

        for p in candidate_prefixes:
            rem_p = clean[len(p):] if p else clean
            for s in candidate_suffixes:
                stem = rem_p[:-len(s)] if s and rem_p.endswith(s) else (rem_p if not s else '')
                if not stem or len(stem) < 2:
                    continue

                # Exact stem in roots
                if len(stem) == 3:
                    if stem in self.valid_roots:
                        return {
                            "word": word, "normalized": clean, "type": "derived_form_i",
                            "prefix": p, "root": stem, "wazn": "فَعَلَ", "suffix": s,
                            "is_valid_root": True
                        }
                    stem_bare = stem.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
                    if stem_bare in self.valid_roots:
                        return {
                            "word": word, "normalized": clean, "type": "derived_form_i",
                            "prefix": p, "root": stem_bare, "wazn": "فَعَلَ", "suffix": s,
                            "is_valid_root": True
                        }

                # Doubled Root (2-letter stem -> geminate, e.g. عل -> علل, حق -> حقق)
                if len(stem) == 2:
                    doubled = stem + stem[-1]
                    if doubled in self.valid_roots:
                        return {
                            "word": word, "normalized": clean, "type": "geminate_mudaaf",
                            "prefix": p, "root": doubled, "wazn": "فَعَلَ", "suffix": s,
                            "is_valid_root": True
                        }

                # Doubled Active Participle (تام -> تمم, عام -> عمم, خاص -> خصص)
                if len(stem) == 3 and stem[1] == 'ا':
                    cand_doubled = stem[0] + stem[2] + stem[2]
                    if cand_doubled in self.valid_roots:
                        return {
                            "word": word, "normalized": clean, "type": "active_participle_geminate",
                            "prefix": p, "root": cand_doubled, "wazn": "فَاعِل", "suffix": s,
                            "is_valid_root": True
                        }

                # Hollow imperfect / jussive (يكن -> كون, يقول -> قول, يسير -> سير)
                if len(stem) == 3 and stem[0] in ['ي', 'ت', 'ن', 'أ']:
                    cand_hollow_w = stem[1] + 'و' + stem[2]
                    if cand_hollow_w in self.valid_roots:
                        return {
                            "word": word, "normalized": clean, "type": "verb_hollow_ajwaf",
                            "prefix": p, "root": cand_hollow_w, "wazn": "يَفْعُلُ", "suffix": s,
                            "is_valid_root": True
                        }
                    cand_hollow_y = stem[1] + 'ي' + stem[2]
                    if cand_hollow_y in self.valid_roots:
                        return {
                            "word": word, "normalized": clean, "type": "verb_hollow_ajwaf",
                            "prefix": p, "root": cand_hollow_y, "wazn": "يَفْعِلُ", "suffix": s,
                            "is_valid_root": True
                        }
                    cand_gem = stem[1] + stem[2] + stem[2]
                    if cand_gem in self.valid_roots:
                        return {
                            "word": word, "normalized": clean, "type": "verb_geminate",
                            "prefix": p, "root": cand_gem, "wazn": "يَفْعُلُ", "suffix": s,
                            "is_valid_root": True
                        }

                # Mithāl Wāwī (يجب -> وجب, يجد -> وجد, يصف -> وصف)
                if len(stem) == 3 and stem[0] in ['ي', 'ت', 'ن', 'أ']:
                    cand_mithal = 'و' + stem[1:]
                    if cand_mithal in self.valid_roots:
                        return {
                            "word": word, "normalized": clean, "type": "verb_assimilated_mithal",
                            "prefix": p, "root": cand_mithal, "wazn": "يَعِلُ", "suffix": s,
                            "is_valid_root": True
                        }

                # Trilateral Awzān Matching
                for pat, wazn_name, sem_type in AWZAN_PATTERNS:
                    m = pat.match(stem)
                    if m:
                        rads = list(m.groups())
                        if len(rads) == 1:
                            cand_root = rads[0]
                        elif len(rads) == 2:
                            cand_root = rads[0] + 'و' + rads[1]
                        else:
                            cand_root = ''.join(rads)

                        cand_root = ROOT_CANONICAL_MAP.get(cand_root, cand_root)
                        if cand_root in self.valid_roots:
                            return {
                                "word": word, "normalized": clean, "type": sem_type,
                                "prefix": p, "root": cand_root, "wazn": wazn_name, "suffix": s,
                                "is_valid_root": True
                            }

        # Fallback heuristic
        return {
            "word": word, "normalized": clean, "type": "unresolved",
            "prefix": "", "root": clean[:3] if len(clean) >= 3 else clean, "wazn": "<UNK>", "suffix": "",
            "is_valid_root": clean[:3] in self.valid_roots if len(clean) >= 3 else False
        }

    def analyze_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """Analyzes all words in an Arabic sentence."""
        words = sentence.strip().split()
        return [self.analyze_word(w) for w in words]

    def extract_roots(self, sentence: str) -> List[str]:
        """Extracts the underlying consonantal roots from a sentence."""
        res = self.analyze_sentence(sentence)
        return [r["root"] for r in res if r["root"]]
