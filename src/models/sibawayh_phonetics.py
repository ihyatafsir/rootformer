#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sibawayh_phonetics.py
Formalization of Sibawayh's 17-Dimensional Phonetic Vector Space (Al-Kitab, Ch. 230+)
Unifies Arabic phonemes, Sibawayh's branch letters (Huruf Far'iyyah),
and foreign articulatory mappings (Latin phonemes) into a rigorous geometric metric space.

Scholastic Note on Al-Khalīl vs. Sībawayh Ordering:
- Al-Khalīl ibn Aḥmad (Kitāb al-ʿAyn): Organizes the dictionary starting from ʿAyn (ع),
  ascending to the lips (ف ب م), and placing the weak/hollow letters (و ي ا ء) at the end.
- Sībawayh (Al-Kitāb): Organizes phonetically from deepest physiological locus (Aqṣā al-Ḥalq)
  beginning with Hamza (ء), Hāʾ (هـ), Alif (ا), through middle/upper throat, tongue, teeth, to lips.
Both systems are preserved and cross-referenced herein.
"""

import math
from typing import Dict, List, Tuple, Optional

# 17 Phonetic Dimensions of Sibawayh
DIMENSIONS = [
    "makhraj_depth",      # 0.0 (Lips / Shafatan) -> 1.0 (Deepest Throat / Aqsa al-Halq)
    "jahr_hams",          # +1.0 (Voiced / Majhur), -1.0 (Voiceless / Mahmus)
    "shiddah_rikhawah",   # +1.0 (Plosive / Shadid), 0.0 (Intermediate / Bayni), -1.0 (Fricative / Rakhw)
    "istila_istifal",     # +1.0 (Elevated / Musta'li), 0.0 (Lowered / Mustafil)
    "itbaq_infitah",      # +1.0 (Velarized/Pharyngealized / Mutbaq), 0.0 (Open / Munfatih)
    "dhalaqah_ismat",     # +1.0 (Liquid/Tip/Lip / Dhalaqiyyah), 0.0 (Solid / Musmatah)
    "safir",              # +1.0 (Sibilance / Safir: ص, ز, س)
    "qalqalah",           # +1.0 (Burst / Qalqalah: ق, ط, ب, ج, د)
    "inhiraf",            # +1.0 (Lateral deflection: ل, ر)
    "takrir",             # +1.0 (Trill / Vibration: ر)
    "tafashshi",          # +1.0 (Diffusion: ش)
    "istitalah",          # +1.0 (Elongation: ض)
    "ghunnah",            # +1.0 (Nasality: ن, م)
    "lin_madd",           # +1.0 (Approximant / Glide: و, ي, ا)
    "coronal",            # +1.0 (Tongue tip/blade: ت, د, ط, ث, ذ, ظ, س, ز, ص, ن, ل, ر)
    "dorsal",             # +1.0 (Tongue body/back: ك, ق, ج, ش, ي)
    "labial",             # +1.0 (Lips: ف, ب, م, و)
]

# Sībawayh's 16 Discrete Makhārij (Al-Kitāb, Vol. 4, p. 431)
SIBAWAYH_MAKHRAJ_ORDINAL = {
    1: {"name": "أقصى الحلق", "letters": ['ء', 'ه', 'ا'], "depth": 1.00},
    2: {"name": "أوسط الحلق", "letters": ['ع', 'ح'], "depth": 0.88},
    3: {"name": "أدنى الحلق", "letters": ['غ', 'خ'], "depth": 0.75},
    4: {"name": "أقصى اللسان فوق", "letters": ['ق'], "depth": 0.65},
    5: {"name": "أقصى اللسان أسفل من القاف", "letters": ['ك'], "depth": 0.58},
    6: {"name": "وسط اللسان بينه وبين وسط الحنك", "letters": ['ج', 'ش', 'ي'], "depth": 0.48},
    7: {"name": "أول حافة اللسان وما يليها من الأضراس", "letters": ['ض'], "depth": 0.40},
    8: {"name": "حافة اللسان من أدناها إلى منتهى طرف اللسان", "letters": ['ل'], "depth": 0.32},
    9: {"name": "طرف اللسان بينه وبين ما فويق الثنايا", "letters": ['ن'], "depth": 0.28},
    10: {"name": "مخرج النون أدخل في ظهر اللسان", "letters": ['ر'], "depth": 0.26},
    11: {"name": "بين طرف اللسان وأصول الثنايا", "letters": ['ط', 'د', 'ت'], "depth": 0.22},
    12: {"name": "بين طرف اللسان وفويق الثنايا (الأسلية)", "letters": ['ص', 'ز', 'س'], "depth": 0.18},
    13: {"name": "بين طرف اللسان وأطراف الثنايا (اللثوية)", "letters": ['ظ', 'ذ', 'ث'], "depth": 0.14},
    14: {"name": "باطن الشفة السفلية وأطراف الثنايا العلى", "letters": ['ف'], "depth": 0.08},
    15: {"name": "ما بين الشفتين", "letters": ['ب', 'م', 'و'], "depth": 0.02},
    16: {"name": "الخياشيم (مخرج النون الخفيفة والغنة)", "letters": ['ن_خفيفة'], "depth": 0.28},
}

# Sibawayh Standard 29 Arabic Consonants & Glides
SIBAWAYH_ARABIC_VECTORS = {
    # Al-Halq (Throat)
    'ء': [1.00,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'أ': [1.00,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'إ': [1.00,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'آ': [1.00,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'ؤ': [1.00,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    'ئ': [1.00,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
    'ه': [0.98, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'ع': [0.88,  1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'ح': [0.85, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'غ': [0.75,  1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'خ': [0.72, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    # Aqsa & Wasat al-Lisan
    'ق': [0.65,  1.0,  1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'ك': [0.58, -1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'ج': [0.48,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'ش': [0.46, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'ي': [0.44,  1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    # Haffah & Taraf al-Lisan
    'ض': [0.40,  1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'ل': [0.32,  1.0,  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'ن': [0.28,  1.0,  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    'ر': [0.26,  1.0,  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    # Alveolar / Dental (Nat'iyyah & Asaliyyah & Lithawiyyah)
    'ط': [0.22,  1.0,  1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'د': [0.22,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'ت': [0.22, -1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'ص': [0.18, -1.0, -1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'ز': [0.18,  1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'س': [0.18, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'ظ': [0.14,  1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'ذ': [0.14,  1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    'ث': [0.14, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    # Al-Shafatayn (Lips)
    'ف': [0.08, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    'ب': [0.02,  1.0,  1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    'م': [0.02,  1.0,  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    'و': [0.02,  1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    'ا': [0.95,  1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'ة': [0.98, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'ى': [0.44,  1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
}

# The 6 Approved Secondary Branch Letters (Ḥurūf Farʿiyyah Mustaḥsanah - Sībawayh p. 431)
SIBAWAYH_BRANCH_VECTORS = {
    'ن_خفيفة':   [0.28, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0], # Nasalized nun in ikhfa'
    'ء_بين_بين': [0.98, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], # Softened glottal stop (tashil)
    'ا_إمالة':   [0.70, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0], # Fronted alif toward ya
    'ش_كالجيم':  [0.47, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # Voiced palatal fricative
    'ص_كالزاي':  [0.18, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # Voiced velarized sibilant (ishmam)
    'ا_تفخيم':   [0.90, 1.0, -0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # Pharyngealized alif (Hijazi)
}

# The 7 Substandard / Dialectal Letters (Ḥurūf Mustarẓalah - Sībawayh p. 431)
SIBAWAYH_SUBSTANDARD_VECTORS = {
    'ك_بين_الجيم': [0.53, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # Persian / Yemeni G
    'ج_كالشين':    [0.47, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # Devoiced Jim
    'ض_ضعيفة':    [0.35, 1.0, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0], # Weak un-elongated Dad
    'ص_كالسين':   [0.18, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # De-velarized Sad
    'ط_كالتاء':   [0.22, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # De-velarized Taa
    'ظ_كالثاء':   [0.14, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # De-velarized Zhaa
    'ب_كالفاء':   [0.05, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Latin P/V substitution
}

# Latin Articulatory Mappings in Sibawayh's Space
SIBAWAYH_LATIN_VECTORS = {
    'a': [0.85,  1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # Open vowel ~ Alif
    'b': [0.02,  1.0,  1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Bilabial voiced stop ~ Ba
    'c': [0.58, -1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # Velar voiceless stop ~ Kaf
    'd': [0.22,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # Alveolar voiced stop ~ Dal
    'e': [0.50,  1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0], # Mid-front vowel ~ Imalah Ya
    'f': [0.08, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Labiodental voiceless ~ Fa
    'g': [0.58,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # Velar voiced stop ~ Kaf al-Mu'jamah
    'h': [0.98, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Glottal voiceless ~ Ha
    'i': [0.44,  1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], # High-front vowel ~ Ya
    'j': [0.48,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # Palatal affricate ~ Jim
    'k': [0.58, -1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # Velar voiceless ~ Kaf
    'l': [0.32,  1.0,  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # Lateral liquid ~ Lam
    'm': [0.02,  1.0,  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], # Bilabial nasal ~ Meem
    'n': [0.28,  1.0,  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], # Alveolar nasal ~ Noon
    'o': [0.15,  1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.8], # Back rounded vowel ~ Waw
    'p': [0.02, -1.0,  1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Bilabial voiceless stop ~ Ba/Fa
    'q': [0.65,  1.0,  1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # Uvular stop ~ Qaf
    'r': [0.26,  1.0,  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # Alveolar trill/liquid ~ Ra
    's': [0.18, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # Alveolar sibilant ~ Seen
    't': [0.22, -1.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # Alveolar voiceless stop ~ Ta
    'u': [0.05,  1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], # High back rounded vowel ~ Waw
    'v': [0.08,  1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Labiodental voiced ~ Fa/Ba
    'w': [0.02,  1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], # Labial glide ~ Waw
    'x': [0.38, -1.0,  0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0], # ks compound
    'y': [0.44,  1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], # Palatal glide ~ Ya
    'z': [0.18,  1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # Alveolar voiced sibilant ~ Zay
    ' ': [0.00,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Pause/Sil
}

# Dimension weights according to Sibawayh's emphasis in Al-Kitab (Makhraj locus & Major Features)
DIM_WEIGHTS = [
    6.0,  # makhraj_depth (core locus: dominant physiological constraint)
    1.0,  # jahr_hams
    1.2,  # shiddah_rikhawah
    1.2,  # istila_istifal
    1.5,  # itbaq_infitah (critical distinction: Sad vs Seen, Ta vs Taa)
    1.0,  # dhalaqah_ismat
    1.0,  # safir
    1.0,  # qalqalah
    0.8,  # inhiraf
    0.8,  # takrir
    0.8,  # tafashshi
    1.2,  # istitalah (unique to Dad)
    1.2,  # ghunnah
    1.0,  # lin_madd
    2.0,  # coronal (organ of articulation)
    2.0,  # dorsal
    3.0,  # labial
]

def get_vector(char: str) -> Optional[List[float]]:
    """Returns the 17D Sibawayh vector for any Arabic, Branch, or Latin character."""
    if char in SIBAWAYH_ARABIC_VECTORS:
        return SIBAWAYH_ARABIC_VECTORS[char]
    if char in SIBAWAYH_BRANCH_VECTORS:
        return SIBAWAYH_BRANCH_VECTORS[char]
    if char in SIBAWAYH_SUBSTANDARD_VECTORS:
        return SIBAWAYH_SUBSTANDARD_VECTORS[char]
    if char.lower() in SIBAWAYH_LATIN_VECTORS:
        return SIBAWAYH_LATIN_VECTORS[char.lower()]
    return None

def sibawayh_distance(c1: str, c2: str) -> float:
    """
    Computes the weighted physiological articulatory distance between any two phonemes.
    D(c1, c2) = sqrt( sum( w_k * (phi_k(c1) - phi_k(c2))^2 ) )
    """
    v1 = get_vector(c1)
    v2 = get_vector(c2)
    if v1 is None or v2 is None:
        return float('inf')
    
    sq_dist = 0.0
    for val1, val2, w in zip(v1, v2, DIM_WEIGHTS):
        sq_dist += w * ((val1 - val2) ** 2)
    return math.sqrt(sq_dist)

def classify_sibawayh_pair(c1: str, c2: str) -> Dict[str, any]:
    """
    Classifies a pair of phonemes according to Sibawayh's classical Idgham ontology:
    - Mutamathilan (Identical) -> Wajib al-Idgham
    - Mutajanisan (Homorganic: same locus, different attributes)
    - Mutaqariban (Proximate: close locus/sifat) -> Ja'iz al-Idgham
    - Mutaba'idan (Distant: separate loci) -> Mumtani'
    """
    v1 = get_vector(c1)
    v2 = get_vector(c2)
    if v1 is None or v2 is None:
        return {"category": "Unknown", "dist": float('inf'), "idgham": "Mumtani'"}
    
    dist = sibawayh_distance(c1, c2)
    makhraj_diff = abs(v1[0] - v2[0])
    
    if dist < 1e-4:
        return {"category": "Mutamathilan", "dist": dist, "idgham": "Wajib (Obligatory)"}
    elif makhraj_diff < 0.03:
        return {"category": "Mutajanisan", "dist": dist, "idgham": "Ja'iz / Mustahabb (Permissible)"}
    elif dist < 1.40:
        return {"category": "Mutaqariban", "dist": dist, "idgham": "Ja'iz (Conditional)"}
    else:
        return {"category": "Mutaba'idan", "dist": dist, "idgham": "Mumtani' (Forbidden)"}

def find_nearest_arabic_neighbor(latin_char: str) -> Tuple[str, float]:
    """Finds the closest authentic Arabic phoneme in Sibawayh's space for a Latin character."""
    best_char = 'ا'
    min_d = float('inf')
    for ar_c in SIBAWAYH_ARABIC_VECTORS.keys():
        d = sibawayh_distance(latin_char, ar_c)
        if d < min_d:
            min_d = d
            best_char = ar_c
    return best_char, min_d

def find_nearest_latin_neighbor(arabic_char: str) -> Tuple[str, float]:
    """Finds the closest Latin phoneme in Sibawayh's space for an Arabic consonant."""
    best_char = 'a'
    min_d = float('inf')
    for lat_c in SIBAWAYH_LATIN_VECTORS.keys():
        if lat_c == ' ': continue
        d = sibawayh_distance(arabic_char, lat_c)
        if d < min_d:
            min_d = d
            best_char = lat_c
    return best_char, min_d

if __name__ == '__main__':
    print("=" * 70)
    print(">>> SIBAWAYH 17-DIMENSIONAL PHONETIC VECTOR SPACE VERIFIED <<<")
    print(f"Total Makharij Groups: {len(SIBAWAYH_MAKHRAJ_ORDINAL)}")
    print(f"Standard Arabic Consonants: {len(SIBAWAYH_ARABIC_VECTORS)}")
    print(f"Branch Allophones (Far'iyyah): {len(SIBAWAYH_BRANCH_VECTORS)}")
    print(f"Substandard Allophones (Mustarzalah): {len(SIBAWAYH_SUBSTANDARD_VECTORS)}")
    print("=" * 70)
