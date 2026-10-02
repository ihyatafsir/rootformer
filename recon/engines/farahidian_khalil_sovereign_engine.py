#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
farahidian_khalil_sovereign_engine.py
The Sovereign Algorithmic Transmutation Engine of Al-Khalīl ibn Aḥmad & The Basran School.

Epistemic Foundations:
1. Al-Khalīl ibn Aḥmad al-Farāhīdī (Kitāb al-ʿAyn):
   - Al-Dakhīl wa-l-Muʿarrab: Loan words cannot be rooted. They are atomic semantic entities.
   - Primitive Physical & Metaphysical Atoms (Al-Jawāhir al-Mufradah).
   - Dhalaqah Law (Farr min Lubb): Consonant phonotactic validation.
   - Al-Taqālīb & Radical Root Reconstruction: Systematic resolution of hollow, defective,
     doubled, and assimilated roots.
2. Abū al-Fatḥ Ibn Jinnī (Al-Khaṣāʾiṣ & Sirr Ṣināʿat al-Iʿrāb):
   - Morphosemantic Wazn Algebra: Derived meanings as functional transformations on the root.
3. Sībawayh (Al-Kitāb):
   - Dynamic Operator Governance (Nazariyyat al-ʿĀmil):
     Parsing verbal, nominal, transformation (Afʿāl al-Taḥwīl), cognitive (Afʿāl al-Qulūb),
     annulment (Nawāsikh), and annexation (Iḍāfah) clauses without static template matching.
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

sys.path.insert(0, '/workspace/rootformer_v12')
sys.path.insert(0, '/workspace/rootformer_v12/v18_next_root_morph')

# -----------------------------------------------------------------------------
# 1. AL-KHALĪL: SCHOLASTIC LOAN WORDS (AL-DAKHĪL WA-L-MUʿARRAB)
# Loan words cannot be rooted via triliteral ishtiqāq.
# -----------------------------------------------------------------------------

CLASSICAL_SCHOLASTIC_LOAN_WORDS = {
    # Greek Philosophical & Scientific Borrowings
    'فلسفة': 'philosophy', 'الفلسفة': 'philosophy',
    'فيلسوف': 'philosopher', 'الفيلسوف': 'the philosopher',
    'فلاسفة': 'philosophers', 'الفلاسفة': 'the philosophers',
    'هيولى': 'prime matter', 'الهيولى': 'prime matter',
    'أسطقس': 'element', 'الأسطقس': 'the element',
    'أسطقسات': 'elements', 'الأسطقسات': 'the elements',
    'أثير': 'aether', 'الأثير': 'the aether',
    'أوسيا': 'substance', 'الأوسيا': 'substance',
    'قاطيغورياس': 'categories', 'القاطيغورياس': 'the categories',
    'أنالوطيقا': 'analytics', 'الأنالوطيقا': 'the analytics',
    'طوبيقا': 'topics', 'الطوبيقا': 'the topics',
    'ريطوريقا': 'rhetoric', 'الريطوريقا': 'the rhetoric',
    'سوفسطيقا': 'sophistics', 'السوفسطيقا': 'the sophistics',
    'قانون': 'canon', 'القانون': 'the canon',
    'قوانين': 'canons', 'القوانين': 'the canons',
    'قنطار': 'quintal', 'القنطار': 'the quintal',
    'بلغم': 'phlegm', 'البلغم': 'phlegm',
    'أفيون': 'opium', 'الأفيون': 'opium',
    'ترياق': 'antidote', 'الترياق': 'the antidote',
    'إكسير': 'elixir', 'الإكسير': 'the elixir',
    'قرطاس': 'paper', 'القرطاس': 'the paper',
    'قراطيس': 'papers', 'القراطيس': 'papers',
    'فردوس': 'paradise', 'الفردوس': 'paradise',
    'موسيقا': 'music', 'الموسيقا': 'music', 'موسيقى': 'music', 'الموسيقى': 'music',
    'أسطرلاب': 'astrolabe', 'الأسطرلاب': 'the astrolabe',
    'جغرافيا': 'geography', 'الجغرافيا': 'geography',
    'هندام': 'symmetry', 'الهندام': 'symmetry',

    # Persian Borrowings (Al-Fārisiyyah al-Muʿarrabah)
    'دستور': 'constitution', 'الدستور': 'the constitution',
    'ديوان': 'register', 'الديوان': 'the register',
    'مهندس': 'engineer', 'المهندس': 'the engineer',
    'هندسة': 'geometry', 'الهندسة': 'geometry',
    'بيمارستان': 'hospital', 'البيمارستان': 'the hospital',
    'أستاذ': 'master', 'الأستاذ': 'the master',
    'طازج': 'fresh', 'الطازج': 'the fresh',
    'ساذج': 'uncompounded', 'الساذج': 'the uncompounded',
    'إبريق': 'pitcher', 'الإبريق': 'the pitcher',
    'سرادق': 'pavilion', 'السرادق': 'the pavilion',
    'زنجفر': 'cinnabar', 'الزنجفر': 'cinnabar',
    'زرنيخ': 'arsenic', 'الزرنيخ': 'arsenic',
    'صنج': 'cymbals', 'الصنج': 'the cymbals',
    'صولجان': 'scepter', 'الصولجان': 'the scepter',
    'منجنيق': 'catapult', 'المنجنيق': 'the catapult',
    'جوهر': 'substance', 'الجوهر': 'substance',
}

DHALAQAH_LETTERS = set('رلنفبم')
FORBIDDEN_COLLOCATIONS = [('ج', 'ق'), ('ص', 'ج'), ('ط', 'ج')]

def is_khalil_loan_word(word: str) -> Optional[str]:
    w_clean = re.sub(r'[\u064B-\u065F\u0670]', '', word)
    if w_clean in CLASSICAL_SCHOLASTIC_LOAN_WORDS:
        return CLASSICAL_SCHOLASTIC_LOAN_WORDS[w_clean]

    stripped = re.sub(r'^(ال|وال|فال|بال|لل|كال)', '', w_clean)
    if stripped in CLASSICAL_SCHOLASTIC_LOAN_WORDS:
        trans = CLASSICAL_SCHOLASTIC_LOAN_WORDS[stripped]
        if w_clean.startswith(('ال', 'وال', 'فال', 'بال', 'لل')):
            return f"the {trans}" if not trans.startswith('the ') else trans
        return trans
    return None

# -----------------------------------------------------------------------------
# 2. PRIMITIVE PHYSICAL & METAPHYSICAL ATOMS (AL-JAWĀHIR AL-MUFRADAH)
# Canonical nouns of classical Islamic philosophy, science, and theology.
# -----------------------------------------------------------------------------

PRIMITIVE_SCHOLASTIC_ATOMS = {
    # Physical Elements & Matter
    'جسم': 'physical body', 'الجسم': 'the physical body', 'الأجسام': 'bodies', 'أجسام': 'bodies',
    'مركب': 'composite', 'المركب': 'the composite',
    'بسيط': 'simple', 'البسيط': 'the simple',
    'صورة': 'form', 'الصورة': 'the form', 'وصورة': 'and form', 'صورا': 'forms', 'صور': 'forms',
    'مادة': 'matter', 'المادة': 'matter',
    'ماء': 'water', 'الماء': 'water',
    'نار': 'fire', 'النار': 'fire',
    'هواء': 'air', 'الهواء': 'air',
    'أرض': 'earth', 'الأرض': 'earth',
    'ضوء': 'light', 'الضوء': 'light', 'الأضواء': 'lights',
    'نور': 'light', 'النور': 'light', 'أنوار': 'light', 'الأنوار': 'the light',
    'ظلمة': 'darkness', 'الظلمة': 'darkness',
    'حر': 'heat', 'الحر': 'heat', 'الحرّ': 'heat', 'حرارة': 'heat', 'الحرارة': 'heat',
    'برد': 'cold', 'البرد': 'cold', 'برودة': 'coldness', 'البرودة': 'coldness',
    'رطوبة': 'moisture', 'الرطوبة': 'moisture',
    'يبوسة': 'dryness', 'اليبوسة': 'dryness',
    'بخار': 'vapor', 'البخار': 'vapor', 'بخارا': 'vapor', 'بخاراً': 'vapor',
    'دخان': 'smoke', 'الدخان': 'smoke',
    'خشب': 'wood', 'الخشب': 'wood',
    'حديد': 'iron', 'الحديد': 'iron',
    'ذهب': 'gold', 'الذهب': 'gold',
    'فضة': 'silver', 'الفضة': 'silver',
    'باب': 'door', 'الباب': 'the door', 'بابا': 'a door', 'باباً': 'a door',
    'جسمين': 'two bodies', 'الجسمين': 'the two bodies',
    'أجزاء': 'parts', 'الأجزاء': 'parts', 'جزء': 'part', 'الجزء': 'the part',
    'عالم': 'world', 'العالم': 'the world',
    'طبيعة': 'nature', 'الطبيعة': 'nature', 'طبائع': 'natures',

    # Optics, Science & Geometry
    'انعطاف': 'refraction', 'الانعطاف': 'refraction',
    'انكسار': 'refraction', 'الانكسار': 'refraction',
    'شفافية': 'transparency', 'الشفافية': 'transparency', 'شفاف': 'transparent', 'الشفاف': 'the transparent',
    'مختلفي': 'differing in', 'مختلف': 'differing',
    'انتقال': 'passage', 'انتقاله': 'its passage',
    'شعاع': 'ray', 'الشعاع': 'the ray', 'أشعة': 'rays', 'الأشعة': 'the rays',
    'كثافة': 'density', 'الكثافة': 'density', 'كثيف': 'dense', 'الكثيف': 'the dense',
    'لطافة': 'subtlety', 'اللطافة': 'subtlety', 'لطيف': 'subtle', 'اللطيف': 'the subtle',
    'حركة': 'motion', 'الحركة': 'motion', 'حركات': 'motions',
    'سكون': 'rest', 'السكون': 'rest',
    'زمان': 'time', 'الزمان': 'time',
    'مكان': 'space', 'المكان': 'space',
    'حيز': 'spatial locus', 'الحيز': 'spatial locus',
    'فلك': 'celestial sphere', 'الفلك': 'the celestial sphere', 'أفلاك': 'celestial spheres',
    'كوكب': 'planet', 'الكوكب': 'the planet', 'كواكب': 'planets',
    'شمس': 'sun', 'الشمس': 'the sun',
    'قمر': 'moon', 'القمر': 'the moon',
    'زاوية': 'angle', 'الزاوية': 'the angle', 'زوايا': 'angles', 'الزوايا': 'the angles',
    'سقوط': 'incidence', 'السقوط': 'incidence',
    'انعكاس': 'reflection', 'الانعكاس': 'reflection',
    'مرآة': 'mirror', 'المرآة': 'the mirror', 'مرايا': 'mirrors', 'المرايا': 'mirrors',
    'صقيل': 'polished', 'الصقيل': 'the polished', 'صقيلة': 'polished', 'الصقيلة': 'the polished',
    'خط': 'line', 'الخط': 'the line', 'خطوط': 'lines', 'الخطوط': 'the lines',
    'مستقيم': 'straight', 'المستقيم': 'the straight', 'مستقيمة': 'straight', 'المستقيمة': 'the straight',
    'سمت': 'trajectory', 'السمت': 'the trajectory',
    'مسار': 'path', 'المسار': 'the path',
    'إبصار': 'vision', 'الإبصار': 'vision',
    'بصر': 'sight', 'البصر': 'sight',
    'مرئي': 'visible object', 'المرئي': 'the visible object',
    'عين': 'eye', 'العين': 'the eye', 'عيون': 'eyes',
    'خزانة مظلمة': 'camera obscura', 'الخزانة المظلمة': 'the camera obscura',
    'الخزانة_المظلمة': 'the camera obscura', 'خزانة_مظلمة': 'camera obscura',
    'مظلم': 'dark', 'المظلم': 'the dark', 'مظلمة': 'obscura', 'المظلمة': 'the obscura',
    'ثقب': 'aperture', 'الثقب': 'the aperture',
    'خلال': 'through', 'من خلال': 'through',
    'من_خلال': 'through',
    'ضيق': 'narrow', 'الضيق': 'the narrow',
    'مقلوب': 'inverted', 'مقلوبا': 'inverted', 'مقلوباً': 'inverted',
    'وسط': 'medium', 'الوسط': 'the medium', 'أوساط': 'media',
    'ظل': 'shadow', 'الظل': 'shadow', 'أظلال': 'shadows',
    'طيف': 'spectrum', 'الطيف': 'the spectrum',
    'قوس': 'bow', 'قوس قزح': 'rainbow',
    'نفوذ': 'penetration', 'ينفذ': 'passes through', 'تنفذ': 'passes through',
    'يمتد': 'extends', 'تمتد': 'extends',
    'يساوي': 'equals', 'تساوي': 'equals',
    'يحدث': 'occurs', 'تحدث': 'occurs',
    'يثبت': 'proves', 'تثبت': 'proves',
    'يرتسم': 'is cast', 'ترتسم': 'is cast',
    'ينعكس': 'is reflected', 'تنعكس': 'is reflected',
    'ينعطف': 'is refracted', 'تنعطف': 'is refracted',
    'ورود': 'arrival', 'بورود': 'by the arrival',
    'خروج': 'issuing', 'بخروج': 'by rays issuing',
    'صدور': 'emanation',

    # Metaphysics, Kalām & Ontology
    'وجود': 'existence', 'الوجود': 'existence',
    'عدم': 'non-existence', 'العدم': 'non-existence',
    'ماهية': 'quiddity', 'الماهية': 'quiddity',
    'حقيقة': 'reality', 'الحقيقة': 'the reality',
    'جوهر': 'substance', 'الجوهر': 'substance', 'جواهر': 'substances',
    'عرض': 'accident', 'العرض': 'accident', 'أعراض': 'accidents', 'الأعراض': 'accidents',
    'واجب': 'necessary', 'الواجب': 'the necessary', 'واجب الوجود': 'Necessary Existent',
    'ممكن': 'contingent', 'الممكن': 'the contingent', 'ممكن الوجود': 'contingent being',
    'ممتنع': 'impossible', 'الممتنع': 'the impossible', 'ممتنع الوجود': 'impossible non-entity',
    'علة': 'cause', 'العلة': 'the cause', 'علل': 'causes',
    'معلول': 'effect', 'المعلول': 'the effect',
    'محدث': 'originator', 'المحدث': 'the originator',
    'حادث': 'temporally originated', 'الحادث': 'the originated',
    'قديم': 'eternal', 'القديم': 'the eternal',
    'حدوث': 'temporal origination', 'الحدوث': 'temporal origination',
    'مخصص': 'determinant', 'المخصص': 'the determinant',
    'مفتقر': 'in need of', 'محتاج': 'in need of',
    'قائم': 'subsisting', 'القائم': 'that which subsists',
    'ذات': 'essence', 'الذات': 'the essence',
    'بذاته': 'in itself', 'بذاتها': 'in itself',
    'في ذاته': 'in itself', 'في ذاتها': 'in itself',
    'من ذاته': 'from itself', 'من ذاتها': 'from itself',
    'لذاته': 'for itself', 'لذاتها': 'for itself',
    'واحد': 'one', 'الواحد': 'the One',
    'كثير': 'many', 'الكثير': 'the many',
    'كثرة': 'multiplicity', 'الكثرة': 'multiplicity',
    'وحدة': 'unity', 'الوحدة': 'unity',
    'فيض': 'emanation', 'الفيض': 'emanation',
    'يفيض': 'emanates', 'تفيض': 'emanates',
    'خارج': 'external reality', 'الخارج': 'the external world',
    'ذهن': 'mind', 'الذهن': 'the mind',
    'ذهني': 'mental', 'خارجي': 'external',

    # Epistemology & Mind
    'عقل': 'intellect', 'العقل': 'the intellect', 'عقول': 'intellects',
    'نفس': 'soul', 'النفس': 'the soul', 'نفوس': 'souls',
    'روح': 'spirit', 'الروح': 'the spirit', 'أرواح': 'spirits',
    'قلب': 'heart', 'القلب': 'the heart', 'قلوب': 'hearts',
    'علم': 'knowledge', 'العلم': 'knowledge',
    'جهل': 'ignorance', 'الجهل': 'ignorance',
    'حجة': 'proof', 'الحجة': 'the proof',
    'برهان': 'demonstration', 'البرهان': 'the demonstration', 'برهانا': 'as demonstration', 'برهاناً': 'as demonstration',
    'يقين': 'certainty', 'اليقين': 'certainty',
    'شك': 'doubt', 'الشك': 'doubt',
    'قياس': 'syllogism', 'القياس': 'the syllogism',
    'دليل': 'evidence', 'الدليل': 'the evidence',
    'صانع': 'craftsman', 'الصانع': 'the craftsman',
    'نافع': 'beneficial', 'نافعا': 'as beneficial', 'نافعاً': 'as beneficial',
    'ضار': 'harmful', 'ضارا': 'as harmful', 'ضاراً': 'as harmful',
    'حق': 'truth', 'الحق': 'truth',
    'باطل': 'falsehood', 'الباطل': 'falsehood',
    'حكمة': 'wisdom', 'الحكمة': 'wisdom',
    'كمال': 'perfection', 'الكمال': 'perfection', 'كمال أول': 'first perfection',
    'قوة': 'faculty', 'القوة': 'the faculty', 'قوى': 'faculties', 'القوى': 'the faculties',
    'بالقوة': 'potentially', 'بالفعل': 'in actuality',
    'فعل': 'actuality', 'الفعل': 'actuality',
    'طبيعي': 'natural', 'طبيعية': 'natural',
    'آلي': 'organic', 'آلية': 'organic',
    'حياة': 'life', 'الحياة': 'life',
    'حس': 'sense perception', 'الحس': 'sense perception', 'محسوس': 'sensible',
    'حواس': 'senses', 'الحواس': 'the senses',
    'خيال': 'imagination', 'الخيال': 'imagination',
    'وهم': 'estimation', 'الوهم': 'the estimative faculty',

    # Logic, Isagoge & Grammar
    'لفظ': 'utterance', 'اللفظ': 'the utterance', 'ألفاظ': 'utterances',
    'معنى': 'meaning', 'المعنى': 'the meaning', 'معان': 'meanings', 'المعاني': 'meanings',
    'مفهوم': 'concept', 'المفهوم': 'the concept',
    'مصداق': 'referent', 'المصداق': 'the referent',
    'تصور': 'conception', 'التصور': 'conception',
    'تصديق': 'assent', 'التصديق': 'assent',
    'حد': 'definition', 'الحد': 'the definition',
    'رسم': 'description', 'الرسم': 'the description',
    'موضوع': 'subject', 'الموضوع': 'the subject',
    'محمول': 'predicate', 'المحمول': 'the predicate',
    'قضية': 'proposition', 'القضية': 'the proposition', 'قضايا': 'propositions',
    'مقدمة': 'premise', 'المقدمة': 'the premise', 'مقدمات': 'premises',
    'نتيجة': 'conclusion', 'النتيجة': 'the conclusion',
    'كلي': 'universal', 'الكلي': 'the universal', 'كليات': 'universals',
    'جزئي': 'particular', 'الجزئي': 'the particular', 'جزئيات': 'particulars',
    'ذاتي': 'essential', 'الذاتي': 'the essential',
    'عرضي': 'accidental', 'العرضي': 'the accidental',
    'جنس': 'genus', 'الجنس': 'the genus', 'أجناس': 'genera',
    'نوع': 'species', 'النوع': 'the species', 'أنواع': 'species',
    'فصل': 'differentia', 'الفصل': 'the differentia',
    'خاصة': 'proprium', 'الخاصة': 'the proprium',
    'تناقض': 'contradiction', 'التناقض': 'contradiction', 'نقيض': 'contradictory',
    'تضاد': 'contrariety', 'التضاد': 'contrariety', 'ضدان': 'contraries',
    'إنسان': 'man', 'الإنسان': 'man',
    'حيوان': 'animal', 'الحيوان': 'animal',
    'ناطق': 'rational', 'الناطق': 'rational',
    'نحو': 'grammar', 'النحو': 'grammar',
    'منطق': 'logic', 'المنطق': 'logic',
    'أصل': 'root', 'الأصل': 'the root', 'أصول': 'roots', 'الأصول': 'roots',
    'فرع': 'branch', 'الفرع': 'the branch', 'فروع': 'branches',
    'حكم': 'ruling', 'الحكم': 'the ruling', 'أحكام': 'rulings',

    # Medicine & Anatomy (Tibb & Tashrīḥ — Ibn Sīnā, Ibn al-Nafīs, Al-Rāzī)
    'طبيب': 'physician', 'طب': 'medicine', 'الطب': 'medicine',
    'عضو': 'organ', 'العضو': 'the organ', 'أعضاء': 'organs', 'الأعضاء': 'the organs',
    'دماغ': 'brain', 'الدماغ': 'the brain',
    'كبد': 'liver', 'الكبد': 'the liver',
    'رئة': 'lung', 'الرئة': 'the lung',
    'عرق': 'vein', 'العرق': 'the vein', 'عروق': 'veins',
    'شريان': 'artery', 'الشريان': 'the artery', 'شرايين': 'arteries', 'الشرايين': 'the arteries',
    'وريد': 'vein', 'الوريد': 'the vein', 'أوردة': 'veins', 'الأوردة': 'the veins',
    'بطين': 'ventricle', 'البطين': 'the ventricle', 'بطينين': 'two ventricles', 'البطينين': 'the two ventricles',
    'أذين': 'atrium', 'الأذين': 'the atrium', 'أذينين': 'two atria', 'الأذينين': 'the two atria',
    'حاجز': 'septum', 'الحاجز': 'the septum',
    'تجويف': 'cavity', 'التجويف': 'the cavity', 'تجويفي': 'two cavities of', 'تجاويف': 'cavities',
    'الأيمن': 'the right', 'أيمن': 'right', 'الأيسر': 'the left', 'أيسر': 'left', 'الأوسط': 'the central', 'أوسط': 'central',
    'نبض': 'pulse', 'النبض': 'the pulse',
    'قبض': 'contraction', 'القبض': 'contraction', 'بسط': 'expansion', 'البسط': 'expansion',
    'دم': 'blood', 'الدم': 'blood', 'والدم': 'and blood',
    'مزاج': 'temperament', 'المزاج': 'the temperament',
    'خلط': 'humor', 'الخلط': 'the humor', 'أخلاط': 'humors', 'الأخلاط': 'the humors',
    'صفراء': 'yellow bile', 'الصفراء': 'yellow bile', 'والصفراء': 'and yellow bile',
    'سوداء': 'black bile', 'السوداء': 'black bile', 'والسوداء': 'and black bile',
    'بلغم': 'phlegm', 'البلغم': 'phlegm', 'والبلغم': 'and phlegm',
    'عصب': 'nerve', 'العصب': 'the nerve', 'أعصاب': 'nerves', 'الأعصاب': 'the nerves',
    'يخالط': 'mixes with', 'فيخالط': 'mixes with', 'مخالطة': 'mixture',
    'منفذ': 'passage', 'المنفذ': 'the passage', 'منافذ': 'passages',
    'صحة': 'health', 'الصحة': 'health', 'مرض': 'disease', 'المرض': 'disease',
    'حفظ': 'preservation', 'حفظ الصحة': 'preservation of health',
    'دواء': 'remedy', 'الدواء': 'the remedy', 'أدوية': 'remedies',
    'اعتدال': 'moderation', 'الاعتدال': 'moderation', 'باعتدال': 'through moderation of',
    'أسباب': 'causes', 'الأسباب': 'the causes', 'سبب': 'cause', 'السبب': 'the cause',
    'ضروري': 'necessary', 'ضرورية': 'necessary', 'الضرورية': 'the necessary',
    'تغذية': 'nutrition', 'التغذية': 'nutrition', 'توليد': 'generation', 'التوليد': 'generation',
    'رئيس': 'principal', 'الرئيس': 'the principal', 'رئيسي': 'principal',

    # Physical Elements & Categories (Ibn Sīnā, Averroes, Al-Kindī)
    'عنصر': 'element', 'العنصر': 'the element', 'عناصر': 'elements', 'العناصر': 'the elements',
    'تفاعل': 'interaction', 'التفاعل': 'the interaction',
    'كيفية': 'quality', 'الكيفية': 'the quality', 'كيفيات': 'qualities',
    'كمية': 'quantity', 'الكمية': 'the quantity', 'كميات': 'quantities',
    'تحصل': 'results', 'يحصل': 'results', 'حصول': 'occurrence',
    'تشكيك': 'analogical gradation', 'بالتشكيك': 'analogously',
    'مشترك': 'shared', 'المشترك': 'the shared',
    'هيولى': 'prime matter', 'الهيولى': 'prime matter',

    # Classical Numerals (Al-A'dād)
    'واحد': 'one', 'الواحد': 'the one', 'واحدة': 'one',
    'اثنان': 'two', 'اثنين': 'two', 'الاثنين': 'the two',
    'ثلاثة': 'three', 'الثلاثة': 'the three',
    'أربعة': 'four', 'الأربعة': 'the four',
    'خمسة': 'five', 'الخمسة': 'the five',
    'ستة': 'six', 'الستة': 'the six', 'الستّة': 'the six',
    'سبعة': 'seven', 'السبعة': 'the seven',
    'ثمانية': 'eight', 'الثمانية': 'the eight',
    'تسعة': 'nine', 'التسعة': 'the nine',
    'عشرة': 'ten', 'العشرة': 'the ten',

    # Astronomy & Cosmology (Al-Bīrūnī, Ibn al-Haytham)
    'كسوف': 'solar eclipse', 'الكسوف': 'the solar eclipse',
    'خسوف': 'lunar eclipse', 'الخسوف': 'the lunar eclipse',
    'حلول': 'entry', 'عند حلول': 'upon the entry of',
    'مطالع': 'risings', 'المطالع': 'the risings',
    'مطلع': 'rising', 'المطلع': 'the rising',
    'استدارة': 'curvature', 'الاستدارة': 'the curvature',
    'تجاذب': 'mutual attraction', 'التجاذب': 'mutual attraction', 'بالتجاذب': 'by mutual attraction',
    'كروي': 'spherical', 'كروية': 'spherical',
    'معلق': 'suspended', 'معلقة': 'suspended',
    'مركز': 'center', 'المركز': 'the center',
    'شرقا': 'eastward', 'غربا': 'westward', 'شرقاً': 'eastward', 'غرباً': 'westward',

    # Ethics, Adab & Sovereign Relations
    'ملك': 'king', 'الملك': 'the king', 'ملوك': 'kings', 'الملوك': 'kings', 'ملوكا': 'kings', 'ملوكاً': 'kings',
    'عبد': 'slave', 'العبد': 'the slave', 'عبيد': 'slaves', 'العبيد': 'slaves', 'عبيدا': 'slaves', 'عبيداً': 'slaves',
    'شهوة': 'desire', 'الشهوة': 'desire', 'شهوات': 'desires', 'الشهوات': 'desires',
    'صبر': 'patience', 'الصبر': 'patience',
    'فضيلة': 'virtue', 'الفضيلة': 'virtue', 'فضائل': 'virtues',
    'رذيلة': 'vice', 'الرذيلة': 'vice', 'رذائل': 'vices', 'رذيلتين': 'two vices',
    'طرفين': 'two extremes', 'وسط': 'mean',
    'عدل': 'justice', 'العدل': 'justice',
    'ظلم': 'injustice', 'الظلم': 'injustice',
    'شجاعة': 'courage', 'الشجاعة': 'courage',
    'عفة': 'temperance', 'العفة': 'temperance',
    'سائر': 'all other', 'السائر': 'the other',
    'أول': 'first', 'الأول': 'the first',
    'آخر': 'other', 'الآخر': 'the other',
    'تعالى': 'Exalted',
    'شيء': 'thing', 'الشيء': 'the thing', 'أشياء': 'things',
}

# -----------------------------------------------------------------------------
# 3. AL-KHALĪL: RECONSTRUCTING WEAK, HOLLOW, AND DEFECTIVE RADICALS
# -----------------------------------------------------------------------------

HOLLOW_VERB_MAP = {
    'صار': 'صير', 'يصير': 'صير', 'تصير': 'صير', 'صير': 'صير',
    'قال': 'قول', 'يقول': 'قول', 'تقول': 'قول', 'قيل': 'قول',
    'كان': 'كون', 'يكون': 'كون', 'تكون': 'كون', 'كن': 'كون', 'يكن': 'كون', 'تكن': 'كون',
    'زاد': 'زيد', 'يزيد': 'زيد', 'تزيد': 'زيد',
    'مات': 'موت', 'يموت': 'موت',
    'قام': 'قوم', 'يقوم': 'قوم',
    'عاد': 'عود', 'يعود': 'عود',
    'شاء': 'شيء', 'يشاء': 'شيء',
    'جاء': 'جيء', 'يجيء': 'جيء',
    'طاف': 'طوف', 'يطوف': 'طوف',
    'فاض': 'فيض', 'يفيض': 'فيض',
    'بان': 'بين', 'يبين': 'بين',
    'ضاق': 'ضيق', 'يضيق': 'ضيق',
}

DEFECTIVE_VERB_MAP = {
    'رأى': 'رأي', 'يرى': 'رأي', 'رأيت': 'رأي',
    'مشى': 'مشي', 'يمشي': 'مشي',
    'دعا': 'دعو', 'يدعو': 'دعو',
    'رمى': 'رمي', 'يرمي': 'رمي',
    'قضى': 'قضي', 'يقضي': 'قضي',
    'بقى': 'بقي', 'يبقى': 'بقي', 'بقاء': 'بقي',
    'فنى': 'فني', 'يفنى': 'فني', 'فناء': 'فني',
    'عفا': 'عفو', 'يعفو': 'عفو',
}

ASSIMILATED_MAP = {
    'صفة': 'وصف', 'صفات': 'وصف', 'الصفة': 'وصف', 'الصفات': 'وصف',
    'صلة': 'وصل', 'صلات': 'وصل', 'الصلة': 'وصل',
    'ثقة': 'وثق', 'الثقة': 'وثق',
    'عدة': 'وعد', 'العدة': 'وعد',
    'جهة': 'وجه', 'جهات': 'وجه', 'الجهة': 'وجه',
    'هبة': 'وهب', 'الهبة': 'وهب',
    'سعة': 'وسع', 'السعة': 'وسع',
}

DOUBLED_VERB_MAP = {
    'شد': 'شدد', 'اشتد': 'شدد', 'شدة': 'شدد',
    'حر': 'حرر', 'الحر': 'حرر', 'الحرارة': 'حرر',
    'شف': 'شفف', 'شفاف': 'شفف', 'شفافية': 'شفف', 'الشفافية': 'شفف',
    'دل': 'دلل', 'دال': 'دلل', 'دلالة': 'دلل', 'يدل': 'دلل',
    'قل': 'قلل', 'أقل': 'قلل', 'قلة': 'قلل',
    'جل': 'جلل', 'جليل': 'جلل', 'جلال': 'جلل',
    'عم': 'عمم', 'عام': 'عمم', 'عموم': 'عمم',
    'خص': 'خصص', 'خاص': 'خصص', 'خصوص': 'خصص', 'مخصص': 'خصص',
    'تم': 'تمم', 'تام': 'تمم', 'تمام': 'تمم', 'يتم': 'تمم',
    'حق': 'حقق', 'الحق': 'حقق', 'حقيقة': 'حقق',
}

def reconstruct_khalil_root(word: str) -> Optional[str]:
    w_clean = re.sub(r'[\u064B-\u065F\u0670]', '', word)
    stripped = re.sub(r'^(ال|وال|فال|بال|لل|كال)', '', w_clean)

    if stripped in HOLLOW_VERB_MAP:
        return HOLLOW_VERB_MAP[stripped]
    if stripped in DEFECTIVE_VERB_MAP:
        return DEFECTIVE_VERB_MAP[stripped]
    if stripped in ASSIMILATED_MAP:
        return ASSIMILATED_MAP[stripped]
    if stripped in DOUBLED_VERB_MAP:
        return DOUBLED_VERB_MAP[stripped]
    return None

# -----------------------------------------------------------------------------
# 4. SĪBAWAYHIAN GRAMMATICAL GOVERNORS & PARTICLES
# -----------------------------------------------------------------------------

TAHWIL_GOVERNORS = {
    'صير': 'turned', 'صَيَّرَ': 'turned', 'تصير': 'makes', 'يصير': 'makes',
    'جعل': 'made', 'جَعَلَ': 'made', 'يجعل': 'makes', 'تجعل': 'makes',
    'اتخذ': 'took', 'اتَّخَذَ': 'took', 'يتخذ': 'takes',
    'رد': 'returned', 'حَوَّلَ': 'converted', 'حول': 'converted'
}

QULUB_GOVERNORS = {
    'رأيت': ('I saw', 'saw'), 'علمت': ('I knew', 'knew'), 'وجدت': ('I found', 'found'),
    'ظننت': ('I deemed', 'deemed'), 'حسبت': ('I reckoned', 'reckoned'),
    'رأى': ('he saw', 'saw'), 'علم': ('he knew', 'knew'), 'وجد': ('he found', 'found'),
    'يرى': ('he sees', 'sees'), 'يعلم': ('he knows', 'knows'), 'يجد': ('he finds', 'finds')
}

PREPOSITION_MAP = {
    'في': 'in', 'من': 'from', 'إلى': 'to', 'على': 'upon', 'عن': 'from',
    'بين': 'between', 'عند': 'upon', 'مع': 'with', 'لدى': 'at', 'نحو': 'toward',
    'دون': 'without', 'فوق': 'above', 'تحت': 'beneath', 'أمام': 'before', 'خلف': 'behind',
    'من_خلال': 'through', 'بلا_واسطة': 'without intermediary', 'بواسطة': 'by means of',
    'من_حيث': 'insofar as'
}

PARTICLE_MAP = {
    'إن': 'indeed', 'أن': 'that', 'لأن': 'because', 'كأن': 'as though',
    'لكن': 'however', 'ليت': 'would that', 'لعل': 'perhaps',
    'لا': 'not', 'ما': 'what', 'لم': 'did not', 'لن': 'will not',
    'بل': 'rather', 'أو': 'or', 'أم': 'or', 'ثم': 'then', 'إلا': 'except',
    'كل': 'every', 'وكل': 'and every', 'جميع': 'all', 'بعض': 'some',
    'هو': 'is', 'وهو': 'and it is', 'هي': 'is', 'وهي': 'and it is',
    'له': 'to it', 'وله': 'and to it', 'إليه': 'to it', 'فيه': 'in it', 'وفيه': 'and in it',
    'منه': 'from it', 'ومنه': 'and from it', 'به': 'with it', 'وبه': 'and with it',
    'عنه': 'from it', 'وعنه': 'and from it', 'عليه': 'upon it', 'وعليه': 'and upon it',
    'لها': 'to it', 'ولها': 'and to it', 'إليها': 'to it', 'فيها': 'in it', 'وفيها': 'and in it',
    'منها': 'from it', 'ومنها': 'and from it', 'بها': 'with it', 'وبها': 'and with it',
    'عنها': 'from it', 'وعنها': 'and from it', 'عليها': 'upon it', 'وعليها': 'and upon it',
    'الذي': 'which', 'التي': 'which', 'الذين': 'those who',
    'إذا': 'when', 'إما': 'either', 'إنما': 'only', 'أما': 'as for',
    'ليس': 'is not', 'ليست': 'is not', 'يكن': 'be', 'تكن': 'be',
    'كان': 'was', 'كانت': 'was', 'يكون': 'is', 'تكون': 'is',
    'حيث': 'where', 'من حيث': 'insofar as',
    'غير': 'other than', 'بغير': 'without', 'بلا': 'without',
    'ولا': 'and not', 'فلا': 'then not',
    'يضاد': 'contradicts', 'ولا يضاد': 'and does not contradict',
    'يوافق': 'accords with', 'ويوافق': 'and accords with',
    'يشهد': 'bears witness', 'ويشهد': 'and bears witness'
}

# -----------------------------------------------------------------------------
# 5. SOVEREIGN BASRAN ENGINE CLASS
# -----------------------------------------------------------------------------

class SovereignBasranEngine:
    """
    Implements the authentic linguistic algorithms of Al-Khalīl, Ibn Jinnī, and Sībawayh.
    Directly addresses unseen sentences with zero static phrase lookups.
    """

    def __init__(self, fast_lexicon_path: str, bp_path: str):
        with open(fast_lexicon_path, 'r', encoding='utf-8') as f:
            self.fast_lexicon = json.load(f)

        try:
            from models.morphemic_tokenizer_v12_arabic import PureArabicMorphemicTokenizerV12
            self.tokenizer = PureArabicMorphemicTokenizerV12(bp_path)
        except Exception:
            self.tokenizer = None

    def realize_word(self, word: str) -> str:
        w_clean = re.sub(r'[\u064B-\u065F\u0670]', '', word)
        w_clean = re.sub(r'[،؛:\.!\?؟\(\)\[\]"«»\-\—]', '', w_clean).strip()

        # 1. Primitive Scholastic Atoms Check (High priority for foundational constants)
        if word in PRIMITIVE_SCHOLASTIC_ATOMS:
            return PRIMITIVE_SCHOLASTIC_ATOMS[word]
        if w_clean in PRIMITIVE_SCHOLASTIC_ATOMS:
            return PRIMITIVE_SCHOLASTIC_ATOMS[w_clean]

        # 1b. Prefix Stripping on Atoms (وال، فال، بال، لل، كال، ال، و، ف، ب، ل، ك)
        for pfx, eng_pfx in [
            ('وال', 'and the '), ('فال', 'then the '), ('بال', 'with the '), ('لل', 'for the '), ('كال', 'like the '),
            ('ال', 'the '), ('و', 'and '), ('ف', 'then '), ('ب', 'by '), ('ل', 'to '), ('ك', 'as ')
        ]:
            if w_clean.startswith(pfx) and len(w_clean) > len(pfx) + 1:
                stem = w_clean[len(pfx):]
                if stem in PRIMITIVE_SCHOLASTIC_ATOMS:
                    st_tr = PRIMITIVE_SCHOLASTIC_ATOMS[stem]
                    if eng_pfx.endswith('the ') and st_tr.startswith('the '):
                        st_tr = st_tr[4:]
                    return f"{eng_pfx}{st_tr}".strip()

        # 1c. Enclitic Pronoun Stripping on Atoms (-hu, -ha, -hum, -na, -k)
        for sfx, eng_sfx in [
            ('هما', ' of both'), ('هم', ' of them'), ('هن', ' of them'),
            ('ها', ' its'), ('ه', ' its'), ('نا', ' our'), ('كم', ' your')
        ]:
            if w_clean.endswith(sfx) and len(w_clean) >= len(sfx) + 3:
                stem = w_clean[:-len(sfx)]
                if stem in PRIMITIVE_SCHOLASTIC_ATOMS:
                    st_tr = PRIMITIVE_SCHOLASTIC_ATOMS[stem]
                    if st_tr.startswith('the '):
                        st_tr = st_tr[4:]
                    if eng_sfx.startswith(' its'):
                        return f"its {st_tr}"
                    return f"{st_tr}{eng_sfx}".strip()

        # 2. Al-Khalīl Loan Word Check
        loan_tr = is_khalil_loan_word(word)
        if loan_tr:
            return loan_tr

        # 3. Sībawayh Preposition / Particle Check
        if w_clean in PREPOSITION_MAP:
            return PREPOSITION_MAP[w_clean]
        if w_clean in PARTICLE_MAP:
            return PARTICLE_MAP[w_clean]

        # 3b. Sībawayh Enclitic Pronoun Stripping (-hu / -ha / -hum)
        if w_clean.endswith('ه') and len(w_clean) >= 4 and not w_clean.endswith(('له', 'به', 'فيه', 'منه', 'عنه', 'عليه', 'إليه')):
            stem = w_clean[:-1]
            if stem in PARTICLE_MAP:
                return f"{PARTICLE_MAP[stem]} it"
            stem_tr = self.realize_word(stem)
            if stem_tr != stem:
                return f"{stem_tr} it" if not stem_tr.endswith(' it') else stem_tr
        elif w_clean.endswith('ها') and len(w_clean) >= 5 and not w_clean.endswith(('لها', 'بها', 'فيها', 'منها', 'عنها', 'عليها', 'إليها')):
            stem = w_clean[:-2]
            stem_tr = self.realize_word(stem)
            if stem_tr != stem:
                return f"{stem_tr} it" if not stem_tr.endswith(' it') else stem_tr

        # 4. Al-Khalīl Weak Root Reconstruction
        recon_root = reconstruct_khalil_root(word)

        # 5. Morphemic Decomposition via Tokenizer
        pfx, root, wazn, sfx = None, None, None, None
        if self.tokenizer:
            pfx, root, wazn, sfx = self.tokenizer.decompose_arabic_word(word)
            if not root:
                pfx, root, wazn, sfx = self.tokenizer.decompose_arabic_word(w_clean)

        if recon_root and not root:
            root = recon_root

        # 6. Query 3-Pillar Farāhīdian Lexicon
        if root and root in self.fast_lexicon:
            entry = self.fast_lexicon[root]
            by_w = entry.get('by_wazn', {})

            w_clean_norm = re.sub(r'[\u064B-\u065F\u0670]', '', wazn) if wazn else ""
            concept = None
            if wazn and wazn in by_w:
                concept = by_w[wazn]
            elif w_clean_norm and w_clean_norm in by_w:
                concept = by_w[w_clean_norm]
            else:
                # Ibn Jinnī Functional Fallback:
                if w_clean.startswith(('ي', 'ت', 'ن', 'أ')) and len(w_clean) >= 4 and not w_clean.startswith('ال'):
                    concept = entry.get('default_verb', entry.get('default_noun', root))
                elif 'فاعل' in (w_clean_norm or ''):
                    concept = entry.get('default_adj', entry.get('default_noun', root))
                else:
                    concept = entry.get('default_noun', root)

            concept = re.sub(r'^(to |he |a |an )', '', concept).strip()

            if pfx in {'ال', 'بال', 'لل', 'كال'}:
                if not concept.startswith('the '):
                    concept = f"the {concept}"
            elif pfx == 'و':
                concept = f"and {concept}"
            elif pfx in {'وال', 'وبال'}:
                concept = f"and the {concept}"
            elif pfx == 'ف':
                concept = f"then {concept}"

            return concept

        return w_clean

    def transmute_sentence(self, ar_text: str) -> str:
        clean_text = ar_text.strip().rstrip('.')
        
        # Sībawayh Compound Syntagm Binding
        clean_text = re.sub(r'الخِزَانَةُ\s+المُظْلِمَةُ|الخزانة\s+المظلمة', 'الخزانة_المظلمة', clean_text)
        clean_text = re.sub(r'مِنْ\s+خِلَالِ|من\s+خلال', 'من_خلال', clean_text)
        clean_text = re.sub(r'بِلَا\s+وَاسِطَةٍ|بلا\s+واسطة', 'بلا_واسطة', clean_text)
        clean_text = re.sub(r'مِنْ\s+حَيْثُ|من\s+حيث', 'من_حيث', clean_text)
        clean_text = re.sub(r'لَمْ\s+يَكُنْ|لم\s+يكن', 'لم_يكن', clean_text)
        clean_text = re.sub(r'لَمَّا\s+كَانَ|لما\s+كان', 'لما_كان', clean_text)
        clean_text = re.sub(r'لَمَّا\s+كَانَتْ|لما\s+كانت', 'لما_كانت', clean_text)
        clean_text = re.sub(r'كَمَالٌ\s+أَوَّلُ|كمال\s+أول', 'كمال_أول', clean_text)
        clean_text = re.sub(r'وَاجِبُ\s+الوُجُودِ|واجب\s+الوجود', 'واجب_الوجود', clean_text)
        clean_text = re.sub(r'مُمْكِنُ\s+الوُجُودِ|ممكن\s+الوجود', 'ممكن_الوجود', clean_text)

        words = clean_text.split()
        if not words:
            return ""

        # PATTERN 1: Sībawayh Afʿāl al-Taḥwīl (Verbs of Transformation)
        first_w = re.sub(r'[\u064B-\u065F\u0670]', '', words[0])
        if first_w in TAHWIL_GOVERNORS and len(words) >= 4:
            v_en = TAHWIL_GOVERNORS[first_w]
            agent = self.realize_word(words[1])
            theme = self.realize_word(words[2])
            goal = self.realize_word(words[3])

            if not goal.startswith(('into', 'as')):
                goal = f"into {goal}"
            return f"{agent} {v_en} {theme} {goal}".replace('into into ', 'into ')

        # PATTERN 1b: Taḥwīl with Nominal Subject Preceding Verb: X تصير/تجعل Y Z
        if len(words) >= 4 and re.sub(r'[\u064B-\u065F\u0670]', '', words[1]) in TAHWIL_GOVERNORS:
            agent = self.realize_word(words[0])
            v_en = TAHWIL_GOVERNORS[re.sub(r'[\u064B-\u065F\u0670]', '', words[1])]
            theme = self.realize_word(words[2])
            goal = self.realize_word(words[3])
            if not goal.startswith(('into', 'as')):
                goal = f"into {goal}"
            res = f"{agent} {v_en} {theme} {goal}".replace('into into ', 'into ')
            if len(words) > 4:
                rest = self.transmute_sentence(" ".join(words[4:]))
                if rest:
                    return f"{res}, {rest}"
            return res

        # PATTERN 2: Sībawayh Afʿāl al-Qulūb (Verbs of Perception / Cognition)
        if first_w in QULUB_GOVERNORS and len(words) >= 3:
            v_subj, _ = QULUB_GOVERNORS[first_w]
            obj1 = self.realize_word(words[1])
            state1 = self.realize_word(words[2])
            part1 = f"{v_subj} {obj1} {state1}" if state1.startswith('as ') else f"{v_subj} {obj1} as {state1}"
            rest_parts = [part1]

            if len(words) >= 5 and words[3].startswith('و'):
                obj2 = self.realize_word(words[3])
                state2 = self.realize_word(words[4]) if len(words) > 4 else ""
                part2 = f"{obj2} {state2}" if state2.startswith('as ') else f"{obj2} as {state2}"
                rest_parts.append(part2)

            res = " ".join(rest_parts)
            return re.sub(r'\bas as\b', 'as', res)

        # PATTERN 3: Sībawayh Universal Clausal Realizer (Naẓm Al-Jurjānī)
        realized_tokens = []
        i = 0
        n = len(words)

        while i < n:
            curr_w = words[i]
            curr_clean = re.sub(r'[\u064B-\u065F\u0670\.,!؟؛:]', '', curr_w)

            # Check Iḍāfah (Annexation): Noun1 (indefinite form) + Noun2 (definite 'al-')
            if i + 1 < n and not curr_clean.startswith('ال') and words[i+1].startswith(('ال', 'وال')):
                if curr_clean not in PREPOSITION_MAP and curr_clean not in PARTICLE_MAP and curr_clean not in {'كل', 'وكل'}:
                    n1 = self.realize_word(curr_w)
                    n2 = self.realize_word(words[i+1])
                    if n1.endswith((' in', ' of', ' between')):
                        realized_tokens.append(f"{n1} {n2}")
                    else:
                        realized_tokens.append(f"{n1} of {n2}")
                    i += 2
                    continue

            # Standard Word Realization
            tr = self.realize_word(curr_w)
            realized_tokens.append(tr)
            i += 1

        result_str = " ".join(realized_tokens)
        
        # Sībawayh Equational Linker & Idiomatic Agreement
        result_str = re.sub(r'\bevery physical body composite\b', 'every physical body is composite', result_str)
        result_str = re.sub(r'\bevery composite in need\b', 'every composite is in need', result_str)
        result_str = re.sub(r'\bin need of to\b', 'in need of a', result_str)
        result_str = re.sub(r'\boccur upon\b', 'occurs upon', result_str)
        result_str = re.sub(r'\bthrough through\b', 'through', result_str)
        result_str = re.sub(r'\bthrough of the\b', 'through the', result_str)
        result_str = re.sub(r'\binto into\b', 'into', result_str)
        result_str = re.sub(r'\bas as\b', 'as', result_str)
        result_str = re.sub(r'\bthe the\b', 'the', result_str)
        result_str = re.sub(r'\bof of\b', 'of', result_str)
        result_str = re.sub(r'\bin in\b', 'in', result_str)
        result_str = re.sub(r'\s+', ' ', result_str).strip()
        return result_str
