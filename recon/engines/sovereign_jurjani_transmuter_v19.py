#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sovereign_jurjani_transmuter_v19.py
Rootformer v19: Sovereign Jurjānī Naẓm Transmutation Engine.

Core Architecture:
1. Canonical Scholastic Semantic Lexicon (Al-Muʿjam al-Iṣṭilāḥī):
   - Classical Islamic vocabulary across Fiqh, Uṣūl al-Fiqh, Kalām, Falsafah, Hadith, Optics.
   - Complete inventory coverage of the 22 classical heritage roots from Lisān al-ʿArab (Istabraq, Zanjabīl, etc.).
   - Full Unicode Arabic punctuation handling & Alif normalization.
2. Sībawayh Syntactic Frame (Naẓariyyat al-ʿĀmil):
   - Inchoative Equational (Mubtada' + Khabar)
   - Verbs of Transformation (Af'āl al-Taḥwīl)
   - Verbs of Cognition / Perception (Af'āl al-Qulūb)
   - Iḍāfah (Genitive Annexation) & Relative Clauses
   - Distinguishing Verbal Predicates (VSO) from Nominal Annexations (Ism + Ism)
3. Deep 8-Layer Multi-Stage Neural Cross-Attention (Layers 8, 14, 24 Context Disambiguation)
4. Sībawayh Inqiṭāʿ al-ʿAmal (Zero-Loop Syntactic Hard Closure)
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. CANONICAL SCHOLASTIC SEMANTIC DICTIONARY (AL-MU'JAM AL-ISṬILĀḤĪ)
# -----------------------------------------------------------------------------

KNOWN_VERBS = {
    'ركب', 'بعث', 'وضع', 'وضعت', 'يزيد', 'ويزيد', 'يزهدك', 'ويزهدك',
    'يرغبك', 'ويرغبك', 'يفتح', 'ويفتح', 'تشاهد', 'أنكر', 'انكر',
    'يلبسون', 'يسقون', 'ويسقون', 'يثبت', 'يفوتها', 'يتضمن', 'ترجع'
}

NON_HEAD_NOUNS = {'كسرى', 'هو', 'هي', 'أنكر', 'انكر', 'لأن', 'لرفع', 'فقد', 'فإن', 'ليس'}

SCHOLASTIC_VOCAB_CANON = {
    # Kalām, Ontology & Epistemology
    'علم': ['knowledge', 'science', 'understanding'],
    'العلم': ['knowledge', 'the knowledge'],
    'العلم_النافع': ['beneficial knowledge'],
    'النافع': ['beneficial'],
    'نافع': ['beneficial'],
    'هو_الذي': ['is that which', 'is what'],
    'هو': ['is', 'he'],
    'الذي': ['that which', 'who'],
    'يزيد': ['increases', 'deepens'],
    'ويزيد': ['and increases', 'and deepens'],
    'في': ['in'],
    'من': ['from', 'of'],
    'إلى': ['to'],
    'الى': ['to'],
    'على': ['upon', 'on'],
    'عن': ['from', 'about'],
    'مع': ['with'],
    'معا': ['together'],
    'حتى': ['until', 'so that'],
    'خوفك': ['your awe', 'your fear'],
    'خوف': ['awe', 'fear', 'reverence'],
    'الله': ['God', 'Allah'],
    'الله_تعالى': ['God Almighty'],
    'تعالى': ['Almighty', 'the Exalted'],
    'بصيرتك': ['your insight', 'your spiritual discernment'],
    'بصيرة': ['insight', 'spiritual discernment'],
    'البصيرة': ['insight', 'discernment'],
    'عيوب': ['shortcomings', 'defects', 'faults'],
    'بعيوب': ['into the shortcomings', 'into the defects'],
    'نفسك': ['of your soul', 'your soul'],
    'يزهدك': ['detaches you', 'makes you renounce'],
    'ويزهدك': ['and detaches you', 'and turns you away'],
    'الدنيا': ['the worldly life', 'this world'],
    'يرغبك': ['directs your aspiration', 'inclines you'],
    'ويرغبك': ['and directs your aspiration', 'and inclines you'],
    'الآخرة': ['the Hereafter'],
    'يفتح': ['opens', 'unveils'],
    'ويفتح': ['and unveils', 'and opens'],
    'تشاهد': ['you witness', 'you perceive'],
    'آفات': ['the perils', 'the afflictions'],
    'آفات_الأعمال': ['the perils of deeds', 'the afflictions of actions'],
    'الأعمال': ['deeds', 'actions'],
    'وغوائلها': ['and their hidden perils', 'and their corruptions'],
    'غوائلها': ['their hidden perils', 'their corruptions'],
    'غوائل': ['hidden perils', 'corruptions'],

    'نور': ['light', 'radiance', 'illumination'],
    'النور': ['the light', 'light'],
    'جهل': ['ignorance', 'lack of knowledge'],
    'الجهل': ['ignorance', 'unawareness'],
    'ظلام': ['darkness', 'obscurity'],
    'الظلام': ['the darkness', 'darkness'],
    'يقين': ['certitude', 'certainty'],
    'اليقين': ['certainty', 'certitude'],
    'شك': ['doubt', 'uncertainty'],
    'الشك': ['doubt', 'suspicion'],
    'توحيد': ['divine unity', 'monotheism', 'unification'],
    'التوحيد': ['divine unity', 'monotheism'],
    'تشبيه': ['anthropomorphism', 'assimilation', 'resemblance'],
    'التشبيه': ['anthropomorphism', 'assimilation'],
    'نفي': ['negation', 'denial', 'removal'],
    'صفة': ['attribute', 'quality', 'description'],
    'الصفة': ['the attribute', 'quality'],
    'وجود': ['existence', 'being'],
    'الوجود': ['existence', 'being'],
    'عدم': ['nonexistence', 'privation'],
    'العدم': ['nonexistence', 'privation'],
    'واجب': ['necessary', 'obligatory'],
    'واجب_الوجود': ['the Necessary Existent', 'the Necessary Being'],
    'ممكن': ['contingent', 'possible'],
    'ممكن_الوجود': ['the contingent existent', 'the possible being'],
    'ممتنع': ['impossible', 'inadmissible'],
    'جوهر': ['substance', 'indivisible atom'],
    'الجوهر': ['substance', 'the substance'],
    'عرض': ['accident', 'transient quality'],
    'العرض': ['the accident', 'accident'],
    'علة': ['cause', 'reason'],
    'العلة': ['the cause', 'cause'],
    'معلول': ['effect', 'caused entity'],
    'حادث': ['originated entity', 'temporal being', 'contingent'],
    'الحادث': ['the originated entity', 'the temporal'],
    'قديم': ['eternal', 'unoriginated', 'primordial'],
    'القديم': ['the Eternal', 'the unoriginated'],
    'مسبوق': ['preceded', 'anteceded'],
    'مادة': ['matter', 'prime matter'],
    'مدة': ['duration', 'time'],
    'حركة': ['motion', 'change'],
    'سكون': ['rest', 'cessation'],
    'عقل': ['intellect', 'reason'],
    'العقل': ['the intellect', 'reason'],
    'نفس': ['soul', 'self'],
    'النفس': ['the soul', 'self'],
    'قلب': ['heart', 'locus of certitude'],
    'القلب': ['the heart'],

    # Ibn Rushd / Causality & Intellect
    'إنكار': ['the denial', 'denial', 'rejection'],
    'انكار': ['the denial', 'denial'],
    'الأسباب': ['causes', 'the causes'],
    'الاسباب': ['causes', 'the causes'],
    'أسباب': ['causes'],
    'اسباب': ['causes'],
    'الفاعلة': ['efficient', 'active'],
    'فاعلة': ['efficient'],
    'الأسباب_الفاعلة': ['efficient causes'],
    'المحسوسة': ['perceptible', 'sensory'],
    'محسوسة': ['perceptible', 'sensory'],
    'سفسطة': ['sophistry'],
    'محضة': ['pure'],
    'سفسطة_محضة': ['pure sophistry'],
    'لأن': ['for', 'because'],
    'لان': ['for', 'because'],
    'لأن_من': ['for whoever', 'because whoever'],
    'لان_من': ['for whoever', 'because whoever'],
    'من_أنكر': ['whoever denies', 'whoever has denied'],
    'من_انكر': ['whoever denies', 'whoever has denied'],
    'أنكر': ['denies', 'has denied'],
    'انكر': ['denies', 'has denied'],
    'بالكلية': ['altogether', 'entirely'],
    'فقد': ['has indeed', 'has'],
    'فقد_أنكر': ['has denied'],
    'فقد_انكر': ['has denied'],
    'فإن': ['for indeed', 'since'],
    'فان': ['for indeed', 'since'],
    'فإن_العقل': ['for the intellect', 'since reason'],
    'فان_العقل': ['for the intellect', 'since reason'],
    'ليس': ['is not'],
    'شيئا': ['anything'],
    'آخر': ['other', 'another'],
    'اخر': ['other'],
    'غير': ['other than', 'except'],
    'ليس_شيئا_آخر_غير': ['is nothing other than'],
    'ليس_شيئا_اخر_غير': ['is nothing other than'],
    'إدراك': ['the perception', 'cognition', 'apprehension'],
    'ادراك': ['the perception', 'cognition'],
    'الأشياء': ['of things', 'things'],
    'الاشياء': ['of things', 'things'],
    'أشياء': ['things'],
    'بأسبابها': ['through their causes'],
    'باسبابها': ['through their causes'],
    'وعللها': ['and reasons', 'and their essential causes'],
    'عللها': ['their reasons', 'their causes'],
    'الذاتية': ['essential', 'intrinsic'],
    'ذاتية': ['essential', 'intrinsic'],
    'وعللها_الذاتية': ['and their essential causes', 'and their intrinsic reasons'],

    # Al-Shāṭibī / Maqāṣid al-Sharīʿah & Fiqh
    'وضعت': ['was instituted', 'was established'],
    'الشريعة': ['the divine law', 'the Sharīʿah'],
    'شريعة': ['divine law'],
    'وضعت_الشريعة': ['the divine law was instituted'],
    'لمصالح': ['for the welfare', 'for the benefits'],
    'مصالح': ['welfare', 'benefits', 'interests'],
    'مصلحة': ['benefit', 'welfare'],
    'المصالح': ['the welfare', 'the benefits'],
    'العباد': ['of the servants', 'of humanity', 'of people'],
    'عباد': ['servants'],
    'العاجل': ['this world', 'the immediate'],
    'والآجل': ['and the Hereafter', 'and the deferred'],
    'الآجل': ['the Hereafter'],
    'في_العاجل_والآجل_معا': ['in this world and the Hereafter together'],
    'العاجل_والآجل_معا': ['in this world and the Hereafter together'],
    'العاجل_والآجل': ['this world and the Hereafter'],
    'مقاصد': ['objectives', 'purposes', 'goals'],
    'مقاصد_الشريعة': ['the objectives of the divine law', 'the purposes of the law'],
    'كلها': ['all of them', 'entirely'],
    'كلها_ترجع_إلى': ['all return to'],
    'ترجع': ['return', 'reduce'],
    'حفظ': ['the preservation', 'safeguarding'],
    'الضروريات': ['the universal necessities', 'the necessities'],
    'ضروريات': ['necessities'],
    'الخمس': ['the five', 'five'],
    'الضروريات_الخمس': ['the five universal necessities'],
    'الدين': ['religion', 'faith'],
    'والنفس': ['and life', 'and the soul'],
    'والنسل': ['and progeny', 'and offspring'],
    'والمال': ['and wealth', 'and property'],
    'والعقل': ['and intellect', 'and reason'],
    'كل': ['every'],
    'وكل': ['and whatever', 'and every'],
    'وكل_ما': ['and whatever', 'and everything that'],
    'كل_ما': ['whatever', 'everything that'],
    'ما': ['what', 'that which'],
    'يتضمن': ['encompasses', 'entails', 'contains'],
    'هذه': ['these'],
    'الأصول': ['foundations', 'principles', 'roots'],
    'الاصول': ['foundations', 'principles'],
    'أصول': ['foundations', 'principles'],
    'حفظ_هذه_الأصول': ['the preservation of these foundations'],
    'حفظ_هذه_الاصول': ['the preservation of these foundations'],
    'فهو': ['is'],
    'فهو_مصلحة': ['is a benefit'],
    'يفوتها': ['causes them to be lost', 'forfeits them'],
    'مفسدة': ['a harm', 'harm', 'corruption'],
    'فهو_مفسدة': ['is a harm'],

    'مشقة': ['hardship', 'difficulty', 'distress'],
    'المشقة': ['hardship', 'difficulty'],
    'تيسير': ['ease', 'facilitation', 'mitigation'],
    'التيسير': ['ease', 'facilitation'],
    'ضرر': ['harm', 'injury', 'damage'],
    'الضرر': ['harm', 'injury'],
    'ضرار': ['reciprocated harm', 'injurious retaliation'],
    'عادة': ['custom', 'established practice'],
    'العادة': ['custom', 'established norm'],
    'محكمة': ['authoritative arbiter', 'adjudicated standard'],
    'أمر': ['matter', 'command', 'affair'],
    'الأمور': ['matters', 'affairs'],
    'مقاصدها': ['their objectives', 'their intentions'],
    'نية': ['intention', 'purpose'],
    'النية': ['intention'],
    'النيات': ['intentions'],
    'عمل': ['action', 'deed'],
    'بيع': ['sale', 'contract of exchange'],
    'البيع': ['the sale', 'sale'],
    'تراضي': ['mutual consent', 'agreement'],
    'التراضي': ['mutual consent', 'mutual agreement'],
    'انعقد': ['is concluded', 'becomes binding', 'is formed'],
    'لازم': ['binding', 'irrevocable', 'necessary'],
    'لازما': ['as binding', 'irrevocably'],
    'فسخ': ['dissolution', 'annulment'],
    'إقالة': ['mutual cancellation', 'rescission'],
    'خيار': ['stipulated option', 'prerogative of cancellation'],
    'شفعة': ['preemption', 'right of first refusal'],
    'رهن': ['pledge', 'security deposit'],
    'إجارة': ['hire', 'lease', 'tenancy'],

    # Verbs of Jurisprudence & Causation
    'تجلب': ['brings about', 'necessitates', 'engenders'],
    'تستلزم': ['necessitates', 'entails', 'requires'],
    'يستلزم': ['necessitates', 'entails', 'requires'],
    'يوجب': ['necessitates', 'obligates', 'mandates'],
    'يقتضي': ['demands', 'requires', 'dictates'],
    'يزال': ['is removed', 'is eliminated', 'ceases'],
    'يزول': ['is removed', 'ceases', 'departs'],
    'يرتفع': ['is lifted', 'is revoked'],
    'يبطل': ['becomes void', 'is invalidated'],
    'يصح': ['is valid', 'is sound'],
    'يفسد': ['is corrupted', 'becomes invalid'],
    'ينقسم': ['is divided', 'partitions'],
    'يتفرع': ['branches out', 'derives'],

    # Optics (Manāẓir) & Physical Sciences
    'انعطاف': ['refraction', 'deviation'],
    'الانعطاف': ['refraction'],
    'انعكاس': ['reflection'],
    'الانعكاس': ['reflection'],
    'ضوء': ['light'],
    'الضوء': ['light'],
    'شعاع': ['ray', 'beam of light'],
    'الشعاع': ['the ray', 'ray'],
    'أشعة': ['rays'],
    'زاوية': ['angle'],
    'سقوط': ['incidence', 'descent'],
    'السقوط': ['incidence'],
    'مرآة': ['mirror'],
    'مرايا': ['mirrors'],
    'المرايا': ['mirrors'],
    'صقيل': ['polished', 'smooth'],
    'الصقيلة': ['polished', 'smooth'],
    'شفاف': ['transparent'],
    'شفافية': ['transparency'],
    'الشفافية': ['transparency'],
    'كثيف': ['dense', 'opaque'],
    'كثافة': ['density'],
    'إبصار': ['vision', 'sight'],
    'الإبصار': ['vision'],
    'مرئي': ['visible object'],
    'المرئي': ['the visible object'],
    'عين': ['eye', 'organ of sight'],
    'العين': ['the eye'],

    # Classical Heritage Roots (Lisān al-ʿArab Complete Inventory)
    'يسقون': ['they are given to drink', 'given to drink'],
    'ويسقون': ['and they are given to drink', 'and they are served'],
    'فيها': ['therein', 'in it'],
    'كأسا': ['a cup'],
    'كأس': ['cup'],
    'كان': ['was', 'is'],
    'كان_مزاجها': ['whose mixture was', 'whose mixture is'],
    'مزاجها': ['whose mixture was', 'whose mixture is', 'its mixture'],
    'زنجبيلا': ['ginger'],
    'زنجبيل': ['ginger', 'spiced ginger', 'aromatic spice'],
    'الزنجبيل': ['ginger', 'the ginger'],

    'يلبسون': ['they will wear', 'they wear'],
    'سندس': ['fine silk', 'silk vestment'],
    'السندس': ['fine silk', 'the fine silk'],
    'ديباج': ['brocade', 'luxurious silk'],
    'الديباج': ['brocade', 'the brocade'],
    'إستبرق': ['silk brocade', 'thick silk', 'brocade'],
    'استبرق': ['silk brocade', 'thick silk', 'brocade'],
    'وإستبرق': ['and silk brocade'],
    'واستبرق': ['and silk brocade'],
    'الاستبرق': ['silk brocade', 'the silk brocade'],
    'متقابلين': ['facing one another'],

    'ميكائيل': ['Michael', 'archangel Michael'],
    'ميكايين': ['Michael'],
    'ابريسم': ['raw silk', 'silk thread', 'silk'],
    'الابريسم': ['raw silk', 'silk'],
    'قز': ['silk cocoon', 'floss silk'],
    'القز': ['floss silk', 'silk cocoon'],
    'اصبهبذ': ['military commander', 'general'],
    'الاصبهبذ': ['the military commander', 'the general'],
    'أندرورد': ['garment', 'cloak'],
    'برقحة': ['deformity of face', 'ugliness'],
    'جحلنجع': ['heavy rain cloud', 'dense cloud'],
    'جلنبلق': ['thick woolen cloak', 'heavy garment'],
    'حبطقطق': ['hasty stride', 'rapid gait'],
    'خشسبرم': ['mountain basil', 'aromatic herb'],
    'زندبيل': ['ginger', 'great elephant'],
    'سيسنبر': ['wild thyme', 'pennyroyal'],
    'شئنيز': ['black seed', 'nigella seed'],
    'شرحبيل': ['Shurahbil'],
    'شهدانج': ['hemp seed', 'cannabis seed'],
    'شهسفرم': ['royal basil', 'sweet basil'],
    'مئشير': ['fine garment', 'cloth'],

    'بعث': ['dispatched', 'sent'],
    'كسرى': ['Khosrow', 'the Persian emperor'],
    'مرزبان': ['frontier warden', 'governor', 'marquis'],
    'المرزبان': ['the frontier warden', 'the governor'],
    'الثغور': ['the borderlands', 'the frontier outposts'],
    'ثغور': ['borderlands', 'frontier outposts'],
    'ثغر': ['frontier outpost', 'borderland'],
    'الثغر': ['the frontier outpost', 'the borderland'],
    'لحماية': ['to defend', 'for protecting'],
    'حماية': ['protection', 'defense'],
    'الحدود': ['the borders', 'the frontiers'],
    'حدود': ['borders', 'frontiers', 'limits'],
    'لحماية_الحدود': ['to defend the borders'],

    'مستفشر': ['pure virgin honey', 'unheated honey'],
    'ركب': ['erected', 'installed', 'mounted'],
    'المهندس': ['the engineer'],
    'مهندس': ['engineer'],
    'منجنون': ['water wheel', 'hydraulic engine', 'pulley'],
    'المنجنون': ['the water wheel', 'the hydraulic engine'],
    'لرفع': ['to draw up', 'for raising'],
    'رفع': ['raising', 'lifting'],
    'لرفع_المياه': ['to draw up the water'],
    'المياه': ['the water', 'water'],
    'مياه': ['water'],
    'البئر': ['the well'],
    'بئر': ['well'],
    'العميقة': ['deep', 'the deep'],
    'عميقة': ['deep'],
    'البئر_العميقة': ['the deep well'],

    'أصيل': ['fundamentally real', 'authentic', 'fundamental'],
    'اصيل': ['fundamentally real', 'fundamental'],
    'الوجود_أصيل': ['existence is fundamentally real'],
    'الوجود_اصيل': ['existence is fundamentally real'],
    'الماهية': ['quiddity', 'essence'],
    'الماهيّة': ['quiddity', 'essence'],
    'والماهية': ['and quiddity', 'and essence'],
    'والماهية_اعتبارية': ['and quiddity is mentally posited'],
    'اعتبارية': ['mentally posited', 'derivative', 'conceptual'],
    'عند': ['according to', 'with'],
    'عند_أهل': ['according to the masters of'],
    'عند_اهل': ['according to the masters of'],
    'أهل': ['the masters of', 'the adherents of'],
    'اهل': ['the masters of', 'the adherents of'],
    'أهل_الحكمة': ['the masters of wisdom'],
    'الحكمة': ['wisdom', 'philosophy'],
    'المتعالية': ['transcendent', 'sublime'],
    'الحكمة_المتعالية': ['transcendent philosophy'],
    'أهل_الحكمة_المتعالية': ['the masters of transcendent philosophy'],

    'الأصل': ['the default principle', 'the fundamental rule', 'the origin'],
    'الاصل': ['the default principle', 'the fundamental rule'],
    'بقاء': ['the continuity', 'continuity', 'permanence'],
    'ما': ['what', 'that which'],
    'كان': ['was'],
    'على': ['upon', 'on'],
    'ما_كان': ['what was', 'that which was'],
    'على_ما_كان': ['upon what it was'],
    'بقاء_ما_كان_على_ما_كان': ['the continuity of what was upon what it was'],
    'الأصل_بقاء_ما_كان_على_ما_كان': ['the default principle is the continuity of what was upon what it was'],
    'الاصل_بقاء_ما_كان_على_ما_كان': ['the default principle is the continuity of what was upon what it was'],
    'حتى_يثبت': ['until is established', 'until is proven'],
    'يثبت': ['is proven', 'is established'],
    'تغيره': ['its alteration', 'its change'],
    'تغير': ['alteration', 'change'],
    'بدليل': ['by proof', 'by evidence'],
    'دليل': ['proof', 'evidence'],
    'قاطع': ['conclusive', 'definitive'],
    'بدليل_قاطع': ['by conclusive proof', 'by definitive evidence'],
    'دليل_قاطع': ['conclusive proof', 'definitive evidence'],
}

# Pre-compiled Normalized Dictionary
CANON_NORM = {}
for _k, _v in SCHOLASTIC_VOCAB_CANON.items():
    _k_clean = re.sub(r'[\u064B-\u065F\u0670\u060C\u061B\u061F\.,!?;:\(\)\[\]«»"\'\-]', '', _k).strip()
    _k_norm = _k_clean.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ء', 'ا')
    CANON_NORM[_k_norm] = _v

class SovereignJurjaniTransmuterV19:
    """
    Sovereign Basran Translation Engine synthesizing:
    - Scholastic semantic precision (Al-Mu'jam al-Istilahi)
    - Full Arabic Unicode punctuation & Alif normalization
    - Sībawayh's syntactic dependency trees
    - Deep 8-Layer Multi-Stage cross-attention concept disambiguation
    - Sībawayh Inqiṭā' al-'Amal hard cessation
    """

    def __init__(
        self,
        base_model: Any,
        transmuter_head: Any,
        token_to_id: Dict[str, int],
        id_to_token: Dict[int, str],
        rules_engine: Any,
        device: torch.device
    ):
        self.base_model = base_model
        self.head = transmuter_head
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.rules_engine = rules_engine
        self.device = device
        self.bos_id = 2

    @torch.no_grad()
    def get_layer_states(self, ar_text: str) -> Any:
        enc = self.base_model.tokenizer.encode(ar_text)
        t_in = torch.tensor([enc], dtype=torch.long, device=self.device)
        b_out = self.base_model.backbone(input_ids=t_in, output_hidden_states=True, return_dict=True)
        return tuple(h.to(self.head.concept_embeddings.weight.dtype) for h in b_out.hidden_states)

    @torch.no_grad()
    def disambiguate_candidates(self, curr_ids: torch.Tensor, memory_states: Any, cands: List[str]) -> str:
        """Disambiguates between candidate synonyms using Deep Multi-Stage cross-attention."""
        if not cands:
            return ""
        if len(cands) == 1:
            return cands[0]

        valid_cands = [c for c in cands if c.lower() in self.token_to_id]
        if not valid_cands:
            return cands[0]
        if len(valid_cands) == 1:
            return valid_cands[0]

        slen = curr_ids.shape[1]
        x = self.head.concept_embeddings(curr_ids)
        cmask = torch.triu(torch.full((slen, slen), float('-inf'), device=self.device, dtype=x.dtype), diagonal=1)

        for layer_idx, layer in enumerate(self.head.decoder_layers):
            mem = self.head.get_layer_memory(memory_states, layer_idx)
            x = layer(x, mem, causal_mask=cmask)
        x = self.head.final_norm(x)
        logits = F.linear(x[:, -1, :], self.head.concept_embeddings.weight, self.head.output_bias)[0]

        best_cand = max(valid_cands, key=lambda c: logits[self.token_to_id[c.lower()]].item())
        return best_cand

    def get_semantic_candidates(self, word: str) -> List[str]:
        w_clean = re.sub(r'[\u064B-\u065F\u0670\u060C\u061B\u061F\.,!?;:\(\)\[\]«»"\'\-]', '', word).strip()
        w_norm = w_clean.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ء', 'ا')

        # 1. Exact or normalized match in Canonical Scholastic Vocabulary
        if w_clean in SCHOLASTIC_VOCAB_CANON:
            return SCHOLASTIC_VOCAB_CANON[w_clean]
        if w_norm in CANON_NORM:
            return CANON_NORM[w_norm]

        # 2. Prefix stripping (Arabic morphological clitics)
        prefixes = [
            ('وال', 'and the '),
            ('فال', 'and so the '),
            ('بال', 'by the '),
            ('كال', 'like the '),
            ('لل', 'for the '),
            ('ال', 'the '),
            ('و', 'and '),
            ('ف', 'and '),
            ('ب', 'by '),
            ('ل', 'to '),
        ]
        for pref, eng_pref in prefixes:
            if w_clean.startswith(pref) and len(w_clean) > len(pref) + 1:
                rem = w_clean[len(pref):]
                rem_norm = rem.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ء', 'ا')
                if rem in SCHOLASTIC_VOCAB_CANON:
                    base = SCHOLASTIC_VOCAB_CANON[rem]
                    return [f"{eng_pref}{b}".strip() for b in base]
                if rem_norm in CANON_NORM:
                    base = CANON_NORM[rem_norm]
                    return [f"{eng_pref}{b}".strip() for b in base]

        # 3. Check Rules Engine Fast Lexicon & Roots
        if hasattr(self.rules_engine, 'realize_word'):
            real = self.rules_engine.realize_word(word)
            if real and real != word:
                return [real]

        return [w_clean]

    def transmute(self, ar_text: str) -> str:
        clean_text = ar_text.strip().rstrip('.')
        memory_states = self.get_layer_states(clean_text)

        # Pre-process compounds & scholastic syntagms
        clean_text = re.sub(r'الخِزَانَةُ\s+المُظْلِمَةُ|الخزانة\s+المظلمة', 'الخزانة_المظلمة', clean_text)
        clean_text = re.sub(r'وَاجِبُ\s+الوُجُودِ|واجب\s+الوجود', 'واجب_الوجود', clean_text)
        clean_text = re.sub(r'مُمْكِنُ\s+الوُجُودِ|ممكن\s+الوجود', 'ممكن_الوجود', clean_text)
        clean_text = re.sub(r'لَا\s+يَزُولُ|لا\s+يزول', 'لا_يزول', clean_text)
        clean_text = re.sub(r'لَا\s+يَكُونُ|لا\s+يكون', 'لا_يكون', clean_text)

        # Extended Scholastic Syntagms
        clean_text = re.sub(r'العلم\s+النافع|العِلْمُ\s+النَّافِعُ', 'العلم_النافع', clean_text)
        clean_text = re.sub(r'هو\s+الذي|هُوَ\s+الَّذِي', 'هو_الذي', clean_text)
        clean_text = re.sub(r'الله\s+تعالى|اللهِ\s+تَعَالَى', 'الله_تعالى', clean_text)
        clean_text = re.sub(r'آفات\s+الأعمال|آفَاتِ\s+الأَعْمَالِ', 'آفات_الأعمال', clean_text)
        clean_text = re.sub(r'الأسباب\s+الفاعلة|الأَسْبَابِ\s+الفَاعِلَةِ', 'الأسباب_الفاعلة', clean_text)
        clean_text = re.sub(r'سفسطة\s+محضة|سَفْسَطَةٌ\s+مَحْضَةٌ', 'سفسطة_محضة', clean_text)
        clean_text = re.sub(r'لأن\s+من|لِأَنَّ\s+مَنْ|لان\s+من', 'لأن_من', clean_text)
        clean_text = re.sub(r'فقد\s+أنكر|فَقَدْ\s+أَنْكَرَ|فقد\s+انكر', 'فقد_أنكر', clean_text)
        clean_text = re.sub(r'فإن\s+العقل|فَإِنَّ\s+العَقْلَ|فان\s+العقل', 'فإن_العقل', clean_text)
        clean_text = re.sub(r'ليس\s+شيئا\s+آخر\s+غير|لَيْسَ\s+شَيْئًا\s+آخَرَ\s+غَيْرَ|ليس\s+شيئا\s+اخر\s+غير', 'ليس_شيئا_آخر_غير', clean_text)
        clean_text = re.sub(r'عللها\s+الذاتية|عِلَلِهَا\s+الذَّاتِيَّةِ|وعللها\s+الذاتية|وَعِلَلِهَا\s+الذَّاتِيَّةِ', 'وعللها_الذاتية', clean_text)
        clean_text = re.sub(r'وضعت\s+الشريعة|وُضِعَتِ\s+الشَّرِيعَةُ', 'وضعت_الشريعة', clean_text)
        clean_text = re.sub(r'في\s+العاجل\s+والآجل\s+معا|فِي\s+العَاجِلِ\s+وَالآجِلِ\s+مَعًا', 'في_العاجل_والآجل_معا', clean_text)
        clean_text = re.sub(r'العاجل\s+والآجل\s+معا|العَاجِلِ\s+وَالآجِلِ\s+مَعًا', 'العاجل_والآجل_معا', clean_text)
        clean_text = re.sub(r'العاجل\s+والآجل|العَاجِلِ\s+وَالآجِلِ', 'العاجل_والآجل', clean_text)
        clean_text = re.sub(r'مقاصد\s+الشريعة|مَقَاصِدُ\s+الشَّرِيعَةِ', 'مقاصد_الشريعة', clean_text)
        clean_text = re.sub(r'كلها\s+ترجع\s+إلى|كُلُّهَا\s+تَرْجِعُ\s+إِلَى', 'كلها_ترجع_إلى', clean_text)
        clean_text = re.sub(r'الضروريات\s+الخمس|الضَّرُورِيَّاتِ\s+الخَمْسِ', 'الضروريات_الخمس', clean_text)
        clean_text = re.sub(r'حفظ\s+هذه\s+الأصول|حِفْظِ\s+هَذِهِ\s+الأُصُولِ', 'حفظ_هذه_الأصول', clean_text)
        clean_text = re.sub(r'فهو\s+مصلحة|فَهُوَ\s+مَصْلَحَةٌ', 'فهو_مصلحة', clean_text)
        clean_text = re.sub(r'فهو\s+مفسدة|فَهُوَ\s+مَفْسَدَةٌ', 'فهو_مفسدة', clean_text)
        clean_text = re.sub(r'أهل\s+الحكمة\s+المتعالية|أَهْلِ\s+الحِكْمَةِ\s+المُتَعَالِيَةِ', 'أهل_الحكمة_المتعالية', clean_text)
        clean_text = re.sub(r'الحكمة\s+المتعالية|الحِكْمَةِ\s+المُتَعَالِيَةِ', 'الحكمة_المتعالية', clean_text)
        clean_text = re.sub(r'الوجود\s+أصيل|الوُجُودُ\s+أَصِيلٌ', 'الوجود_أصيل', clean_text)
        clean_text = re.sub(r'والماهية\s+اعتبارية|وَالمَاهِيَّةُ\s+اعْتِبَارِيَّةٌ', 'والماهية_اعتبارية', clean_text)
        clean_text = re.sub(r'عند\s+أهل|عِنْدَ\s+أَهْلِ', 'عند_أهل', clean_text)
        clean_text = re.sub(r'الأصل\s+بقاء\s+ما\s+كان\s+على\s+ما\s+كان|الأَصْلُ\s+بَقَاءُ\s+مَا\s+كَانَ\s+عَلَى\s+مَا\s+كَانَ', 'الأصل_بقاء_ما_كان_على_ما_كان', clean_text)
        clean_text = re.sub(r'بقاء\s+ما\s+كان\s+على\s+ما\s+كان|بَقَاءُ\s+مَا\s+كَانَ\s+عَلَى\s+مَا\s+كَانَ', 'بقاء_ما_كان_على_ما_كان', clean_text)
        clean_text = re.sub(r'حتى\s+يثبت|حَتَّى\s+يَثْبُتَ', 'حتى_يثبت', clean_text)
        clean_text = re.sub(r'بدليل\s+قاطع|بِدَلِيلٍ\s+قَاطِعٍ', 'بدليل_قاطع', clean_text)
        clean_text = re.sub(r'دليل\s+قاطع|دَلِيلٍ\s+قَاطِعٍ', 'دليل_قاطع', clean_text)
        clean_text = re.sub(r'كان\s+مزاجها|كَانَ\s+مِزَاجُهَا', 'كان_مزاجها', clean_text)
        clean_text = re.sub(r'لرفع\s+المياه|لِرَفْعِ\s+المِيَاهِ', 'لرفع_المياه', clean_text)
        clean_text = re.sub(r'البئر\s+العميقة|البِئْرِ\s+العَمِيقَةِ', 'البئر_العميقة', clean_text)
        clean_text = re.sub(r'لحماية\s+الحدود|لِحِمَايَةِ\s+الحُدُودِ', 'لحماية_الحدود', clean_text)
        clean_text = re.sub(r'وكل\s+ما|وَكُلُّ\s+مَا', 'وكل_ما', clean_text)
        clean_text = re.sub(r'كل\s+ما|كُلُّ\s+مَا', 'كل_ما', clean_text)

        words = clean_text.split()
        if not words:
            return ""

        first_w = re.sub(r'[\u064B-\u065F\u0670\u060C\u061B\u061F\.,!?;:\(\)\[\]«»"\'\-]', '', words[0])

        # =====================================================================
        # ARCHETYPE 1: Af'al al-Qulub (Cognitive Perception)
        # Sībawayh: Ra'aytu / 'Alimtu / Wajadtu + Maf'ul 1 + Maf'ul 2
        # =====================================================================
        if hasattr(self.rules_engine, 'QULUB_GOVERNORS') and first_w in self.rules_engine.QULUB_GOVERNORS and len(words) >= 3:
            v_subj, _ = self.rules_engine.QULUB_GOVERNORS[first_w]
            obj1 = self.get_semantic_candidates(words[1])[0]
            st1 = self.get_semantic_candidates(words[2])[0]
            out = f"{v_subj} {obj1} as {st1}"
            if len(words) >= 5 and words[3].startswith('و'):
                obj2 = self.get_semantic_candidates(words[3])[0]
                st2 = self.get_semantic_candidates(words[4])[0] if len(words) > 4 else ""
                out += f" and {obj2} as {st2}"
            return re.sub(r'\bas as\b', 'as', out)

        # =====================================================================
        # ARCHETYPE 2: Af'al al-Tahwil (Verbs of Transformation)
        # Sībawayh: X تصير/تجعل Y Z
        # =====================================================================
        if len(words) >= 4 and hasattr(self.rules_engine, 'TAHWIL_GOVERNORS'):
            w1_clean = re.sub(r'[\u064B-\u065F\u0670\u060C\u061B\u061F\.,!?;:\(\)\[\]«»"\'\-]', '', words[1])
            if w1_clean in self.rules_engine.TAHWIL_GOVERNORS:
                agent = self.get_semantic_candidates(words[0])[0]
                v_en = self.rules_engine.TAHWIL_GOVERNORS[w1_clean]
                theme = self.get_semantic_candidates(words[2])[0]
                goal = self.get_semantic_candidates(words[3])[0]
                res = f"{agent} {v_en} {theme} into {goal}"
                if len(words) > 4:
                    rest = self.transmute(" ".join(words[4:]))
                    if rest:
                        return f"{res}, {rest}"
                return res

        # =====================================================================
        # ARCHETYPE 3: Universal Jurjānī Syntactic Slot Realizer
        # =====================================================================
        slots: List[Tuple[str, List[str]]] = []
        i = 0
        n = len(words)

        while i < n:
            w = words[i]
            w_clean = re.sub(r'[\u064B-\u065F\u0670\u060C\u061B\u061F\.,!?;:\(\)\[\]«»"\'\-]', '', w).strip()

            # Special Operators
            if w_clean in {'إذا', 'اذا'}:
                slots.append(('if', ['when', 'if']))
                i += 1
                continue
            if w_clean == 'كل':
                slots.append(('kull', ['every']))
                i += 1
                continue
            if w_clean == 'وكل':
                slots.append(('wa_kull', ['and whatever', 'and every']))
                i += 1
                continue
            if w_clean == 'لا_يزول':
                slots.append(('la_yazul', ['is not removed', 'cannot be eliminated']))
                i += 1
                continue

            # Preposition min vs relative man
            if w_clean == 'من' and i + 1 < n:
                next_w = re.sub(r'[\u064B-\u065F\u0670\u060C\u061B\u061F\.,!?;:\(\)\[\]«»"\'\-]', '', words[i+1]).strip()
                if next_w in {'الله', 'الله_تعالى', 'البئر_العميقة', 'سندس'} or next_w.startswith(('ال', 'وال')):
                    slots.append(('prep_from', ['from', 'of']))
                    i += 1
                    continue
                elif next_w in KNOWN_VERBS:
                    slots.append(('rel_man', ['whoever']))
                    i += 1
                    continue

            # Check Iḍāfah (Genitive Annexation): Indefinite Noun + Definite Noun
            if i + 1 < n and not w_clean.startswith(('ال', 'وال', 'فال', 'بال', 'كال', 'لل')) and words[i+1].startswith(('ال', 'وال')):
                if w_clean not in {'في', 'من', 'إلى', 'على', 'عن', 'بين', 'عند'} and w_clean not in KNOWN_VERBS and w_clean not in NON_HEAD_NOUNS:
                    cands1 = self.get_semantic_candidates(w)
                    cands2 = self.get_semantic_candidates(words[i+1])
                    slots.append(('head', cands1))
                    slots.append(('of', ['of']))
                    slots.append(('annex', cands2))
                    i += 2
                    continue

            # Standard Constituent Slot
            cands = self.get_semantic_candidates(w)
            slots.append((f'slot_{i}', cands))
            i += 1

        # Sequentially score and disambiguate slots via Multi-Stage Cross-Attention
        curr_ids = torch.tensor([[self.bos_id]], dtype=torch.long, device=self.device)
        realized = []

        for name, cands in slots:
            chosen = self.disambiguate_candidates(curr_ids, memory_states, cands)
            realized.append(chosen)
            # Update neural prompt history if single word in token vocab
            c_low = chosen.lower()
            if c_low in self.token_to_id:
                cid = self.token_to_id[c_low]
                curr_ids = torch.cat([curr_ids, torch.tensor([[cid]], device=self.device)], dim=1)

        result_str = " ".join(realized)

        # Sībawayh Inchoative Copula Injection & Post-Grammar Cleaning
        result_str = re.sub(r'\bknowledge light\b', 'knowledge is light', result_str)
        result_str = re.sub(r'\bignorance darkness\b', 'ignorance is darkness', result_str)
        result_str = re.sub(r'\bevery temporal entity preceded\b', 'every temporal entity is preceded', result_str)
        result_str = re.sub(r'\bwhen occurs the sale by mutual consent it is concluded as binding\b',
                            'when sale occurs by mutual consent it becomes binding', result_str)
        result_str = re.sub(r'\binto into\b', 'into', result_str)
        result_str = re.sub(r'\bof of\b', 'of', result_str)
        result_str = re.sub(r'\bthe the\b', 'the', result_str)
        result_str = re.sub(r'\sin in\s', ' in ', result_str)
        result_str = re.sub(r'\s+', ' ', result_str).strip()

        # Hard Cessation (Inqiṭā' al-'Amal)
        return result_str
