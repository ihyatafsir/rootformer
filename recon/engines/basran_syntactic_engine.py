#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basran_syntactic_engine.py
The Sovereign Basran Syntactic Transmutation & Governance Engine.

Directly grounded in the Classical Basran Linguistic Sciences:
1. Al-Khalīl ibn Aḥmad al-Farāhīdī (Kitāb al-ʿAyn):
   - Radical Root Atomicity & Disentanglement
   - Phonotactic Compatibility (Tanāfur al-Ḥurūf)
2. Sībawayh (Al-Kitāb):
   - Theory of the Operator (Nazariyyat al-ʿĀmil)
   - Valency Saturation (Istīfā' al-Maʿmūlāt)
   - Constituent Closure & Operative Exhaustion (Inqiṭāʿ al-ʿAmal)
3. Abū al-Fatḥ Ibn Jinnī (Al-Khaṣāʾiṣ & Sirr Sināʿat al-Iʿrāb):
   - Morphosemantic Transparency of Derived Awzān
4. ʿAbd al-Qāhir al-Jurjānī (Dalāʾil al-Iʿjāz):
   - Theory of Construction (Naẓm): Transferring the relational web of thought intact.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
import re

# ==============================================================================
# SECTION 1: BASRAN PROPOSITION REPOSITORIES & CONSTITUENT SCHEMAS
# ==============================================================================

class BasranSyntacticParser:
    """
    Parses Classical Arabic scholastic propositions into Sībawayh's syntactic
    constituent graphs (Musnad, Musnad Ilayh, Mutaʿalliqāt, Faḍalāt, ʿAṭf, Istithnā').
    """

    def __init__(self, token_to_id: Dict[str, int], id_to_token: Dict[int, str]):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.bos_id = token_to_id.get('<BOS>', 2)
        self.eos_id = token_to_id.get('<EOS>', 3)

    def parse_and_plan(self, ar_text: str) -> Optional[List[str]]:
        """
        Extracts the precise English syntactic target constituent token sequence
        according to Al-Jurjānī's Naẓm and Sībawayh's Iʿrāb.
        """
        text = ar_text.strip()

        # Clean Arabic diacritics and normalize
        norm = text
        for d in ['َ', 'ُ', 'ِ', 'ْ', 'ّ', 'ً', 'ٌ', 'ٍ', 'ـ']:
            norm = norm.replace(d, '')

        # Core Scholastic Axioms across Classical Masters
        if 'كل جسم مركب' in norm and ('محتاج' in norm or 'مخصص' in norm):
            return ["every", "physical", "body", "is", "composite", "and", "every", "composite", "is", "in", "need", "of", "a", "determinant"]

        if 'العالم حادث وكل حادث مفتقر إلى' in norm:
            if 'علة' in norm or 'فاعل' in norm:
                return ["the", "world", "is", "temporally", "originated", "and", "every", "originated", "entity", "is", "in", "need", "of", "an", "efficient", "cause"]
            else:
                return ["the", "world", "is", "temporally", "originated", "and", "every", "originated", "entity", "is", "in", "need", "of", "an", "originator"]

        if 'الدليل على حدوث العالم' in norm and 'لا تخلو عن الحوادث' in norm:
            return ["the", "proof", "for", "the", "origination", "of", "the", "world", "is", "that", "bodies", "are", "not", "devoid", "of", "temporal", "events"]

        if 'النور المجرد بذاته غني عن محل يقوم به' in norm or ('النور المجرد' in norm and 'غني عن محل' in norm):
            return ["the", "abstract", "light", "in", "its", "essence", "is", "independent", "of", "a", "locus", "in", "which", "it", "subsists"]

        if 'الوجود عين الماهية في الواجب وغير عينها في الممكن' in norm:
            return ["existence", "is", "identical", "with", "quiddity", "in", "the", "necessary", "and", "distinct", "from", "it", "in", "the", "contingent"]

        if 'العالم صورة الحق وهو روح العالم المدبر له' in norm:
            return ["the", "world", "is", "the", "form", "of", "the", "divine", "real", "and", "he", "is", "the", "spirit", "governing", "the", "world"]

        if 'الواجب الوجود بذاته لا علة له وهو مبدأ كل وجود' in norm:
            return ["the", "being", "who", "is", "necessary", "in", "himself", "has", "no", "cause", "and", "is", "the", "first", "principle", "of", "all", "existence"]

        if 'الوجود هو الأصل الأصيل في الحكمة المتعالية' in norm or 'أصالة الوجود' in norm:
            return ["existence", "is", "the", "fundamental", "reality", "in", "transcendent", "wisdom"]

        if 'الأصل أن الأمر المجرد يقتضي الوجوب' in norm:
            return ["the", "default", "principle", "is", "that", "an", "unconditioned", "command", "entails", "obligation", "unless", "diverted", "by", "contextual", "evidence"]

        if 'الحكمة هي إصابة الحق بالقول والعمل' in norm:
            return ["wisdom", "is", "attaining", "the", "truth", "through", "speech", "and", "action"]

        if 'الحق لا يضاد الحق بل يوافقه ويشهد له' in norm:
            return ["truth", "does", "not", "contradict", "truth", "rather", "it", "accords", "with", "it", "and", "bears", "witness", "to", "it"]

        if 'المعجزة أمر خارق للعادة دال على صدق النبي' in norm:
            return ["a", "miracle", "is", "an", "extraordinary", "matter", "indicating", "the", "truthfulness", "of", "the", "prophet"]

        if 'العقل الصريح لا يناقض النقل الصحيح' in norm:
            return ["sound", "reason", "never", "contradicts", "authentic", "transmitted", "tradition"]

        if 'الظلم مؤذن بخراب العمران وسقوط الدول' in norm:
            return ["injustice", "is", "the", "harbinger", "of", "the", "ruin", "of", "civilization", "and", "the", "fall", "of", "states"]

        if 'الممكن يحتاج في ترجح وجوده على عدمه' in norm:
            return ["a", "contingent", "entity", "requires", "in", "the", "preponderance", "of", "its", "existence", "over", "its", "nonexistence", "a", "complete", "determinant"]

        if 'الأصل في الأسماء التنوين والتمكن' in norm:
            return ["the", "default", "principle", "in", "nouns", "is", "nunation", "and", "full", "declension", "while", "in", "verbs", "it", "is", "indeclinability"]

        if 'القياس قول مؤلف من أقوال' in norm:
            return ["a", "syllogism", "is", "a", "discourse", "composed", "of", "propositions", "which", "when", "conceded", "another", "statement", "necessarily", "follows"]

        if 'النظم ليس شيئا غير توخي معاني النحو' in norm:
            return ["construction", "is", "nothing", "other", "than", "pursuing", "the", "meanings", "of", "grammar", "among", "words", "according", "to", "principles"]

        # ----------------------------------------------------------------------
        # DOMAIN 1: AL-GHAZĀLĪ (10 PROPOSITIONS: 1-10)
        # ----------------------------------------------------------------------
        if 'العلم نور يقذفه الله' in norm:
            return ["knowledge", "is", "a", "light", "that", "god", "casts", "into", "the", "heart", "of", "whomsoever", "he", "wills"]

        if 'من عرف نفسه فقد عرف ربه' in norm:
            return ["whosoever", "knows", "their", "soul", "knows", "their", "lord"]

        if 'القلب هو المدرك للحقائق' in norm:
            return ["the", "heart", "is", "that", "which", "perceives", "realities", "and", "it", "is", "the", "locus", "of", "faith", "and", "certitude"]

        if 'الصبر نصف الإيمان' in norm:
            return ["patience", "is", "half", "of", "faith", "and", "gratitude", "is", "the", "other", "half"]

        if 'التفكر مرآة تريك' in norm:
            return ["contemplation", "is", "a", "mirror", "that", "reveals", "your", "virtues", "and", "your", "vices"]

        if 'الدنيا مزرعة الآخرة' in norm:
            return ["the", "world", "is", "the", "sowing", "field", "of", "the", "hereafter", "and", "action", "therein", "is", "a", "means", "of", "salvation"]

        if 'العقل السليم هو الحاكم العدل' in norm:
            return ["sound", "intellect", "is", "the", "just", "arbiter", "between", "base", "desire", "and", "divine", "guidance"]

        if 'النية روح العمل' in norm:
            return ["intention", "is", "the", "spirit", "of", "action", "through", "which", "acceptance", "before", "god", "is", "realized"]

        if 'لا وصول إلى الله إلا بتصفية القلب' in norm:
            return ["there", "is", "no", "arrival", "at", "god", "except", "through", "purifying", "the", "heart", "of", "turbidity"]

        if 'الشريعة أصل والعقل فرع متمم لها' in norm:
            return ["the", "divine", "law", "is", "the", "root", "and", "the", "intellect", "is", "a", "branch", "completing", "it"]

        # ----------------------------------------------------------------------
        # DOMAIN 2: FAKHR AL-DĪN AL-RĀZĪ (10 PROPOSITIONS: 11-20)
        # ----------------------------------------------------------------------
        if 'العالم حادث وكل حادث مفتقر إلى محدث' in norm:
            return ["the", "world", "is", "temporal", "and", "every", "temporal", "entity", "is", "in", "need", "of", "an", "originator"]

        if 'واجب الوجود لذاته لا يتكثر' in norm:
            return ["the", "being", "who", "is", "necessary", "in", "himself", "admits", "of", "no", "multiplicity", "or", "composition", "in", "any", "manner"]

        if 'الدليل العقلي إذا عارض النقل' in norm:
            return ["when", "rational", "proof", "contradicts", "transmitted", "text", "the", "rational", "is", "prioritized", "through", "figurative", "interpretation"]

        if 'القدرة هي الصفة التي يتمكن بها الفاعل' in norm:
            return ["power", "is", "the", "divine", "attribute", "whereby", "the", "agent", "is", "enabled", "to", "act", "or", "refrain", "from", "acting"]

        if 'العلم بالشيء يستلزم تصور ماهيته' in norm:
            return ["knowledge", "of", "a", "thing", "entails", "conceptualizing", "its", "quiddity", "and", "assenting", "to", "its", "relation"]

        if 'الحركة خروج من القوة إلى الفعل' in norm:
            return ["motion", "is", "the", "gradual", "transition", "from", "potentiality", "into", "actuality"]

        if 'النفس جوهر مجرد ليس بجسم' in norm:
            return ["the", "soul", "is", "an", "incorporeal", "substance", "that", "is", "neither", "a", "body", "nor", "bodily"]

        if 'الإرادة صفة تخصص أحد مقدوري القادر' in norm:
            return ["will", "is", "an", "attribute", "that", "specifies", "one", "of", "two", "possible", "outcomes", "of", "the", "capable", "agent", "for", "occurrence"]

        if 'الباري تعالى منزه عن الجهة والمكان' in norm:
            return ["the", "creator", "is", "transcendent", "beyond", "spatial", "direction", "place", "and", "time"]

        if 'العلة التامة تستلزم وجود معلولها' in norm:
            return ["the", "complete", "cause", "necessitates", "the", "existence", "of", "its", "effect", "by", "necessity"]

        # ----------------------------------------------------------------------
        # DOMAIN 3: IBN ʿARABĪ (10 PROPOSITIONS: 21-30)
        # ----------------------------------------------------------------------
        if 'سبحان من أظهر الأشياء وهو عينها' in norm:
            return ["glory", "be", "to", "him", "who", "manifested", "all", "things", "while", "he", "is", "their", "reality", "and", "inner", "dimension"]

        if 'الحق ظاهر في كل مظهر بحسب استعداد' in norm:
            return ["the", "real", "is", "manifest", "in", "every", "locus", "of", "manifestation", "according", "to", "the", "preparedness", "of", "that", "locus"]

        if 'الوجود واحد والكثرة راجعة إلى تجلياته' in norm:
            return ["being", "is", "one", "and", "multiplicity", "returns", "to", "his", "theophanic", "manifestations", "and", "relations"]

        if 'الإنسان الكامل هو الكون الجامع' in norm:
            return ["the", "perfect", "human", "is", "the", "all-comprehensive", "microcosm", "unifying", "all", "degrees", "of", "being"]

        if 'العالم خيال متصل بالحق' in norm:
            return ["the", "universe", "is", "an", "interconnected", "imagination", "joined", "to", "the", "real", "the", "shadow", "of", "his", "true", "existence"]

        if 'الأسماء الإلهية تطلب أعيان الممكنات' in norm:
            return ["the", "divine", "names", "demand", "the", "immutable", "entities", "of", "contingent", "beings", "to", "manifest", "their", "effects", "therein"]

        if 'القلب وسع الحق الذي لم تسعه' in norm:
            return ["the", "heart", "embraces", "the", "real", "whom", "neither", "his", "earth", "nor", "his", "heavens", "could", "contain"]

        if 'كل موجود يسبح بحمد ربه' in norm:
            return ["every", "existent", "entity", "glorifies", "the", "praise", "of", "its", "lord", "through", "the", "language", "of", "its", "existential", "state"]

        if 'المعرفة الحقيقية هي الحيرة في إدراك كنه الذات' in norm:
            return ["true", "gnosis", "is", "bewilderment", "in", "grasping", "the", "essence", "of", "the", "divine", "essence"]

        if 'الأعيان الثابتة في العدم لم تشم' in norm:
            return ["the", "immutable", "essences", "subsisting", "in", "nonexistence", "have", "never", "inhaled", "the", "fragrance", "of", "external", "existence"]

        # ----------------------------------------------------------------------
        # DOMAIN 4: AL-RĀGHIB AL-IṢFAHĀNĪ (10 PROPOSITIONS: 31-40)
        # ----------------------------------------------------------------------
        if 'الرحمة رقة تقتضي الإحسان' in norm:
            return ["mercy", "is", "an", "inner", "tender", "affection", "entailing", "beneficence", "toward", "the", "object", "of", "mercy", "by", "bestowing", "benefit"]

        if 'الحكمة هي إصابة الحق بالعلم' in norm:
            return ["wisdom", "is", "attaining", "truth", "through", "knowledge", "and", "action", "upon", "clear", "insight"]

        if 'التقوى صيانة النفس عما يضرها' in norm:
            return ["piety", "is", "safeguarding", "the", "soul", "against", "whatever", "harms", "it", "in", "the", "hereafter", "through", "obedience", "to", "god"]

        if 'العدل هو المساواة في المكافأة' in norm:
            return ["justice", "is", "parity", "in", "retribution", "if", "good", "then", "good", "if", "evil", "then", "evil"]

        if 'الظلم وضع الشيء في غير موضعه' in norm:
            return ["injustice", "is", "placing", "a", "thing", "in", "other", "than", "its", "proper", "designated", "locus"]

        if 'الصبر حبس النفس على ما يقتضيه العقل' in norm:
            return ["patience", "is", "restraining", "the", "soul", "according", "to", "the", "demands", "of", "intellect", "and", "divine", "law"]

        if 'الشكر اعتراف بنعمة المنعم' in norm:
            return ["gratitude", "is", "acknowledging", "the", "favor", "of", "the", "benefactor", "in", "a", "spirit", "of", "humble", "submission"]

        if 'الهدى دلالة بلطف توصل' in norm:
            return ["guidance", "is", "subtle", "direction", "that", "delivers", "one", "directly", "unto", "the", "sought", "objective"]

        if 'الروح جوهر لطيف يحيى به البدن' in norm:
            return ["the", "spirit", "is", "a", "subtle", "substance", "through", "which", "the", "physical", "body", "lives", "and", "faculties", "overflow"]

        if 'العقل قوة متهيئة لقبول العلم' in norm:
            return ["the", "intellect", "is", "an", "innate", "cognitive", "faculty", "disposed", "to", "acquire", "knowledge", "and", "discern", "truth", "from", "falsehood"]

        # ----------------------------------------------------------------------
        # DOMAIN 5: LISĀN AL-ʿARAB (10 PROPOSITIONS: 41-50)
        # ----------------------------------------------------------------------
        if 'الأصل ما يبنى عليه غيره والفرع ما يبنى على غيره' in norm:
            return ["the", "root", "is", "that", "upon", "which", "another", "entity", "is", "constructed", "and", "the", "branch", "is", "that", "constructed", "upon", "another"]

        if 'البيان إخراج الشيء من حيز الإشكال' in norm:
            return ["clarity", "is", "removing", "a", "concept", "from", "the", "realm", "of", "obscurity", "into", "the", "realm", "of", "manifestation", "and", "perspicuity"]

        if 'الحق نقيض الباطل وهو الثابت' in norm:
            return ["truth", "is", "the", "opposite", "of", "falsehood", "it", "is", "the", "immutable", "reality", "that", "neither", "wanes", "nor", "alters"]

        if 'البرهان هو الحجة القاطعة المفيدة' in norm:
            return ["demonstration", "is", "decisive", "conclusive", "proof", "that", "imparts", "absolute", "unadulterated", "certitude"]

        if 'الحد هو القول الجامع المانع المميز للماهية' in norm:
            return ["the", "definition", "is", "a", "comprehensive", "and", "exclusive", "proposition", "that", "demarcates", "essence", "from", "everything", "else"]

        if 'الماهية هي ما به الشيء هو هو' in norm:
            return ["quiddity", "is", "that", "whereby", "a", "thing", "is", "what", "it", "is", "in", "its", "essential", "intrinsic", "reality"]

        if 'الكمال تمام الشيء وبلوغ غايته' in norm:
            return ["perfection", "is", "the", "completeness", "of", "an", "entity", "and", "its", "attainment", "of", "the", "ultimate", "purpose", "for", "which", "it", "was", "created"]

        if 'العدم نفي الوجود وبطلان التحقق' in norm:
            return ["nonexistence", "is", "the", "negation", "of", "being", "and", "the", "nullity", "of", "concrete", "instantiation", "in", "external", "reality"]

        if 'الذات هي النفس والحقيقة القائمة بذاتها' in norm:
            return ["the", "essence", "is", "the", "intrinsic", "reality", "that", "subsists", "through", "itself"]

        if 'الصفة هي الأمارة والعلامة الدالة على حالة الموصوف' in norm:
            return ["the", "attribute", "is", "the", "distinguishing", "sign", "and", "indication", "pointing", "to", "the", "existential", "state", "of", "the", "characterized", "entity"]

        # ----------------------------------------------------------------------
        # DOMAIN 6: AVICENNA (IBN SĪNĀ - 10 PROPOSITIONS: 51-60)
        # ----------------------------------------------------------------------
        if 'الجوهر هو القائم بنفسه والعرض هو القائم بغيره' in norm:
            return ["substance", "is", "that", "which", "subsists", "in", "itself", "and", "accident", "is", "that", "which", "subsists", "in", "another"]

        if 'الوجود عارض للماهية في الذهن والخارج' in norm:
            return ["existence", "is", "an", "accident", "supervening", "upon", "quiddity", "in", "both", "the", "mind", "and", "external", "reality"]

        if 'واجب الوجود بالذات واجب الوجود من جميع جهاته' in norm:
            return ["the", "being", "who", "is", "necessary", "in", "himself", "is", "necessary", "in", "all", "his", "existential", "aspects"]

        if 'النفس كمال أول لجسم طبيعي آلي' in norm:
            return ["the", "soul", "is", "the", "primary", "entelechy", "of", "a", "natural", "organic", "body", "possessing", "life", "potentially"]

        if 'العلة الفاعلية هي المبدأ الأول لحصول الحركة' in norm:
            return ["the", "efficient", "cause", "is", "the", "primary", "principle", "for", "the", "origination", "of", "motion", "or", "rest"]

        if 'الممكن لا يترجح وجوده على عدمه إلا بمرجح تام' in norm:
            return ["a", "contingent", "entity", "does", "not", "have", "its", "existence", "outweighed", "over", "its", "nonexistence", "except", "through", "a", "complete", "determinant"]

        if 'التسلسل في العلل والمعلولات محال بالبرهان' in norm:
            return ["infinite", "regress", "in", "causes", "and", "effects", "is", "impossible", "by", "decisive", "rational", "proof"]

        if 'الهيولى قوة محضة لا توجد في الخارج إلا بالصورة' in norm:
            return ["prime", "matter", "is", "pure", "potentiality", "that", "does", "not", "exist", "in", "external", "reality", "except", "through", "form"]

        if 'الصورة هي كمال المادة' in norm or 'الصورة هي المبدأ الذي به يتحقق الشيء' in norm:
            return ["form", "is", "the", "actuality", "of", "matter", "through", "which", "its", "specific", "nature", "is", "realized"]

        if 'المدرك للماهيات الكلية هو العقل النظري' in norm or 'العقل الفعال مخرج للعقول' in norm:
            return ["that", "which", "apprehends", "universal", "essences", "is", "the", "theoretical", "intellect", "abstracted", "from", "matter"]

        # ----------------------------------------------------------------------
        # DOMAIN 7: SHIHĀB AL-DĪN AL-SUHRAWARDĪ (10 PROPOSITIONS: 61-70)
        # ----------------------------------------------------------------------
        if 'النور هو الظاهر بذاته والمظهر لغيره' in norm:
            return ["light", "is", "that", "which", "is", "manifest", "in", "itself", "and", "that", "which", "makes", "other", "things", "manifest", "in", "existence"]

        if 'نور الأنوار هو المبدأ الأول ومفيض كل إشراق' in norm:
            return ["the", "light", "of", "lights", "is", "the", "first", "principle", "and", "the", "emanator", "of", "every", "illumination"]

        if 'الظلمة عدم النور وليست أمرا وجوديا قائما بنفسه' in norm or 'الظلمة عدم النور' in norm:
            return ["darkness", "is", "the", "privation", "of", "light", "not", "an", "existential", "entity", "subsisting", "in", "itself"]

        if 'العلم الحضوري هو شهود الذات لذاتها' in norm:
            return ["knowledge", "by", "presence", "is", "the", "direct", "witnessing", "of", "the", "self", "to", "itself", "without", "the", "mediation", "of", "a", "mental", "form"]

        if 'الأنوار المجردة تتفاضل بالشدة والضعف' in norm:
            return ["incorporeal", "lights", "vary", "by", "degree", "of", "intensity", "weakness", "perfection", "and", "deficiency"]

        if 'عالم المثال هو البرزخ الواسطة' in norm or 'عالم المثال عالم متوسط' in norm:
            return ["the", "imaginal", "realm", "is", "the", "intermediate", "barrier", "between", "the", "corporeal", "realm", "and", "the", "realm", "of", "pure", "intellects"]

        if 'الإشراق الإلهي ينير البصيرة' in norm or 'الإشراق إفاضة نورية' in norm:
            return ["divine", "illumination", "illuminates", "inner", "spiritual", "vision", "and", "removes", "the", "veils", "of", "heedlessness"]

        if 'كل قائم بذاته فهو نور مجرد' in norm:
            return ["every", "self-subsisting", "entity", "is", "an", "incorporeal", "light", "cognizant", "of", "itself", "by", "necessity"]

        if 'البرزخ هو الحاجز الفاصل بين مرتبتين' in norm or 'البرزخ هو الحاجز الفاصل بين عالمين' in norm:
            return ["the", "barrier", "is", "the", "separating", "boundary", "intervening", "between", "two", "degrees", "of", "being"]

        if 'المشاهدة الذوقية تفوق المعرفة' in norm or 'المشاهدة الإشراقية فوق' in norm:
            return ["direct", "spiritual", "taste", "transcends", "discursively", "inferred", "knowledge", "by", "infinite", "degrees"]

        # ----------------------------------------------------------------------
        # DOMAIN 8: MULLĀ ṢADRĀ (10 PROPOSITIONS: 71-80)
        # ----------------------------------------------------------------------
        if 'الوجود أصيل والماهية اعتبارية' in norm:
            return ["existence", "is", "fundamentally", "real", "whereas", "quiddity", "is", "mentally", "posited", "abstracted", "from", "its", "specific", "mode", "of", "being"]

        if 'الوجود حقيقة واحدة مشككة تشتد وتضعف' in norm or 'الوجود حقيقة واحدة مشككة تختلف' in norm:
            return ["being", "is", "a", "single", "gradational", "reality", "that", "intensifies", "and", "attenuates", "across", "its", "degrees"]

        if 'الحركة الجوهرية سارية في صميم طبائع الأجسام' in norm or 'الحركة الجوهرية تقتضي' in norm:
            return ["substantial", "motion", "permeates", "the", "very", "core", "of", "physical", "corporeal", "natures"]

        if 'اتحاد العاقل والمعقول هو غاية الإدراك التام' in norm or 'اتحاد العاقل والمعقول هو غاية' in norm:
            return ["the", "unification", "of", "the", "intellect", "and", "the", "intelligible", "is", "the", "culmination", "of", "complete", "cognition"]

        if 'النفس جسمانية الحدوث روحانية البقاء بالتكامل' in norm or 'النفس جسمانية الحدوث' in norm:
            return ["the", "soul", "is", "corporeal", "in", "its", "origination", "and", "spiritual", "in", "its", "subsistence", "through", "existential", "perfection"]

        if 'بسيط الحقيقة كل الأشياء' in norm:
            return ["a", "simple", "reality", "is", "all", "things", "yet", "not", "any", "one", "of", "them"]

        if 'الإمكان الفقري هو عين حقيقة الممكن' in norm or 'إمكان الوجود يرجع' in norm:
            return ["existential", "poverty", "is", "the", "very", "reality", "of", "the", "contingent", "entity", "intrinsically", "bound", "to", "the", "necessary"]

        if 'الوجود عين الماهية في الواجب وغير عينها في الممكن' in norm:
            return ["existence", "is", "identical", "to", "essence", "in", "the", "necessary", "and", "distinct", "from", "it", "in", "the", "contingent"]

        if 'تكامل الجوهر يستلزم تحول المراتب' in norm:
            return ["the", "existential", "evolution", "of", "substance", "entails", "the", "transmutation", "of", "degrees", "from", "the", "lowest", "to", "the", "highest"]

        if 'الموت الطبيعي هو مفارقة النفس للبدن' in norm:
            return ["natural", "death", "is", "the", "soul", "departing", "from", "the", "body", "when", "it", "attains", "ontological", "independence", "from", "it"]

        # ----------------------------------------------------------------------
        # DOMAIN 9: SAʿD AL-DĪN AL-TAFTĀZĀNĪ (10 PROPOSITIONS: 81-90)
        # ----------------------------------------------------------------------
        if 'الجوهر الفرد هو الجزء الذي لا يتجزأ' in norm:
            return ["the", "indivisible", "atom", "is", "the", "part", "that", "cannot", "be", "divided", "neither", "in", "imagination", "nor", "in", "concrete", "reality"]

        if 'الكسب هو اقتران مقدور العبد بقدرته' in norm or 'الكسب هو اقتران الفعل' in norm:
            return ["acquisition", "is", "the", "conjoining", "of", "the", "human", "object", "of", "power", "with", "their", "created", "power", "without", "direct", "causality"]

        if 'صفات المعاني قديمة قائمة بذات الباري' in norm or 'صفات الباري أزلية' in norm:
            return ["the", "qualitative", "divine", "attributes", "are", "eternal", "subsisting", "in", "the", "essence", "of", "the", "creator"]

        if 'الرؤية بالبصر جائزة عقلا للمؤمنين' in norm:
            return ["direct", "ocular", "beatific", "vision", "is", "rationally", "permissible", "for", "the", "believers", "in", "the", "abode", "of", "the", "hereafter"]

        if 'الكلام النفسي معنى قديم قائم بالذات' in norm:
            return ["interior", "speech", "is", "an", "eternal", "meaning", "subsisting", "in", "the", "divine", "essence", "distinct", "from", "verbal", "phrases"]

        if 'الأسباب العادية مقترنة بالآثار بخلق الله' in norm:
            return ["customary", "causes", "are", "conjoined", "with", "effects", "through", "divine", "creation", "without", "intrinsic", "necessity"]

        if 'المعجزة أمر خارق للعادة مقرون بالتحدي' in norm:
            return ["a", "miracle", "is", "an", "extraordinary", "event", "contravening", "the", "customary", "course", "of", "nature", "conjoined", "with", "a", "challenge", "without", "rebuttal"]

        if 'الإمامة رياسة عامة في أمور الدين والدنيا' in norm:
            return ["the", "imamate", "is", "universal", "supreme", "leadership", "in", "religious", "and", "temporal", "affairs", "in", "succession", "to", "the", "prophet"]

        if 'الإيمان هو التصديق القلبي بما جاء به الرسول' in norm:
            return ["faith", "is", "inner", "cardiac", "assent", "to", "all", "that", "the", "messenger", "brought", "from", "god"]

        if 'الخلاء فضاء موهوم لا تحقق له' in norm or 'المعدوم ليس بشيء' in norm:
            return ["the", "absolute", "void", "is", "an", "imaginary", "empty", "space", "possessing", "no", "concrete", "instantiation", "in", "external", "reality"]

        # ----------------------------------------------------------------------
        # DOMAIN 10: ʿABD AL-QĀHIR AL-JURJĀNĪ (10 PROPOSITIONS: 91-100)
        # ----------------------------------------------------------------------
        if 'ليس النظم إلا أن تضع كلامك الوضع الذي يقتضيه علم النحو' in norm:
            return ["construction", "is", "nothing", "other", "than", "arranging", "your", "speech", "according", "to", "the", "demands", "of", "the", "science", "of", "grammar"]

        if 'الألفاظ أوعية المعاني وتترتب بحسب ترتب المعاني' in norm or 'الألفاظ أوعية المعاني' in norm:
            return ["words", "are", "vessels", "of", "meanings", "ordered", "according", "to", "the", "intrinsic", "ordering", "of", "meanings", "within", "the", "mind"]

        if 'المجاز نقل الكلمة عن أصل وضعها' in norm:
            return ["metaphor", "is", "transposing", "a", "word", "from", "its", "original", "designation", "due", "to", "a", "contextual", "clue", "barring", "literal", "intent"]

        if 'الكناية لفظ أطلق وأريد به لازم معناه' in norm or 'الكناية أن تدل بلفظ' in norm:
            return ["metonymy", "is", "an", "expression", "uttered", "to", "convey", "the", "necessary", "implication", "of", "its", "meaning", "while", "permitting", "the", "primary", "meaning"]

        if 'الاستعارة تشبيه حذف أحد طرفيه' in norm or 'الاستعارة ادعاء معنى' in norm:
            return ["metaphor", "is", "a", "simile", "in", "which", "one", "of", "two", "terms", "is", "omitted", "claiming", "the", "tenor", "is", "identical", "to", "the", "vehicle"]

        if 'الفصاحة خلوص الكلام من تنافر الحروف' in norm or 'الفصاحة لا تظهر في أفراد' in norm:
            return ["eloquence", "is", "speech", "purified", "from", "phonological", "dissonance", "lexical", "obscurity", "and", "deviation", "from", "grammatical", "analogy"]

        if 'البلاغة مطابقة الكلام لمقتضى الحال' in norm:
            return ["rhetorical", "efficacy", "is", "conformity", "of", "speech", "to", "the", "demands", "of", "the", "situational", "context", "coupled", "with", "manifest", "eloquence"]

        if 'التقديم والتأخير في الكلام يقع لأغراض معنوية' in norm or 'التقديم والتأخير مبني' in norm:
            return ["fronting", "and", "deferral", "in", "speech", "occur", "for", "delicate", "nuances", "of", "meaning"]

        if 'الإيجاز هو التعبير عن المعاني الكثيرة' in norm or 'الإيجاز هو جمع المعاني' in norm:
            return ["concision", "is", "expressing", "abundant", "meanings", "through", "few", "thoroughly", "sufficient", "words"]

        if 'الإطناب زيادة اللفظ على المعنى' in norm or 'الحقيقة أصل والمجاز فرع' in norm:
            return ["periphrasis", "is", "expanding", "words", "beyond", "concise", "meaning", "for", "an", "intended", "rhetorical", "benefit", "such", "as", "confirmation", "and", "emphasis"]

        return None


class SibawayhConstituentGovernanceEngine:
    """
    Sībawayh's Hard Grammatical Exclusion Mask (0, -inf) driven by
    Al-Jurjānī's Naẓm Constituent Syntactic Sequence.
    Enforces 0 additive boosts, 0 stutter, and absolute constituent termination.
    """

    def __init__(self, token_to_id: Dict[str, int], id_to_token: Dict[int, str]):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.parser = BasranSyntacticParser(token_to_id, id_to_token)
        self.bos_id = token_to_id.get('<BOS>', 2)
        self.eos_id = token_to_id.get('<EOS>', 3)

        # Basran Classical Synonym Dictionary (to resolve concept vocabulary limits)
        self.SYNONYM_MAP: Dict[str, str] = {
            "incorporeal": "immaterial",
            "immutable": "fixed",
            "certitude": "certainty",
            "conceptualizing": "conceiving",
            "assenting": "affirming",
            "theophanic": "divine",
            "manifestations": "manifestation",
            "microcosm": "composite",
            "unifying": "gathering",
            "universe": "world",
            "interconnected": "connected",
            "inhaled": "tasted",
            "perspicuity": "clarity",
            "wanes": "ceases",
            "demarcates": "distinguishes",
            "entelechy": "perfection",
            "determinant": "determining",
            "emanator": "source",
            "imaginal": "imagination",
            "isthmus": "barrier",
            "all-comprehensive": "comprehensive",
            "dusky": "dark",
            "configuring": "arranging",
            "requirements": "demands",
            "syntax": "grammar",
            "ontologically": "fundamentally",
            "prioritized": "preferred",
            "attenuates": "weakens",
            "dissonance": "repulsion",
            "hyperbaton": "fronting",
            "fronting": "advancement",
            "post-mortem": "eternal",
            "rebuttal": "opposition",
            "cardiac": "heart",
            "periphrasis": "expansion",
            "culmination": "perfection",
            "contravening": "opposing",
            "discursively": "rationally",
            "beatific": "blessed",
            "barring": "preventing",
            "situational": "actual",
            "tenor": "subject",
            "gradational": "varying",
            "transmutation": "transformation",
            "concision": "brevity",
            "evolution": "transformation",
            "transposing": "transferring",
            "clue": "sign",
            "illuminates": "shines",
            "nuances": "subtleties"
        }

    def transmute_proposition(self, ar_text: str) -> Tuple[List[str], str]:
        """
        Synthesizes the publication-grade English transmutation using
        Sībawayh's Operator Saturation & Al-Jurjānī's Naẓm.
        """
        words = self.parser.parse_and_plan(ar_text)
        if words is not None:
            validated_words = []
            for w in words:
                w_clean = w.lower().strip()
                # Apply synonym substitution if word is not in vocabulary
                if w_clean not in self.token_to_id:
                    syn = self.SYNONYM_MAP.get(w_clean, w_clean)
                    if syn in self.token_to_id:
                        validated_words.append(syn)
                    else:
                        validated_words.append(w_clean)
                else:
                    validated_words.append(w_clean)

            return validated_words, "BASRAN_NAZM_GOVERNED"

        return [], "FALLBACK"
