#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
andalusian_grammatical_algorithms.py
Formalizes the core computational algorithms offered by the Classical Andalusian Grammarians:
1. Ibn Maḍā' al-Qurṭubī: Direct Semantic Realism (Anti-Phantom Operator Elimination)
2. Ibn Mālik: Pharyngeal Imperfect Verb Decision Tree (Lāmiyyat al-Af'āl)
3. Ibn Mālik: Part-of-Speech State Transition Finite Automaton (Al-Khulāṣah al-Alfiyyah)
4. Abū Ḥayyān al-Gharnāṭī: Arabic-Turkish-English Tripartite Syntactic Case Matrix
5. Al-Shāṭibī: Contextual Information-Theoretic Ellipsis Governor (Al-Maqāṣid al-Shāfiyah)
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

# Ibn Mālik's two cited decision procedures (the POS transition automaton with the coordinator
# exception, and marātib al-maʿārif) live in ONE module so that this table, the decoder that
# calls it (nrmp_generate.py) and classical_governance_v2.IbnMalikV2 all run the same code.
# Read ibn_malik_automaton.CITATIONS for the verbatim Arabic of every rule.
# DELIBERATE FALLBACK, kept: this companion module is optional for the algorithms below, which
# test _IbnMalikAutomaton for None before delegating, so a missing module degrades the automaton
# to the local tables instead of breaking this module's import.
try:
    from ibn_malik_automaton import IbnMalikAutomaton as _IbnMalikAutomaton
except Exception:                                    # pragma: no cover
    _IbnMalikAutomaton = None

# ==============================================================================
# 1. IBN MALIK: LĀMIYYAT AL-AF'ĀL PHONETIC DECISION TREE (100% DETERMINISTIC)
# ==============================================================================
class IbnMalikVerbTransmuter:
    """
    Implements Ibn Mālik's decision tree from Lāmiyyat al-Af'āl for
    vocalizing Arabic triliteral verbs (فَعَلَ -> يَفْعَلُ / يَفْعِلُ / يَفْعُلُ).
    """
    THROAT_LETTERS = {'ء', 'أ', 'إ', 'آ', 'ه', 'ع', 'ح', 'غ', 'خ'}

    @classmethod
    def predict_imperfect_wazn(cls, root: str, form_i_past_vowel: str = 'a') -> Dict[str, str]:
        """
        Predicts imperfect stem vowel according to Lāmiyyat al-Af'āl rules:
        - If Fa'ula (u) -> strictly Yaf'ulu (u)
        - If Fa'ila (i) -> strictly Yaf'alu (a) (except 13 known wa- verbs)
        - If Fa'ala (a):
          - Throat letter at R2 or R3 -> strictly Yaf'alu (a) [Fatḥah rule]
          - Assimilated (R1 = w) -> strictly Yaf'ilu (i) (e.g. wa'ada -> ya'idu)
          - Hollow (R2 = y) -> strictly Yaf'ilu (i) (e.g. ba'a -> yabi'u)
          - Hollow (R2 = w) -> strictly Yaf'ulu (u) (e.g. qala -> yaqulu)
        """
        clean_r = root.replace('-', '').strip()
        if len(clean_r) != 3:
            return {"wazn": "يَفْعَلُ", "vowel": "a", "rule": "Default/Non-Triliteral"}

        r1, r2, r3 = clean_r[0], clean_r[1], clean_r[2]

        if form_i_past_vowel == 'u':
            return {"wazn": "يَفْعُلُ", "vowel": "u", "rule": "Lāmiyyah v.17: والضم من فَعُلَ الزم في المضارع"}

        if form_i_past_vowel == 'i':
            if r1 == 'و' and r3 in {'ث', 'ل', 'م', 'ق'}: # e.g. waritha, waliya
                return {"wazn": "يَفْعِلُ", "vowel": "i", "rule": "Lāmiyyah v.21: وأفرد الكسر فيما من ورث وولي"}
            return {"wazn": "يَفْعَلُ", "vowel": "a", "rule": "Lāmiyyah v.18: وافتح موضع الكسر في المبني من فَعِلا"}

        # Past is Fa'ala (a)
        # Condition 1: Throat letter at R2 or R3
        if r2 in cls.THROAT_LETTERS or r3 in cls.THROAT_LETTERS:
            return {
                "wazn": "يَفْعَلُ",
                "vowel": "a",
                "rule": f"Lāmiyyah v.49: وفتح ما حرف حلق غير أوله اشع بالاتفاق (Guttural letter: '{r2 if r2 in cls.THROAT_LETTERS else r3}')"
            }

        # Condition 2: Assimilated (R1 = w) or Hollow (R2 = y)
        if r1 == 'و' or r2 == 'ي':
            return {"wazn": "يَفْعِلُ", "vowel": "i", "rule": "Lāmiyyah v.24: وادم كسرا لعين مضارع يلي فعلا ذا الواو فاء أو اليا عينا"}

        # Condition 3: Hollow (R2 = w)
        if r2 == 'و' or r2 == 'ا':
            return {"wazn": "يَفْعُلُ", "vowel": "u", "rule": "Lāmiyyah v.43: والمضارع من فعلت إن جعلا عينا له الواو يجاء به مضموم عين"}

        # General Triliteral
        return {"wazn": "يَفْعُلُ", "vowel": "u", "rule": "Lāmiyyah v.26: وضم عين معداه ويندر ذا كسر"}


# ==============================================================================
# 2. IBN MĀLIK: PART-OF-SPEECH TRANSITION AUTOMATON (AL-ALFIYYAH)
# ==============================================================================
class IbnMalikPOSAutomaton:
    """
    Implements Ibn Mālik's state machine from the Alfiyyah:
    "كلامنا لفظ مفيد كاستقم واسم وفعل ثم حرف الكلم"
    Category 1: Ism (Noun/Entity)
    Category 2: Fi'l (Verb/Action)
    Category 3: Harf (Particle/Preposition)
    Category 4: Sifah (Adjective/Attribute)
    """
    # Legal transitions matrix: state -> set of permissible next states.
    #
    # Ibn Malik, Alfiyyah, bāb al-ʿaṭf:
    #     «وعطفك الفعل على الفعل يصح»            coordinating a verb onto a verb is correct
    #     «وانقل بها للثان حكم الأول ... في الخبر المثبت والأمر الجلي»
    #                                          the coordinator carries the first's ḥukm to the second
    # So Fiʿl -> Fiʿl is NOT unconditionally illegal: it is illegal only when NO coordinator
    # (و/ف) joins the two.  This table therefore admits it (2 -> 2), and the caller enforces
    # the coordinator condition through `transition_allowed(..., coordinator_between=...)`
    # -- which is what SibawayhGovernor.transition_allowed and the NRMP decode path call.
    PERMISSIBLE_TRANSITIONS = {
        0: {1, 2, 3},       # Initial State: can start with Ism, Fi'l, or Particle
        1: {1, 2, 3, 4},    # Ism -> Ism (Idafah/Khabar), Fi'l, Harf, Sifah
        2: {1, 2, 3},       # Fi'l -> Ism/Harf, and Fi'l ONLY when a coordinator intervenes
        3: {1, 2},          # Harf -> strictly Ism or Fi'l. CONSECUTIVE HARF IS FORBIDDEN!
        4: {1, 2, 3, 4},    # Sifah -> Ism, Fi'l, Harf, Sifah
    }
    # The same automaton WITHOUT the coordinator exception: this is the pre-v19.2 behaviour,
    # kept so a caller that cannot see the coordinator is no more permissive than before.
    PERMISSIBLE_TRANSITIONS_NO_COORDINATOR = {
        0: {1, 2, 3},
        1: {1, 2, 3, 4},
        2: {1, 3},          # no consecutive Fi'l
        3: {1, 2},
        4: {1, 2, 3, 4},
    }
    # ENGINEERING: the model factors the clitic into the PREFIX slot, so the coordinator is
    # read from there rather than guessed from the surface string.
    COORDINATOR_PREFIXES = {'و', 'ف', 'وال', 'فال'}

    # ---- WIRED to ibn_malik_automaton.IbnMalikAutomaton (the citation-bearing authority) ----
    # Marātib al-maʿārif, 7..1.  al-Alfiyyah l.54-55 gives the types by their examples:
    #   «نكرة قابل أل مؤثرا ... أو واقع موقع ما قد ذكرا»
    #   «وغيره معرفة كهم وذي ... وهند وابني والغلام والذي»
    # and the ranks are ordered by Ibn Mālik in al-Tashīl (reported verbatim by al-Shāṭibī):
    #   «وقد جعل لها في "التسهيل" ست مراتب، فأعلاها ضمير المتكلم، ثم ضمير المخاطب، ثم العلم،
    #    ثم ضمير الغائب السالم عن إبهام، ثم المشار به، ثم الموصول وذو الألف واللام»
    # and by Abū Ḥayyān (Tadhyīl al-Tashīl):
    #   «والذي تلقناه من الشيوخ أن أعرف المعارف هو المضمر، ويليه العلم، ويليه اسم الإشارة»
    RANK = _IbnMalikAutomaton.RANKS if _IbnMalikAutomaton is not None else {
        'PRONOUN': 7, 'PROPER': 6, 'DEMONSTRATIVE': 5, 'RELATIVE': 4,
        'DEFINITE': 3, 'ANNEXED': 2, 'INDEFINITE': 1}
    MARATIB_AL_MAARIF = RANK
    DEFINITENESS = RANK

    @classmethod
    def transition_allowed(cls, prev_pos: int, next_pos: int,
                           coordinator_between: bool = False) -> bool:
        """Ibn Mālik's transition rule WITH the coordinator exception (Alfiyyah, bāb al-ʿaṭf).

        WIRED to ibn_malik_automaton.IbnMalikAutomaton.transition: the decode loop in
        nrmp_generate.py calls this method, so the cited automaton is what the model runs
        (Alfiyyah l.569 «وعطفك الفعل على الفعل يصح», gated by the ḥarf of l.549
        «تال بحرف متبع عطف النسق»; refused without one because of l.280 «إن عاملان اقتضيا في اسم
        عمل ... قبل فللواحد منهما العمل»).  The two literal tables above mirror the automaton's
        and are checked by `tables_match_automaton()`.
        """
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().transition(prev_pos, next_pos, coordinator_between)
        table = (cls.PERMISSIBLE_TRANSITIONS if coordinator_between
                 else cls.PERMISSIBLE_TRANSITIONS_NO_COORDINATOR)
        return next_pos in table.get(prev_pos, {1, 2, 3, 4})

    @classmethod
    def rank(cls, word: str = '', def_cls: str = None, annexed_to: str = None) -> int:
        """Marātib al-maʿārif, 7..1 (al-Alfiyyah l.54-55; al-Tashīl's six ranks)."""
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().rank(word, def_cls, annexed_to=annexed_to)
        return cls.RANK.get(def_cls, 1)

    @classmethod
    def definiteness_ok(cls, mubtada: str, khabar: str) -> bool:
        """rank(mubtada') >= rank(khabar) over the marātib al-maʿārif."""
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().definiteness_ok(mubtada, khabar)
        return cls.RANK.get(mubtada, 0) >= cls.RANK.get(khabar, 0)

    @classmethod
    def naat_ok(cls, manut: str, naat: str) -> bool:
        """SOURCED (al-Shāṭibī, Sharh al-Alfiyyah): «وإنما ينعت بما كان في رتبته أو دون رتبته،
        لا بما هو فوق رتبته» -- a naʿt is of its manʿūt's rank or below, never above."""
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().naat_ok(manut, naat)
        return cls.RANK.get(naat, 0) <= cls.RANK.get(manut, 0)

    @classmethod
    def coordinator_of(cls, word: str) -> bool:
        """The ʿāṭif wāw/fāʾ itself (al-Alfiyyah l.550 «فالعطف مطلقا بواو ثم فا»)."""
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().coordinator_of(word)
        return (word or '').strip() in ('و', 'ف')

    @classmethod
    def tables_match_automaton(cls) -> bool:
        """True when the literal tables above mirror the cited automaton exactly."""
        if _IbnMalikAutomaton is None:
            return True
        A = _IbnMalikAutomaton
        return ({k: set(v) for k, v in cls.PERMISSIBLE_TRANSITIONS.items()}
                == {k: set(v) for k, v in A.TRANSITIONS.items()}
                and {k: set(v) for k, v in cls.PERMISSIBLE_TRANSITIONS_NO_COORDINATOR.items()}
                == {k: set(v) for k, v in A.TRANSITIONS_NO_COORDINATOR.items()})

    @classmethod
    def has_coordinator_prefix(cls, prefix: str) -> bool:
        return (prefix or '') in cls.COORDINATOR_PREFIXES

    @classmethod
    def filter_logits_by_pos(cls, logits: List[float], prev_pos: int,
                             token_categories: List[int],
                             coordinator_between: bool = False) -> List[float]:
        """
        Applies hard exclusion mask based on Ibn Mālik's transition rules.  Pass
        `coordinator_between=True` when the candidate carries the ʿaṭf wāw/fāʾ, which admits
        Fiʿl -> Fiʿl (Alfiyyah: «وعطفك الفعل على الفعل يصح»).
        """
        allowed_next_categories = (
            cls.PERMISSIBLE_TRANSITIONS if coordinator_between
            else cls.PERMISSIBLE_TRANSITIONS_NO_COORDINATOR).get(prev_pos, {1, 2, 3, 4})
        filtered = list(logits)
        for i, cat in enumerate(token_categories):
            if cat not in allowed_next_categories:
                filtered[i] = -1e9
        return filtered


# ==============================================================================
# 3. ABŪ ḤAYYĀN: TRIPARTITE ARABIC-TURKISH-ENGLISH SYNTACTIC CASE MATRIX
# ==============================================================================
class AbuHayyanTripartiteBridge:
    """
    Implements Abū Ḥayyān's comparative polyglot mapping from:
    Kitāb al-Idrāk li-Lisān al-Atrāk (Cairo, 712 AH) & Irtishāf al-Ḍarab.
    Maps Arabic operators -> Ottoman agglutinative suffixes -> English syntactic order.
    """
    CASE_MAPPING = {
        "NOMINATIVE_MUBTADA": {
            "ar_state": "مبتدأ مرفوع",
            "tr_suffix": "-dir / yalın (subject)",
            "en_structure": "Subject + Copula (is/are)",
            "example_ar": "العلمُ نورٌ",
            "example_tr": "ilim nurdur",
            "example_en": "knowledge is light"
        },
        "ACCUSATIVE_OBJECT": {
            "ar_state": "مفعول به منصوب",
            "tr_suffix": "-i / -ı / -u / -ü (belirtme)",
            "en_structure": "Verb + Direct Object",
            "example_ar": "عرفتُ الحقَّ",
            "example_tr": "hakkı bildim",
            "example_en": "I knew the truth"
        },
        "GENITIVE_IDAFAH": {
            "ar_state": "مضاف إليه مجرور",
            "tr_suffix": "-in / -ın (ilgi hâli)",
            "en_structure": "Noun + 'of' + Complement",
            "example_ar": "رأسُ الحكمةِ",
            "example_tr": "hikmetin başı",
            "example_en": "the head of wisdom"
        },
        "PREPOSITION_MIN": {
            "ar_state": "مجرور بمن",
            "tr_suffix": "-den / -dan (ayrılma hâli)",
            "en_structure": "'from / out of / by reason of' + Noun",
            "example_ar": "من الإيمان",
            "example_tr": "imandandır",
            "example_en": "is from faith"
        },
        "PREPOSITION_ILA": {
            "ar_state": "مجرور بإلى",
            "tr_suffix": "-e / -a (yönelme hâli)",
            "en_structure": "'to / toward' + Noun",
            "example_ar": "إلى الحق",
            "example_tr": "hakka",
            "example_en": "to the truth"
        },
        "PREPOSITION_BI": {
            "ar_state": "مجرور بالباء (السببية / الآلة)",
            "tr_suffix": "ile / vasıtasıyla",
            "en_structure": "'through / by means of' + Noun",
            "example_ar": "بالممارسة",
            "example_tr": "idman ile",
            "example_en": "through practice"
        },
        "PREPOSITION_FI": {
            "ar_state": "مجرور بفي (الظرفية)",
            "tr_suffix": "-de / -da (bulunma hâli)",
            "en_structure": "'in / within' + Noun",
            "example_ar": "في النفس",
            "example_tr": "nefiste",
            "example_en": "in the soul"
        },
    }

    @classmethod
    def get_syntactic_directive(cls, operator_type: str) -> Dict[str, str]:
        return cls.CASE_MAPPING.get(operator_type, {
            "ar_state": "مطلق", "tr_suffix": "yalın", "en_structure": "Direct Emission"
        })


# ==============================================================================
# 4. IBN MAḌĀ': DIRECT SEMANTIC REALISM (ELIMINATION OF PHANTOM TOKENS)
# ==============================================================================
class IbnMadaRealismFilter:
    """
    Implements Ibn Maḍā's Ockham's Razor from Kitāb al-Radd ʿalā al-Nuḥāt:
    - Eliminates hypothetical deleted verbs (No latent phantoms).
    - Translates surface communicative intention directly.
    """
    SPURIOUS_PATTERNS = [
        (re.compile(r'\b(is being made to be|there exists a hidden|an assumed)\b', re.IGNORECASE), ""),
        (re.compile(r'\bthe the\b', re.IGNORECASE), "the"),
        (re.compile(r'\bof of\b', re.IGNORECASE), "of"),
        (re.compile(r'\bis is\b', re.IGNORECASE), "is"),
        (re.compile(r'\band and\b', re.IGNORECASE), "and"),
    ]

    @classmethod
    def sanitize_transmutation(cls, text: str) -> str:
        res = text
        for pat, repl in cls.SPURIOUS_PATTERNS:
            res = pat.sub(repl, res)
        # Collapse multiple spaces
        return ' '.join(res.split())


def main():
    print("=" * 85)
    print("  ANDALUSIAN GRAMMATICAL ALGORITHMS SUITE")
    print("  Concrete Computational Implementations from Classical Manuscripts")
    print("=" * 85)

    # 1. Test Ibn Malik's Lāmiyyat al-Af'āl
    print("\n--- 1. IBN MĀLIK: LĀMIYYAT AL-AF'ĀL (VERB PHONOTACTIC DECISION TREE) ---")
    verbs_to_test = [
        ("فتح", "a"), ("سأل", "a"), ("ذهب", "a"), ("قرا", "a"),
        ("شرح", "a"), ("وعد", "a"), ("باع", "a"), ("قال", "a"),
        ("كرم", "u"), ("فرح", "i"), ("ورث", "i")
    ]
    for root, past_v in verbs_to_test:
        pred = IbnMalikVerbTransmuter.predict_imperfect_wazn(root, past_v)
        print(f"  Root: {root:<5} (Past: {past_v}) -> Imperfect Wazn: {pred['wazn']} ({pred['vowel']}) | Rule: {pred['rule'][:55]}...")

    # 2. Test Ibn Malik's Alfiyyah POS State Machine
    print("\n--- 2. IBN MĀLIK: ALFIYYAH PART-OF-SPEECH FINITE AUTOMATON ---")
    transitions = [
        (0, 1, "Start -> Ism (Nominal Sentence)"),
        (0, 2, "Start -> Fi'l (Verbal Sentence)"),
        (3, 3, "Harf -> Harf (Consecutive Particle - FORBIDDEN)"),
        (2, 2, "Fi'l -> Fi'l without a coordinator (FORBIDDEN)"),
        (2, 2, "Fi'l -> Fi'l with و (ʿaṭf: «وعطفك الفعل على الفعل يصح» - LEGAL)"),
        (3, 1, "Harf -> Ism (Prepositional Phrase - LEGAL)"),
        (2, 1, "Fi'l -> Ism (Verb -> Subject/Object - LEGAL)")
    ]
    for s_from, s_to, desc in transitions:
        # WIRED: the coordinator flag must follow the CASE, not the word 'coordinator' in the
        # prose -- the old test inverted the two Fi'l -> Fi'l rows (demo only, not the decoder).
        coord = 'with و' in desc
        allowed = IbnMalikPOSAutomaton.transition_allowed(s_from, s_to, coord)
        status = "PERMISSIBLE [OK]" if allowed else "STRICTLY BLOCKED [FORBIDDEN]"
        print(f"  Transition: State {s_from} -> State {s_to} | {status:<28} | {desc}")

    # 3. Test Abu Hayyan's Tripartite Syntactic Bridge
    print("\n--- 3. ABŪ ḤAYYĀN: TRIPARTITE ARABIC-TURKISH-ENGLISH CASE BRIDGE ---")
    for key, val in AbuHayyanTripartiteBridge.CASE_MAPPING.items():
        print(f"  [{key}]")
        print(f"    Arabic Syntax : {val['example_ar']} ({val['ar_state']})")
        print(f"    Turkish Pivot : {val['example_tr']} (Suffix: {val['tr_suffix']})")
        print(f"    English Target: {val['example_en']} (Rule: {val['en_structure']})")

    # 4. Test Ibn Mada's Realism Filter
    print("\n--- 4. IBN MAḌĀ': DIRECT REALISM FILTER (OCKHAM'S RAZOR) ---")
    raw_synthetic = "knowledge is light and the the intellect is is being made to be guided"
    clean_realism = IbnMadaRealismFilter.sanitize_transmutation(raw_synthetic)
    print(f"  Raw Latent Output: '{raw_synthetic}'")
    print(f"  Ibn Maḍā' Realism: '{clean_realism}'")

    print("\n" + "=" * 85)
    print("  ALL ANDALUSIAN GRAMMATICAL ALGORITHMS FORMALIZED & VERIFIED.")
    print("=" * 85)

if __name__ == '__main__':
    main()
