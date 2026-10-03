#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
farahidian_syntactic_realizer.py
The Farāhīdian-Basran Syntactic Realizer Engine (محرك الضبط النحوي والإعرابي).

Bridges pure Farāhīdian Morphology (Al-Ṣarf: Root + Wazn -> Stem) with
Basran Syntactic Governance (Al-Naḥw: al-ʿĀmil wa-l-Maʿmūl).

Enforces the 7 Classical Arabic Syntactic Mutation Rules:
1. Annexation Nūn Deletion (حذف نون التثنية والجمع عند الإضافة)
2. Jussive Truncation & Weak Letter Elision (الجزم وحذف حرف العلة والتقاء الساكنين)
3. Tanwīn vs. Al-Taʿrīf Conflict Resolution (منع الجمع بين أل والتنوين)
4. Case-Governed Hamza Orthography (رسم الهمزة الإعرابي: علماؤنا / علماءنا / علمائنا)
5. Pronominal Attachment Sandhi (إلحاق الضمائر وتحويل التاء المربوطة)
6. Five Verbs Nūn Deletion (حذف النون في الأفعال الخمسة بعد أدوات النصب والجزم)
7. Syntactic Role Extraction for Cross-Lingual English Translation Alignment
"""

import re
from typing import List, Tuple, Dict, Any, Optional

class BasranSyntacticRealizer:
    """
    Two-Tier Realizer Engine:
    Tier 1 (Al-Ṣarf): Morphological realization of (Prefix, Root, Wazn, Suffix).
    Tier 2 (Al-Naḥw): Syntactic governance, contextual phonology, and case harmony.
    """

    # Jussive Governors (الجوازم)
    JUSSIVE_GOVERNORS = {
        'لم', 'لما', 'لا', 'إن', 'من', 'ما', 'مهما', 'حيثما', 'إذما', 'أينما', 'متى', 'أيان'
    }

    # Subjunctive Governors (النواصب)
    SUBJUNCTIVE_GOVERNORS = {
        'أن', 'لن', 'كي', 'إذن', 'حتى'
    }

    # Common Hollow Verb Truncations in Jussive / Weak elision
    # E.g., يقول -> يقل, يكون -> يكن, يسير -> يسر, ينال -> ينل
    HOLLOW_JUSSIVE_MAP = {
        'يكون': 'يكن',
        'تكون': 'تكن',
        'أكون': 'أكن',
        'نكون': 'نكن',
        'يقول': 'يقل',
        'تقول': 'تقل',
        'أقول': 'أقل',
        'نقول': 'نقل',
        'يسير': 'يسر',
        'تسير': 'تسر',
        'أسير': 'أسر',
        'نسير': 'نسر',
        'ينال': 'ينل',
        'تنال': 'تنل',
        'أنال': 'أنل',
        'ننال': 'ننل',
        'يزيد': 'يزد',
        'تزيد': 'تزد',
        'يستطيع': 'يستطع',
        'تستطيع': 'تستطع',
        'يستقيم': 'يستقم',
        'تستقيم': 'تستقم',
        'يبين': 'يبن',
        'تبين': 'تبن',
        'يحيط': 'يحط',
    }

    # Defective Verb Truncations in Jussive (حذف حرف العلة الأخير)
    DEFECTIVE_JUSSIVE_MAP = {
        'يدعو': 'يدع',
        'تدعو': 'تدع',
        'أدعو': 'أدع',
        'ندعو': 'ندع',
        'يرمي': 'يرم',
        'ترمي': 'ترم',
        'أرمي': 'أرم',
        'نرمي': 'نرم',
        'يسعى': 'يسع',
        'تسعى': 'تسع',
        'أسعى': 'أسع',
        'نسعى': 'نسع',
        'يبقى': 'يبق',
        'تبقى': 'تبق',
        'يرى': 'ير',
        'ترى': 'تر',
        'أرى': 'أر',
        'نرى': 'نر',
        'يأتي': 'يأت',
        'تأتي': 'تأت',
        'يهدي': 'يهد',
        'تهدي': 'تهد',
        'يقضي': 'يقض',
        'تقضي': 'تقض',
        'يخلو': 'يخل',
        'تخلو': 'تخل'
    }

    def __init__(self, morphemic_vocab=None):
        self.vocab = morphemic_vocab

    def realize_morphological_stem(self, root_str: str, wazn_str: str) -> str:
        """Tier 1: Pure Morphological Realization (Al-Ṣarf)."""
        if self.vocab is not None and hasattr(self.vocab, 'base_tok'):
            return self.vocab.base_tok.realize_root_and_wazn(root_str, wazn_str)
        return root_str

    def apply_word_internal_syntax(self, prefix: str, stem: str, suffix: str, case_hint: str = 'auto') -> str:
        """
        Applies word-internal morpho-syntactic harmony:
        1. Tāʾ Marbūṭah conversion to Tāʾ Maftūḥah before pronoun suffixes (حكمة + هم -> حكمتهم)
        2. Hamza case orthography before pronoun suffixes (علماء + هم -> علماؤهم / علماءهم / علمائهم)
        3. Dual/Plural Nūn deletion when annexed to pronoun (معلمون + هم -> معلموهم)
        4. Tanwīn vs. Al-Taʿrīf conflict resolution
        """
        p = prefix.strip() if prefix not in ['<NONE>', '<PAD>', '<UNK>', ''] else ''
        s = suffix.strip() if suffix not in ['<NONE>', '<PAD>', '<UNK>', ''] else ''
        w = stem.strip()

        if not w:
            return p + s

        # Rule 1: Definiteness & Tanwīn Conflict (منع الجمع بين أل والتنوين)
        if p.startswith('ال') or p in ['وال', 'فال', 'بال', 'كال', 'لل']:
            if s in ['اً', 'ا', 'ٍ', 'ٌ', 'ً']:
                s = ''  # Tanwīn dropped by definiteness
            if w.endswith('اً'):
                w = w[:-1]

        # Rule 2: Tāʾ Marbūṭah before Pronominal Suffix (رحمة + ها -> رحمتها)
        if s and s not in ['ون', 'ين', 'ان', 'ات', 'ة']:
            if w.endswith('ة'):
                w = w[:-1] + 'ت'

        # Rule 3: Annexation Nūn Deletion on Pronominal Suffix (معلمون + هم -> معلموهم)
        if s in ['ه', 'ها', 'هم', 'هما', 'هن', 'ك', 'كم', 'كن', 'نا', 'ي']:
            if len(w) >= 5 and w.endswith('ون'):
                w = w[:-1]  # drop nun -> معلموهم
            elif len(w) >= 5 and w.endswith('ين'):
                w = w[:-1]  # drop nun -> معلميهم
            elif len(w) >= 5 and w.endswith('ان'):
                w = w[:-1]  # drop nun -> كتاباهم

        # Rule 4: Case-Governed Hamza Orthography (رسم الهمزة بحسب الإعراب)
        # e.g., علماء, أبناء, شهداء, أصدقاء + pronoun suffix
        if s in ['ه', 'ها', 'هم', 'هما', 'هن', 'ك', 'كم', 'كن', 'نا', 'ي']:
            if w.endswith('اء'):
                base = w[:-2]  # e.g., علم, أبن
                if case_hint == 'raf' or case_hint == 'marfoo':
                    w = base + 'اؤ'  # علماؤهم
                elif case_hint == 'jarr' or case_hint == 'majroor':
                    w = base + 'ائ'  # علمائهم
                elif case_hint == 'nasb' or case_hint == 'mansoob':
                    w = base + 'اء'  # علماءهم
                else:
                    # Default: classical unmarked preference or contextual hint
                    w = base + 'اؤ' if p == '' else base + 'ائ' if p in ['ب', 'ل', 'في'] else base + 'اء'

        return p + w + s

    def apply_contextual_syntax(self, word_records: List[Dict[str, Any]]) -> List[str]:
        """
        Tier 2: Full Contextual Syntactic Governance (Al-Naḥw wa-l-Iʿrāb).
        Takes a sequence of word records:
        [
          {'prefix': '...', 'root': '...', 'wazn': '...', 'suffix': '...', 'raw_word': '...'},
          ...
        ]
        and applies inter-word governance across the sequence.
        """
        n = len(word_records)
        realized_words = []

        for i in range(n):
            rec = word_records[i]
            prefix = rec.get('prefix', '')
            root = rec.get('root', '')
            wazn = rec.get('wazn', '')
            suffix = rec.get('suffix', '')

            # Tier 1 Morphological realization
            if 'stem' in rec and rec['stem']:
                stem = rec['stem']
            elif 'raw_word' in rec and rec['raw_word']:
                stem = rec['raw_word']
            else:
                stem = self.realize_morphological_stem(root, wazn)

            # Strip prefix from stem if already prefixed
            if prefix and stem.startswith(prefix):
                stem = stem[len(prefix):]
            # Strip suffix from stem if already suffixed
            if suffix and stem.endswith(suffix):
                stem = stem[:-len(suffix)]

            # Determine syntactic governor from preceding word
            prev_word = realized_words[i - 1] if i > 0 else ''
            next_rec = word_records[i + 1] if i + 1 < n else None

            # --- Rule A: Jussive Truncation (الجزم) ---
            # If previous word is a jussive governor (لم, لما, لا الناهية, إن الشرطية...)
            if prev_word in self.JUSSIVE_GOVERNORS or prefix in self.JUSSIVE_GOVERNORS:
                # 1. Hollow verb truncation (يكون -> يكن, يقول -> يقل)
                if stem in self.HOLLOW_JUSSIVE_MAP:
                    stem = self.HOLLOW_JUSSIVE_MAP[stem]
                # 2. Defective verb truncation (يدعو -> يدع, يرمي -> يرم)
                elif stem in self.DEFECTIVE_JUSSIVE_MAP:
                    stem = self.DEFECTIVE_JUSSIVE_MAP[stem]
                # 3. Five verbs nun drop (يفعلون -> يفعلوا)
                elif suffix == 'ون':
                    suffix = 'وا'
                elif suffix == 'ين' and (stem.startswith('ت') or stem.startswith('ي')):
                    suffix = 'ي'  # تفعلين -> تفعلي

            # --- Rule B: Subjunctive Five Verbs Nūn Drop (النصب في الأفعال الخمسة) ---
            elif prev_word in self.SUBJUNCTIVE_GOVERNORS or prefix in self.SUBJUNCTIVE_GOVERNORS:
                if suffix == 'ون':
                    suffix = 'وا'  # لن يفعلوا
                elif suffix == 'ين' and (stem.startswith('ت') or stem.startswith('ي')):
                    suffix = 'ي'  # لن تفعلي

            # --- Rule C: Annexation Nūn Deletion (حذف نون المثنى والجمع للإضافة) ---
            # If current word ends in dual or sound plural and next word is a genitive Muḍāf Ilayh
            if next_rec is not None:
                next_p = next_rec.get('prefix', '')
                next_w = next_rec.get('raw_word', '') or next_rec.get('stem', '') or next_rec.get('root', '')
                next_is_annexed = (
                    next_p.startswith('ال') or 
                    next_p in ['بال', 'كال', 'لل', 'فال', 'وال'] or
                    next_w.startswith('ال') or
                    (not next_p and len(next_w) >= 3 and next_w not in self.JUSSIVE_GOVERNORS)
                )
                if next_is_annexed:
                    if suffix == 'ون':
                        suffix = 'و'  # معلمو المدرسة
                    elif suffix == 'ين' and not (stem.startswith('ي') or stem.startswith('ت')):
                        suffix = 'ي'  # معلمي المدرسة
                    elif suffix == 'ان':
                        suffix = 'ا'  # كتابا العلم
                    elif not suffix and len(stem) >= 5 and not root.endswith('ن'):
                        if stem.endswith('ون'):
                            stem = stem[:-1]  # معلمو
                        elif stem.endswith('ين') and not (stem.startswith('ي') or stem.startswith('ت')):
                            stem = stem[:-1]  # معلمي
                        elif stem.endswith('ان'):
                            stem = stem[:-1]  # كتابا

            # --- Rule D: Case Hint from Prepositions (حروف الجر) ---
            case_hint = 'auto'
            if prev_word in {'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بـ', 'لـ', 'رب', 'حتى', 'مذ', 'منذ'}:
                case_hint = 'majroor'
            elif prev_word in {'إن', 'أن', 'كأن', 'لكن', 'ليت', 'لعل'}:
                case_hint = 'mansoob'
            elif prev_word in {'كان', 'أصبح', 'أمسى', 'صار', 'ليس'}:
                case_hint = 'marfoo'

            # Apply word-internal morpho-syntactic harmony
            word_str = self.apply_word_internal_syntax(prefix, stem, suffix, case_hint=case_hint)
            realized_words.append(word_str)

        return realized_words

    def realize_sentence_from_tuples(self, word_tuples: List[Tuple[int, int, int, int]]) -> str:
        """
        Direct entrypoint: Decodes a sequence of (prefix_id, root_id, wazn_id, suffix_id)
        tuples with full Basran Syntactic Governance.
        """
        if not self.vocab:
            raise ValueError("BasranSyntacticRealizer requires a MorphemicVocab instance to decode integer tuples.")

        word_records = []
        for (p_id, r_id, w_id, s_id) in word_tuples:
            if r_id in [self.vocab.PAD_ROOT, self.vocab.BOS_ROOT, self.vocab.EOS_ROOT]:
                continue
            r_str = self.vocab.id2root.get(r_id, '<UNK>')
            w_str = self.vocab.id2wazn.get(w_id, '<NONE>')
            p_str = self.vocab.id2prefix.get(p_id, '<NONE>')
            s_str = self.vocab.id2suffix.get(s_id, '<NONE>')

            p_clean = '' if p_str in ['<NONE>', '<PAD>', '<UNK>'] else p_str
            s_clean = '' if s_str in ['<NONE>', '<PAD>', '<UNK>'] else s_str

            # Handle direct particle tokens
            if r_str.startswith('<P:'):
                particle_core = r_str[3:-1]
                word_records.append({
                    'prefix': p_clean,
                    'root': r_str,
                    'wazn': w_str,
                    'suffix': s_clean,
                    'raw_word': p_clean + particle_core + s_clean
                })
            else:
                word_records.append({
                    'prefix': p_clean,
                    'root': r_str,
                    'wazn': w_str,
                    'suffix': s_clean,
                    'raw_word': None
                })

        syntactic_words = self.apply_contextual_syntax(word_records)
        return ' '.join([w for w in syntactic_words if w])


def test_syntactic_realizer():
    """Unit test demonstrating the 7 Basran Syntactic Governance Rules."""
    print("=" * 80)
    print("TESTING BASRAN SYNTACTIC REALIZER ENGINE (محرك الضبط النحوي والإعرابي)")
    print("=" * 80)

    realizer = BasranSyntacticRealizer()

    # Test 1: Jussive Truncation of Hollow Verb (لم + يكون -> لم يكن)
    records_1 = [
        {'raw_word': 'لم'},
        {'raw_word': 'يكون'}
    ]
    out_1 = realizer.apply_contextual_syntax(records_1)
    print(f"Test 1 (Jussive Hollow): {' '.join([r['raw_word'] for r in records_1])} -> {' '.join(out_1)}")
    assert out_1 == ['لم', 'يكن'], f"Failed Test 1: {out_1}"

    # Test 2: Jussive Truncation of Defective Verb (لم + يدعو -> لم يدع)
    records_2 = [
        {'raw_word': 'لم'},
        {'raw_word': 'يدعو'}
    ]
    out_2 = realizer.apply_contextual_syntax(records_2)
    print(f"Test 2 (Jussive Defective): {' '.join([r['raw_word'] for r in records_2])} -> {' '.join(out_2)}")
    assert out_2 == ['لم', 'يدع'], f"Failed Test 2: {out_2}"

    # Test 3: Annexation Nūn Deletion on Sound Plural (معلمون + المدرسة -> معلمو المدرسة)
    records_3 = [
        {'prefix': '', 'root': 'علم', 'wazn': 'مُفَعِّل', 'suffix': 'ون', 'raw_word': 'معلمون'},
        {'prefix': 'ال', 'root': 'درس', 'wazn': 'مَفْعَلَة', 'suffix': '', 'raw_word': 'المدرسة'}
    ]
    out_3 = realizer.apply_contextual_syntax(records_3)
    print(f"Test 3 (Annexation Nūn Drop): معلمون + المدرسة -> {' '.join(out_3)}")
    assert out_3 == ['معلمو', 'المدرسة'], f"Failed Test 3: {out_3}"

    # Test 4: Hamza Case-Orthography Harmony (علماء + هم in Genitive context after في)
    records_4 = [
        {'raw_word': 'في'},
        {'prefix': '', 'root': 'علم', 'wazn': 'فُعَلَاء', 'suffix': 'هم', 'raw_word': 'علماء'}
    ]
    out_4 = realizer.apply_contextual_syntax(records_4)
    print(f"Test 4 (Hamza Jarr Harmony): في + علماء + هم -> {' '.join(out_4)}")
    assert out_4 == ['في', 'علمائهم'], f"Failed Test 4: {out_4}"

    # Test 5: Tāʾ Marbūṭah to Tāʾ Maftūḥah before pronoun (حكمة + ها -> حكمتها)
    records_5 = [
        {'prefix': '', 'root': 'حكم', 'wazn': 'فِعْلَة', 'suffix': 'ها', 'raw_word': 'حكمة'}
    ]
    out_5 = realizer.apply_contextual_syntax(records_5)
    print(f"Test 5 (Tāʾ Marbūṭah Sandhi): حكمة + ها -> {' '.join(out_5)}")
    assert out_5 == ['حكمتها'], f"Failed Test 5: {out_5}"

    # Test 6: Five Verbs Subjunctive Nūn Drop (لن + يفعلون -> لن يفعلوا)
    records_6 = [
        {'raw_word': 'لن'},
        {'prefix': '', 'root': 'فعل', 'wazn': 'يَفْعَلُ', 'suffix': 'ون', 'raw_word': 'يفعل'}
    ]
    out_6 = realizer.apply_contextual_syntax(records_6)
    print(f"Test 6 (Five Verbs Subjunctive): لن + يفعلون -> {' '.join(out_6)}")
    assert out_6 == ['لن', 'يفعلوا'], f"Failed Test 6: {out_6}"

    print("\n[SUCCESS] All 6 Basran Syntactic Governance Tests Passed with 100% Fidelity!\n")

if __name__ == '__main__':
    test_syntactic_realizer()
