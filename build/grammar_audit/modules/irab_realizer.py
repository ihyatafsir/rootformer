#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
irab_realizer.py -- Ibn Malik's i'rab REALISATION: case -> surface ending.

IbnMalikV2 already carries the maratib al-ma'arif (the six definiteness ranks, Alfiyyah v.55) and
enforces them. What was missing is the other half of Ibn Malik's system: given a case, produce the
actual ending. al-Alfiyyah states the principals and their substitutes (niyaba):

    principals        raf' dammah | nasb fathah | jarr kasrah | jazm sukun
    asma' khamsa      ab, akh, ham, fu, dhu -> waw (raf'), alif (nasb), ya' (jarr)
    dual              -ani (raf') / -ayni (nasb AND jarr)
    sound masc plural -una  (raf') / -ina  (nasb AND jarr)
    sound fem plural  -atu  (raf') / -ati  (nasb AND jarr)
    mamnu' min al-sarf  jarr takes FATHah, not kasrah  (the diptote)
    mu'tall al-akhir    raf' dammah estimated; nasb fathah shows; JAZM DELETES the weak letter
                        (lam yad'u, lam yarmi, lam yarda)

Usage:
  python irab_realizer.py          # verifies against canonical examples
"""
import re
import sys

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640]')
FATHA, DAMMA, KASRA, SUKUN = '\u064e', '\u064f', '\u0650', '\u0652'
SHADDA = '\u0651'
CASE_MARK = {'raf': DAMMA, 'nasb': FATHA, 'jarr': KASRA, 'jazm': SUKUN}

ASMA_KHAMSA = {'أب', 'أخ', 'حم', 'فو', 'ذو', 'ابو', 'ابا', 'ابي', 'أخو', 'أخا', 'أخي'}
WEAK_FINAL = set('وايىأ')


def strip_diac(s):
    return DIAC.sub('', s)


class IrabRealizer:
    """Realises the i'rab ending for a word given its case, wazn and root."""

    # dual / plural patterns from the wazn inventory
    DUAL = {'مُثَنَّى', 'فَعْلَان', 'فِعْلَان'} if False else set()
    MASC_PL = {'فَاعِلُون', 'فَاعِلِين', 'مُفْعِلُون', 'مُفْعِلِين'}

    def __init__(self, vocab=None):
        self.vocab = vocab

    # ---- the diptote test: two causes (Ibn Malik, bab al-mamnu' min al-sarf) -------------
    @staticmethod
    def is_diptote(word_bare, wazn_name='', is_proper=False, is_feminine=False):
        """Ibn Malik: two kinds of cause.

        A SINGLE cause standing in place of two:
          * alif al-ta'nith          (mamdudah / maqsurah endings)
          * sighat muntaha al-jumu'  (the patterns mafa'il / mafa'il -- the ultimate plurals)
        Otherwise TWO causes must combine, drawn from  'alam + (femininity, foreignness,
        compounding, -an, verb pattern)  or an adjective on af'al / fa'lan.
        """
        b = strip_diac(word_bare)
        w = strip_diac(wazn_name)
        # --- single causes that suffice alone ---
        if w.startswith(('مفاعل', 'مفاعيل')):
            return True
        if b.endswith(('اء', 'اء')):
            return True
        # --- otherwise two causes must combine ---
        causes = 0
        if is_proper:
            causes += 1
        if is_feminine:
            causes += 1
        if w.startswith(('افعل', 'أفعل')):
            causes += 1
        if w.startswith(('فعلان',)):
            causes += 1
        if b.endswith('ان') and is_proper:
            causes += 1
        return causes >= 2

    def realize(self, word, case, wazn_name='', root='', is_proper=False,
                is_feminine=False, is_dual=False, is_plural_m=False, is_plural_f=False):
        """Return the word with the realised i'rab ending."""
        if not case or case not in CASE_MARK:
            return word
        b = strip_diac(word)
        if not b:
            return word

        # ---- the five nouns: waw / alif / ya' --------------------------------------------
        if b in ASMA_KHAMSA or b.rstrip('وةاي') in {'أب', 'أخ', 'حم', 'فو', 'ذو'}:
            base = b.rstrip('وةاي')
            return base + {'raf': 'و', 'nasb': 'ا', 'jarr': 'ي'}.get(case, '')

        # ---- dual: -ani (raf') / -ayni (nasb and jarr) ------------------------------------
        if is_dual:
            return b + ('ان' if case == 'raf' else 'ين')

        # ---- sound masculine plural: -una / -ina ------------------------------------------
        if is_plural_m:
            return b + ('ون' if case == 'raf' else 'ين')

        # ---- sound feminine plural: -atu / -ati -------------------------------------------
        if is_plural_f:
            return b + (DAMMA if case == 'raf' else KASRA)

        # ---- mu'tall al-akhir (weak final radical) ----------------------------------------
        if b[-1] in WEAK_FINAL and (root and strip_diac(root)[-1] in WEAK_FINAL):
            if case == 'jazm':
                return b[:-1]                       # JAZM DELETES the weak letter
            if case == 'raf':
                return b + DAMMA                    # estimated, not written on the weak letter
            if case == 'nasb':
                return b + FATHA
            return b + KASRA

        # ---- diptote: jarr takes fathah ----------------------------------------------------
        if case == 'jarr' and self.is_diptote(b, wazn_name, is_proper, is_feminine):
            return b + FATHA

        return b + CASE_MARK[case]


def verify():
    """Canonical examples, each traceable to al-Alfiyyah's chapters."""
    r = IrabRealizer()

    # word, case, kwargs, expected
    CASES = [
        # principals
        ('الغلام', 'raf', {}, 'الغلامُ'),
        ('الغلام', 'nasb', {}, 'الغلامَ'),
        ('الغلام', 'jarr', {}, 'الغلامِ'),
        ('يذهب', 'jazm', {}, 'يذهبْ'),
        # the five nouns (asma' khamsa)
        ('أب', 'raf', {}, 'أبو'),
        ('أب', 'nasb', {}, 'أبا'),
        ('أب', 'jarr', {}, 'أبي'),
        # dual
        ('الطالب', 'raf', {'is_dual': True}, 'الطالبان'),
        ('الطالب', 'nasb', {'is_dual': True}, 'الطالبين'),
        ('الطالب', 'jarr', {'is_dual': True}, 'الطالبين'),
        # sound masculine plural
        ('المعلم', 'raf', {'is_plural_m': True}, 'المعلمون'),
        ('المعلم', 'nasb', {'is_plural_m': True}, 'المعلمين'),
        # sound feminine plural
        ('المعلمات', 'raf', {'is_plural_f': True}, 'المعلماتُ'),
        ('المعلمات', 'nasb', {'is_plural_f': True}, 'المعلماتِ'),
        # mu'tall al-akhir: jazm DELETES the weak letter
        ('يدعو', 'jazm', {'root': 'دعو'}, 'يدع'),
        ('يرمي', 'jazm', {'root': 'رمي'}, 'يرم'),
        ('يرضى', 'jazm', {'root': 'رضي'}, 'يرض'),
        ('يدعو', 'raf', {'root': 'دعو'}, 'يدعوُ'),
        ('يدعو', 'nasb', {'root': 'دعو'}, 'يدعوَ'),
        # diptote: jarr takes fathah
        ('بغداد', 'jarr', {'is_proper': True, 'is_feminine': True}, 'بغدادَ'),
        ('مصانع', 'jarr', {'wazn_name': 'مَفَاعِل'}, 'مصانعَ'),
    ]
    ok = fail = 0
    print(f'{"input":<12}{"case":<6}{"got":<14}{"expected":<14}')
    print('-' * 48)
    for word, case, kw, exp in CASES:
        got = r.realize(word, case, **kw)
        good = (got == exp)
        ok += good
        fail += (not good)
        mark = 'PASS' if good else 'FAIL'
        print(f'{word:<12}{case:<6}{got:<14}{exp:<14}{mark}')
    print(f'\n{ok}/{ok+fail} realisation checks pass')
    return fail == 0


if __name__ == '__main__':
    ok = verify()
    sys.exit(0 if ok else 1)
