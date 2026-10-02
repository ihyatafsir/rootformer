#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iktifa_apocope.py -- Sibawayh's al-Iktifa' (clitic saturation) and jussive apocope.

TWO halves, both about slots and endings:

(1) AL-IKTIFA' -- "sufficing". Sibawayh, al-Kitab: باب ما يكتفى فيه بالشيء عن الشيء. A valency
    slot can be closed three ways:
      * filled      -- the ma'mul appears
      * LICITLY ABSENT -- the verb is lazim, or the noun suffices for its khabar
      * SATURATED BY A CLITIC -- an attached pronoun FILLS the slot, so no further ma'mul is
        expected:  ضربَهُ  (the -hu IS the maf'ul),  ضربتُ (the -tu IS the fa'il),
        منه / فيه  (the pronoun IS the majrur).
    This matters because without it the constituent stack keeps awaiting a ma'mul that will never
    come, and mis-assigns whatever follows. The (prefix, root, wazn, SUFFIX) tuple already carries
    the clitic, so saturation is computable.

    The clitic's SLOT depends on its HOST -- which is Sibawayh's point:
      after a VERB        -hu is maf'ul bihi        -tu is fa'il
      after a NOUN        -hu is mudaf ilayhi       (possessive)
      after a PREPOSITION -hu is majrur

(2) JUSSIVE APOCOPE -- the realisation rule for jazm and for forming the imperative:
      sound verb          jazm -> sukun                 lam yadhhab
      weak-final (naqis)  jazm -> the weak letter is DELETED   lam yad'u, lam yarmi, lam yarda
      hollow (ajwaf)      the weak letter is deleted    lam yaqul  (from yaqulu)
      with a clitic       NO apocope -- the suffix protects the ending   lam yadribhu
      energetic nun       NO apocope -- the nun protects it
    IMPERATIVE is formed from the jussive stem: if that stem begins with a consonant cluster a
    prothetic hamzat al-wasl is added -- with dammah if the 'ayn is vowelled with dammah,
    otherwise kasrah. A HOLLOW verb drops its weak letter, so no prothetic is needed.

Usage:
  python iktifa_apocope.py        # verifies against canonical examples
"""
import re
import sys

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640]')
FATHA, DAMMA, KASRA, SUKUN = '\u064e', '\u064f', '\u0650', '\u0652'
WEAK = set('وايى')

# ---- clitics, by the slot they saturate, given the host ------------------------------------
VERB_CLITICS = {
    'ه': 'maf_ul_bihi', 'ها': 'maf_ul_bihi', 'هما': 'maf_ul_bihi', 'هم': 'maf_ul_bihi',
    'هن': 'maf_ul_bihi', 'ك': 'maf_ul_bihi', 'كما': 'maf_ul_bihi', 'كم': 'maf_ul_bihi',
    'ني': 'maf_ul_bihi', 'نا': 'maf_ul_bihi',
    'ت': 'fa_il', 'tu': 'fa_il', 'تُ': 'fa_il', 'تُما': 'fa_il', 'تُم': 'fa_il',
    'تُنَّ': 'fa_il', 'وا': 'fa_il', 'ن': 'fa_il', 'تُ': 'fa_il', 'ا': 'fa_il',
}
NOUN_CLITICS = {'ه': 'mudaf_ilayhi', 'ها': 'mudaf_ilayhi', 'هم': 'mudaf_ilayhi',
                'ك': 'mudaf_ilayhi', 'نا': 'mudaf_ilayhi', 'ي': 'mudaf_ilayhi',
                'كم': 'mudaf_ilayhi', 'هن': 'mudaf_ilayhi'}
PREP_CLITICS = {'ه': 'majruz_bil_jarr', 'ها': 'majruz_bil_jarr', 'هم': 'majruz_bil_jarr',
                'ك': 'majruz_bil_jarr', 'نا': 'majruz_bil_jarr', 'ي': 'majruz_bil_jarr',
                'كم': 'majruz_bil_jarr', 'هن': 'majruz_bil_jarr'}

ENERGETIC = {'نَّ', 'نِ'}


def strip_diac(s):
    return DIAC.sub('', s)


class IktifaSaturation:
    """Sibawayh's iktifa': which valency slot a clitic saturates."""

    @staticmethod
    def saturates(suffix, host_class):
        """host_class in {'verb', 'noun', 'prep'}; returns the saturated slot or None."""
        s = strip_diac(suffix or '')
        if not s:
            return None
        table = {'verb': VERB_CLITICS, 'noun': NOUN_CLITICS, 'prep': PREP_CLITICS}[host_class]
        # longest match wins, so 'كما' is not read as 'ك'
        for probe in sorted(table, key=len, reverse=True):
            if s == probe or s.endswith(probe):
                return table[probe]
        return None

    @staticmethod
    def closes_slot(host_class, suffix, verb_is_lazim=False, noun_suffices=False):
        """True if the slot is closed -- filled, saturated, or licitly absent."""
        if IktifaSaturation.saturates(suffix, host_class):
            return True, 'saturated by clitic'
        if host_class == 'verb' and verb_is_lazim:
            return True, 'lazim -- no maf\'ul expected'
        if host_class == 'noun' and noun_suffices:
            return True, 'noun suffices for its khabar'
        return False, 'slot still open'


class JussiveApocope:
    """The realisation rule for jazm, and imperative formation from the jussive stem."""

    @staticmethod
    def jussive(verb_bare, root='', has_clitic=False, has_energetic=False):
        b = strip_diac(verb_bare)
        if not b:
            return verb_bare
        if has_clitic or has_energetic:
            return b                      # the suffix / nun protects the ending
        r = strip_diac(root) if root else ''
        if r and len(r) == 3 and r[-1] in WEAK and b[-1] in WEAK:
            return b[:-1]                 # naqis: delete the final weak letter
        if r and len(r) == 3 and r[1] in WEAK:
            # ajwaf (hollow): the medial weak letter is elided in the jussive
            for i in range(1, len(b)):
                if b[i] in WEAK:
                    return b[:i] + b[i + 1:]
        return b + SUKUN

    ARABIC = set('ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآؤئى')

    @staticmethod
    def imperative(imperfect_bare, root='', wazn_name=''):
        """Form the amr from the imperfect (Ibn Malik, Lamiyyat al-Af'al).

        Two corrections that matter:
          * the prothetic hamzat al-wasl is needed ONLY if the stem would begin with a consonant
            cluster. In يَعِدُ the first radical already carries a kasrah, so the stem is عِد --
            no prothetic (my first version wrongly produced اِعد).
          * the prothetic's VOWEL is read from the WAZN, not the bare string: dammah if the 'ayn
            carries dammah (يَفْعُلُ -> اُنصر), otherwise kasrah (يَفْعِلُ -> اِضرب).
        """
        b = strip_diac(imperfect_bare)
        if not b:
            return imperfect_bare
        r = strip_diac(root) if root else ''
        core = b[1:] if b[0] in 'يتنأ' else b
        # hollow: the medial weak letter is dropped outright, no prothetic needed
        if r and len(r) == 3 and r[1] in WEAK:
            for i in range(len(core)):
                if core[i] in WEAK:
                    return core[:i] + core[i + 1:]
        # mithal (assimilated): the initial weak letter drops
        if core and core[0] in WEAK:
            return core[1:]

        # read the MUDARA'AH pattern: is the first radical vowelled? what vowel does the 'ayn take?
        w = wazn_name or ''
        idx = [i for i, ch in enumerate(w) if ch in JussiveApocope.ARABIC]
        needs_prothetic, ayn_damma = True, False
        if len(idx) >= 2:
            first_rad = idx[1]
            after = w[first_rad + 1] if first_rad + 1 < len(w) else ''
            needs_prothetic = (after == SUKUN)
            if len(idx) >= 3:
                ayn = idx[2]
                ayn_vowel = w[ayn + 1] if ayn + 1 < len(w) else ''
                ayn_damma = (ayn_vowel == DAMMA)
        if not w:
            # no wazn: fall back to the classical default (kasrah) unless the bare form hints dammah
            needs_prothetic = True

        if not needs_prothetic:
            return core
        return ('ا' + DAMMA if ayn_damma else 'ا' + KASRA) + core


def verify():
    ik, ja = IktifaSaturation(), JussiveApocope()
    ok = fail = 0
    print('--- AL-IKTIFA\': clitic saturation (slot depends on host) ---')
    for suffix, host, expect in (('ـهُ', 'verb', 'maf_ul_bihi'), ('ـتُ', 'verb', 'fa_il'),
                                 ('ـهُ', 'noun', 'mudaf_ilayhi'), ('ـنا', 'noun', 'mudaf_ilayhi'),
                                 ('ـهُ', 'prep', 'majruz_bil_jarr'), ('ـكم', 'prep', 'majruz_bil_jarr'),
                                 ('', 'verb', None)):
        got = ik.saturates(suffix, host)
        good = (got == expect)
        ok += good; fail += not good
        print(f'  {suffix or "(none)":<8} after {host:<5} -> {str(got):<18} '
              f'{"PASS" if good else "FAIL expect " + str(expect)}')
    closed, why = ik.closes_slot('verb', 'ـهُ')
    print(f'  slot closure: {closed} ({why})')

    print('\n--- JUSSIVE APOCOPE ---')
    for verb, root, kw, expect in (
            ('يذهب', 'ذهب', {}, 'يذهبْ'),
            ('يدعو', 'دعو', {}, 'يدع'),
            ('يرمي', 'رمي', {}, 'يرم'),
            ('يرضى', 'رضي', {}, 'يرض'),
            ('يقول', 'قول', {}, 'يقل'),
            ('يضربه', 'ضرب', {'has_clitic': True}, 'يضربه'),
            ('يكتبن', 'كتب', {'has_energetic': True}, 'يكتبن'),
    ):
        got = ja.jussive(verb, root, **kw)
        good = (got == expect)
        ok += good; fail += not good
        print(f'  {verb:<10} -> {got:<10} expect {expect:<10} {"PASS" if good else "FAIL"}')

    print('\n--- IMPERATIVE from the jussive stem ---')
    for imp, root, wazn, expect in (('يكتب', 'كتب', 'يَفْعُلُ', 'اُكتب'),
                                    ('ينصر', 'نصر', 'يَنْصُرُ', 'اُنصر'),
                                    ('يبيع', 'بيع', 'يَبِيعُ', 'بع'),
                                    ('يعد', 'وعد', 'يَعِدُ', 'عد'),
                                    ('يفتح', 'فتح', 'يَفْتَحُ', 'اِفتح')):
        got = ja.imperative(imp, root, wazn)
        good = (got == expect)
        ok += good; fail += not good
        print(f'  {imp:<10} wazn {wazn:<10} -> {got:<10} expect {expect:<10} '
              f'{"PASS" if good else "FAIL"}')

    print(f'\n{ok}/{ok+fail} checks pass')
    return fail == 0


if __name__ == '__main__':
    sys.exit(0 if verify() else 1)
