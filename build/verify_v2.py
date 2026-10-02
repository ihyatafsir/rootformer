#!/usr/bin/env python3
"""Verify the corrected constraints against the same cases that failed the original."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'models'))

import nrmp_vocab as nv
from classical_governance_v2 import (AlKhalilV2, SibawayhV2, IbnMalikV2, ShatibiWawV2,
                                     load_lexicon, strip_diac)

cls = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
vocab = cls(str(Path(__file__).resolve().parent / 'data/rootformer_v12_arabic_blueprint.json'))
lex = load_lexicon()

print('=' * 74)
print('CORRECTED (v2) CHECKS')
print('=' * 74)
ok = True

print('\n--- K1 Al-Khalil bare-alif false positive ---')
k = AlKhalilV2(vocab)
forb = k.mask()
dead = [r for r in ('اصل', 'ارض', 'اخذ', 'احد', 'اسس', 'افل', 'اثر')
        if r in vocab.root2id and vocab.root2id[r] in forb]
print(f'  legitimate roots still killed: {dead}')
ok &= not dead

print('\n--- K2 empirical phonotactic matrix ---')
print(f'  unobserved adjacent pairs: {len(k.unobserved_pairs)} '
      f'(these are impossible in the attested inventory)')
print(f'  C1==C2 impossible roots: {len(k.c12)}')

print('\n--- S1/S2 operator classification ---')
s = SibawayhV2(vocab)
def op(word, next_verbal=None):
    return s.operator_of(*vocab.encode_word(word), next_is_verbal=next_verbal)
checks = [('إن', None, 'INNA'), ('أن', None, 'INNA'), ('في', None, 'HARF_JARR'),
          ('إن', True, 'HARF_JAZM'), ('أن', True, 'HARF_NASB'),
          ('لم', None, 'HARF_JAZM'), ('لن', None, 'HARF_NASB'), ('كان', None, 'KANA'),
          ('سوف', None, 'FUTURE'), ('لا', False, 'NONE'), ('لا', True, 'HARF_JAZM')]
for w, nv_, expect in checks:
    got = op(w, nv_)
    flag = 'PASS' if got == expect else 'FAIL'
    ok &= got == expect
    print(f'  [{flag}] {w!r} (next_verbal={nv_}) -> {got} (expect {expect})')

print('\n--- S3 government persistence ---')
def is_verbal(t):
    return t[2] in s.verbal
sents = ['في الأجسام الشفافة', 'في الأجسام', 'من العلم النافع']
for sent in sents:
    enc = vocab.encode_sentence(sent)
    chain = s.government_chain(enc, is_verbal)
    print(f'  {sent:<22} -> {chain}')

print('\n--- M1 coordinator exception ---')
m = IbnMalikV2
a = m.transition_ok(m.FIL, m.FIL, coordinator_between=False)
b = m.transition_ok(m.FIL, m.FIL, coordinator_between=True)
print(f'  [{"PASS" if not a else "FAIL"}] Fi\'l -> Fi\'l WITHOUT coordinator blocked')
print(f'  [{"PASS" if b else "FAIL"}] Fi\'l -> Fi\'l WITH coordinator (و/ف) allowed')
ok &= (not a) and b

print('\n--- K3 Al-Khalil verbatim letter pairs (Kitab al-Ayn) ---')
import re as _re
for r in ('جقق', 'قك', 'كق'):
    print(f'  {r}: in inventory={r in vocab.root2id} killed={r in vocab.root2id and vocab.root2id[r] in forb}')

print('\n--- S6 lam al-amr apocope (Sibawayh + Ibn Jinni) ---')
from classical_governance_v2 import lam_amr_apocope
for pfx, wz, exp in (('ل', 'يَفْعُلُ', True), ('ل', 'فَعَلَ', False), ('ب', 'يَفْعُلُ', False)):
    got = lam_amr_apocope(pfx, wz)
    print(f'  [{"PASS" if got==exp else "FAIL"}] prefix={pfx!r} wazn={wz!r} -> apocope={got}')

print('\n--- M2 Maratib al-Ma\'arif ---')
print(f'  [{"PASS" if m.definiteness_ok("DEFINITE","INDEFINITE") else "FAIL"}] '
      f'definite mubtada + indefinite khabar allowed')
print(f'  [{"PASS" if not m.definiteness_ok("INDEFINITE","DEFINITE") else "FAIL"}] '
      f'indefinite mubtada + definite khabar blocked')

print('\n--- Sh1 waw functions ---')
sh = ShatibiWawV2.classify
cases = [('ISM','ISM',False,False,False,'ATF'), ('ISM','FIL',False,False,False,'HAL'),
         ('FIL','FIL',False,False,False,'ATF'), ('ISM','ISM',True,False,False,'ISTINAF'),
         ('ISM','ISM',False,False,True,'QASAM')]
seen = set()
for pv, nx, st, mf, oath, expect in cases:
    got = sh(pv, nx, st, mf, oath)
    seen.add(got)
    flag = 'PASS' if got == expect else 'FAIL'
    ok &= got == expect
    print(f'  [{flag}] {pv}->{nx} start={st} oath={oath} => {got} (expect {expect})')
print(f'  distinct functions now: {sorted(seen)}')

print('\n' + '=' * 74)
print('ALL v2 CHECKS PASS' if ok else 'SOME v2 CHECKS FAILED')
print('=' * 74)
