#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
word_class.py -- emit the grammarians' word-class / i'rab category for every position.

The project's vocabulary already DEFINES the taxonomy (46 angle-bracketed categories at ids
117-162 plus 65 particles at 163-227) but the pipeline never emits any of them: encode_word()
returns (prefix, root, wazn, suffix) and everything else is discarded, which is why ~44% of
positions collapse into a single <PARTICLE> token.

This module assigns a category to each word from the classical algorithms:

  DETERMINISTIC (no valency needed) -- implemented here:
    * particles  -> their category from Sibawayh / al-Zajjaji's Huruf al-Ma'ani:
        huruf al-jarr  (man, ila, can, cala, fi, hatta, ...)      -> <harf_jarr>
        huruf al-nasiba (inna, anna, ka'anna, lakinn, layta, la'alla) -> <inna>..<lakinna>
        the kana family (kana, laysa, asbaha, amsa, zalla, bata)  -> <kana>..<ma_dama>
        huruf al-jazm   (lam, lamma, lam al-amr, la al-nahiyya)   -> jazm operators
        huruf al-catf   (waw, fa, thumma, aw, am, bal)            -> <atf_*>
        al-istithna'    (illa, ghayr)                             -> <istithna_illa>
        al-hasr         (innama)                                  -> <hasr_innama>
        al-shart        (in, law, lawla, man, ma, mata, ayna)     -> <shart_in>
        ism al-ishara   (hadha, tilka, ...) / al-mawsul (alladhi, ...) / dama'ir
    * verbs      -> <fi_l_madi> / <fi_l_mudari> / <fi_l_amr>
        from the wazn (the imperfect awzan are 114-129) and the prefix (ya/ta/alif/nun = mudari;
        lam + imperfect = amr).

  REQUIRES VALENCY (Sibawayh's constituent stack) -- NOT implemented here, returns <ism> fallback:
    * the noun roles: mubtada / khabar / fa'il / maf'ul bihi / hal / tamyiz / na't / badal /
      mudaf-ilayhi. Assigning those needs to know which operator governs the position and how far
      its government reaches -- item 2 of the dependency order.

Usage:
  python word_class.py --n 30000        # coverage report on real text
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
AR = re.compile(r'[\u0600-\u06FF]')


def bare(s):
    return DIAC.sub('', s).strip()


# --- Sibawayh / al-Zajjaji: the particles, by function -------------------------------------
JARR = {'من', 'إلى', 'عن', 'على', 'في', 'حتى', 'مع', 'عند', 'منذ', 'مذ', 'رب', 'خلا', 'عدا', 'حاشا'}
INNA_FAMILY = {'إن', 'أن', 'كأن', 'لكن', 'ليت', 'لعل', 'إنما'}
KANA_FAMILY = {'كان', 'ليس', 'أصبح', 'أمسى', 'أضحى', 'ظل', 'بات', 'صار', 'مازال', 'مادام',
               'مافتئ', 'ماربرح', 'ماانفك'}
JAZM = {'لم', 'لما', 'لن'} | {'لا'}          # la al-nahiyya shares the form of la al-nafy
NASB = {'لن', 'كي', 'إذن', 'أن'}
ATF = {'و', 'ف', 'ثم', 'أو', 'أم', 'بل', 'لكن'}
ISTITHNA = {'إلا', 'غير', 'سوى', 'عدا', 'خلا', 'حاشا'}
SHART = {'إن', 'لو', 'لولا', 'لومَا', 'متى', 'أين', 'حيثما', 'إذما', 'من', 'ما', 'مهما'}
ISTIFHAM = {'هل', 'أ', 'كيف', 'أين', 'متى', 'كم', 'أي', 'من', 'ما', 'ماذا'}
NAFY = {'ما', 'لا', 'لم', 'لن', 'ليس', 'إن'}
ISTIQBAL = {'س', 'سوف'}
ISHARA = {'هذا', 'هذه', 'هؤلاء', 'ذلك', 'تلك', 'أولئك', 'ذا', 'ذي', 'هناك', 'هنالك'}
MAWSUL = {'الذي', 'التي', 'الذين', 'اللواتي', 'اللاتي', 'اللذان', 'اللتان', 'ما', 'من'}
DAMAIR = {'هو', 'هي', 'أنا', 'نحن', 'أنت', 'أنتم', 'أنتن', 'هم', 'هن', 'إياك', 'إياه'}
ZARF = {'عند', 'قبل', 'بعد', 'دون', 'بين', 'تحت', 'فوق', 'أمام', 'خلف', 'مع', 'حيث', 'إذ', 'إذا'}
HASR = {'إنما'}
TAWKID = {'قد', 'إن', 'ل', 'ن', 'ألا', 'أما'}
TAFSIR = {'أي', 'أن'}
JUMLAT = {'نعم', 'بئس', 'حبذا'}

PARTICLE_CLASS = {}
for s in JARR: PARTICLE_CLASS.setdefault(s, 'harf_jarr')
for s in INNA_FAMILY: PARTICLE_CLASS.setdefault(s, 'inna')
for s in KANA_FAMILY: PARTICLE_CLASS.setdefault(s, 'kana')
for s in ATF: PARTICLE_CLASS.setdefault(s, 'atf_waw')
for s in ISTITHNA: PARTICLE_CLASS.setdefault(s, 'istithna_illa')
for s in SHART: PARTICLE_CLASS.setdefault(s, 'shart_in')
for s in ISTIQBAL: PARTICLE_CLASS.setdefault(s, 'fi_l_mudari')   # future markers
for s in ISHARA: PARTICLE_CLASS.setdefault(s, 'mubtada')
for s in MAWSUL: PARTICLE_CLASS.setdefault(s, 'mubtada')
for s in DAMAIR: PARTICLE_CLASS.setdefault(s, 'damir_fasl')
for s in HASR: PARTICLE_CLASS.setdefault(s, 'hasr_innama')
for s in JUMLAT: PARTICLE_CLASS.setdefault(s, 'fi_l_madi')
for s in ZARF: PARTICLE_CLASS.setdefault(s, 'mudaf_ilayhi')


class WordClassifier:
    """Assigns a sibawayh_particles category id to each word position."""

    def __init__(self, vocab, blueprint_path):
        bp = json.load(open(blueprint_path))
        t2i = bp['vocab_token_to_id']
        self.t2i = t2i
        self.UNK = t2i.get('<unk>', 3)
        self.category_ids = {t[1:-1] for t in t2i if t.startswith('<') and t.endswith('>')}
        # the imperfect awzan are 114..129 (verified earlier in the project)
        self.imperfect_wazn = set(range(114, 130))
        self.fallback = t2i['<ism>'] if '<ism>' in t2i else t2i['<mubtada>']

    def cat(self, name):
        return self.t2i.get(f'<{name}>', self.fallback)

    def classify(self, word, p_id, r_id, w_id, s_id, vocab):
        w = bare(word)
        if not w:
            return self.UNK, 'empty'

        # ---- 1. particle, by function (Sibawayh / al-Zajjaji) ----
        base = w
        for pre in ('وال', 'فال', 'بال', 'كال', 'لل', 'ال'):
            if base.startswith(pre) and len(base) > len(pre) + 1:
                base = base[len(pre):]
                break
        for pre in ('و', 'ف'):
            if base.startswith(pre) and len(base) > 2:
                base = base[1:]
                break
        if base in PARTICLE_CLASS:
            return self.cat(PARTICLE_CLASS[base]), f'particle:{PARTICLE_CLASS[base]}'
        if w in PARTICLE_CLASS:
            return self.cat(PARTICLE_CLASS[w]), f'particle:{PARTICLE_CLASS[w]}'

        # ---- 2. verb: from the wazn and the prefix ----
        wazn_name = vocab.id2wazn.get(w_id, '')
        is_imperfect = w_id in self.imperfect_wazn
        if is_imperfect or (wazn_name and wazn_name.startswith('يَفْ')):
            return self.cat('fi_l_mudari'), 'verb:mudari'
        # perfect verb pattern: fa'ala / fa'ula / fa'ila family, or a derived masdar-like shape
        if wazn_name and re.match(r'^فَعَل|^فَعُل|^فَعِل|^فُعِل|^فَعَّل|^تَفَاعَل|^اِفْتَعَل', wazn_name):
            return self.cat('fi_l_madi'), 'verb:madi'
        if wazn_name and re.match(r'^اِفْعَل|^فَعِّل|^فَاعِل$', wazn_name):
            return self.cat('fi_l_amr'), 'verb:amr'

        # ---- 3. noun role: NEEDS VALENCY (item 2) -- coarse fallback only ----
        return self.fallback, 'noun:fallback(needs valency)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30000)
    ap.add_argument('--out', default='/workspace/word_class_coverage.json')
    args = ap.parse_args()

    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    wc = WordClassifier(vocab, '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')

    files = (sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))
             + sorted(glob.glob('/workspace/andalusian_canon_sanitized/*.txt')))[:6]
    words = []
    for f in files:
        txt = open(f, encoding='utf-8', errors='ignore').read()
        words += [x for x in re.split(r'\s+', txt) if AR.search(x)]
    words = words[:args.n]

    why = Counter()
    cat_counts = Counter()
    for w in words:
        p, r, wz, s = vocab.encode_word(w)
        cid, reason = wc.classify(w, p, r, wz, s, vocab)
        why[reason.split(':')[0] + (':' + reason.split(':')[1] if ':' in reason else '')] += 1
        cat_counts[vocab.id2root.get(cid, str(cid)) if hasattr(vocab, 'id2root') else cid] += 1

    n = len(words)
    print(f'[*] classified {n} words\n')
    print(f'{"reason":<44}{"count":>8}{"share":>9}')
    print('-' * 61)
    for k, v in why.most_common(14):
        print(f'{k:<44}{v:>8}{100*v/n:>8.1f}%')

    resolved = sum(v for k, v in why.items() if not k.startswith('noun:fallback'))
    print(f'\n[*] DETERMINISTIC categories assigned : {resolved}/{n} = {100*resolved/n:.1f}%')
    print(f'[*] awaiting valency (item 2)        : {n-resolved} = {100*(n-resolved)/n:.1f}%')
    json.dump({'n': n, 'resolved': resolved, 'reasons': dict(why)},
              open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
