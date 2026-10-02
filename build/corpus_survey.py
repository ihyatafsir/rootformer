#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corpus_survey.py -- what is ACTUALLY present in the held sources, under orthographic normalisation.

Targeted greps fail on Arabic because the same word appears with different diacritics, alef forms
(ا / أ / إ / آ / ٱ), ya forms (ي / ى), and ta-marbuta (ة / ه). A concept can look "absent" when the
text spells it slightly differently. This normalises first, then counts concept hits per file, so
"missing from my corpus" is a defensible claim rather than an artefact of the search pattern.

Usage: python corpus_survey.py
"""
import glob
import os
import re
import sys
from collections import defaultdict

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')


def norm(s):
    s = DIAC.sub('', s)
    for a, b in (('أ', 'ا'), ('إ', 'ا'), ('آ', 'ا'), ('ٱ', 'ا'),
                 ('ى', 'ي'), ('ة', 'ه'), ('ؤ', 'و'), ('ئ', 'ي')):
        s = s.replace(a, b)
    return s


CONCEPTS = {
    'waw: ma`iyyah (accompaniment)': [r'الواو.{0,30}المعيه', r'واو.{0,20}المعيه', r'بمعني مع'],
    'waw: hal (circumstance)': [r'الواو.{0,30}الحال', r'واو الحال'],
    'waw: isti\\u2019naf': [r'الواو.{0,30}الاستيناف', r'واو الاستيناف'],
    'waw: qasam (oath)': [r'الواو.{0,30}القسم', r'واو القسم'],
    'waw: sababiyyah (cause)': [r'الواو.{0,30}السببيه', r'واو السببيه'],
    'waw: za\\u2019idah (redundant)': [r'الواو الزائده'],
    'waw: thamaniyyah (the eight)': [r'واو الثمانيه', r'الثمانيه'],
    'waw: `atf (coordination)': [r'الواو.{0,25}العطف', r'واو العطف'],
    'waw: rubba': [r'واو الرب', r'الواو.{0,20}رب'],
    'iktifa\\u2019 (sufficing without the operator)': [r'الاكتفاء', r'يكتفي', r'الاستغناء'],
    '`ilal: causes / causality': [r'العلل', r'العلة', r'التعليل'],
    '`ilal: jadaliyyah': [r'العلة الجدليه'],
    '`ilal: qiyasiyyah': [r'العلة القياسيه'],
    '`ilal: ta\\u2019limiyyah': [r'العلة التعليميه'],
    'i\\u2019rab cases': [r'مرفوع', r'منصوب', r'مجرور', r'مجزوم'],
    'definiteness (ma\\u2019rifah/nakirah)': [r'المعرفه', r'النكره'],
    'governance (\\u2018amal / \\u2018amil)': [r'العامل', r'المعمول', r'الاعمال', r'عمل'],
    'constituency / valency': [r'الاسناد', r'المسند', r'الفضله', r'العمده'],
    'tasrif / derivation': [r'التصريف', r'الاشتقاق', r'الابدال'],
    'i\\u2019lal (weak-root change)': [r'الالال'.replace('الالال', 'الاعتلال'), r'المعتل', r'الاعلال'],
    'idgham (gemination)': [r'الادغام'],
    'ibdal (substitution)': [r'الابدال'],
    'hamza rules': [r'الهمزه', r'تخفيف الهمزه', r'تحقيق الهمزه'],
}


def main():
    root = '/home/grem3/Documents/deepseek-harness/default-workspace/rootformer/corpus'
    files = sorted(glob.glob(f'{root}/**/*.txt', recursive=True))
    print(f'[*] {len(files)} source files\n')
    texts = {}
    for f in files:
        try:
            texts[os.path.basename(f)] = norm(open(f, encoding='utf-8', errors='ignore').read())
        except Exception:
            pass

    print(f'{"concept":<46}{"files":>7}{"hits":>8}   best source')
    print('-' * 100)
    missing = []
    for name, pats in CONCEPTS.items():
        tot = 0
        per = {}
        for fn, t in texts.items():
            # the PATTERNS must be normalised too, or any pattern containing
            # ة / أ / إ / ى can never match normalised text
            c = sum(len(re.findall(norm(p), t)) for p in pats)
            if c:
                per[fn] = c
                tot += c
        if per:
            best = max(per.items(), key=lambda x: x[1])
            print(f'{name:<46}{len(per):>7}{tot:>8}   {best[0]} ({best[1]})')
        else:
            print(f'{name:<46}{0:>7}{0:>8}   -- ABSENT --')
            missing.append(name)

    print(f'\n=== concepts with NO hits in the held corpus ({len(missing)}) ===')
    for m in missing:
        print(f'   - {m}')
    if not missing:
        print('   (none -- every concept is attested somewhere in the held sources)')


if __name__ == '__main__':
    main()
