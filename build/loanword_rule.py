#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loanword_rule.py -- al-Khalil's test for words that CANNOT be rooted (al-mu'arrab).

Kitab al-'Ayn states it directly:

    "الحروف الذلق: ر، ل، ن، ف، ب، م. فإذا جاءت كلمة رباعية أو خماسية لا يكون فيها واحد من
     هذه الستة، فاعلم أنها ليست بعربية"
    -- the dhalq/labial letters are ر ل ن ف ب م. If a quadriliteral or quinqueliteral word
       contains NONE of them, know that it is not Arabic.

and he applies it himself:

    "الدعشوقة ... وليست بعربية محضة لتعريتها من حروف الذلق والشفوية"

CRITICAL DETAIL (my first attempt got this wrong): the definite article is NOT a radical. The ل
in الحضاثج is the article's lam; the word's own letters are ح ض ا ث ج, which contain no dhalq
letter -- which is exactly why al-Khalil calls it non-Arabic. Strip ال before applying the test.

WHY IT MATTERS: loan words have NO Arabic root. Forcing one on them is a direct source of both the
<UNK> residue and the unsound (root, pattern) attributions (e.g. 'ballah' -> root ball, which is
a preposition + the Divine Name, not a root at all). The analyser must mark this class rootless.
"""
import re

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
DHALQ = set('رلنفبم')          # al-huruf al-dhalq + the labials, as al-Khalil lists them


def bare(w):
    s = DIAC.sub('', w)
    for a, b in (('أ', 'ا'), ('إ', 'ا'), ('آ', 'ا'), ('ٱ', 'ا'), ('ى', 'ي'), ('ة', 'ه'),
                 ('ؤ', 'و'), ('ئ', 'ي')):
        s = s.replace(a, b)
    return s


def strip_article(w):
    b = bare(w)
    if b.startswith('ال') and len(b) > 4:
        return b[2:]
    if b.startswith('ا') and len(b) > 4:
        return b[1:]
    return b


def is_non_arabic(word, min_len=4):
    """al-Khalil's dhalq test. Returns (non_arabic, bare_form, dhalq_letters_present)."""
    b = strip_article(word)
    if len(b) < min_len:
        return False, b, []
    present = sorted({c for c in b if c in DHALQ})
    return (not present), b, present


if __name__ == '__main__':
    cases = [('الحضاثج', False), ('الخضعثج', False), ('الدعشوقة', False),
             ('الغلام', True), ('فرس', True), ('كتاب', True)]
    for w, should_be_arabic in cases:
        na, b, present = is_non_arabic(w)
        print(f'  {w:<12} bare={b:<9} dhalq={",".join(present) or "NONE":<8} '
              f'-> {"NOT Arabic" if na else "Arabic"}')
