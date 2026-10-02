#!/usr/bin/env python3
"""Verification v4 -- nearest-source attribution + non-citation filtering.

Builds on v3's matcher (imported) but:
  * attributes each inline quote to the NEAREST named author/work in `pre`
    (falling back to `post`), not any name in a 700-char window;
  * flags spans that are code/test artefacts rather than citation claims.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_citations_v3 as V  # noqa: E402

LAT_AUTH = (r"(Sibawayh|Sībawayh|Ibn Jinni|Ibn Jinnī|Ibn Malik|Ibn Mālik|"
            r"Abu Hayyan|Abū Ḥayyān|al-Shatibi|al-Shāṭibī|Shatibi|Shāṭibī|"
            r"Ibn 'Usfur|Ibn Usfur|Ibn ʿUsfūr|al-Khalil|al-Khalīl|Khalil|"
            r"al-Mubarrad|Mubarrad|Ibn Faris|Ibn Fāris|al-Zajjaji|al-Zajjājī|"
            r"Zajjaji|al-Suhayli|al-Suhaylī|Suhayli|Ibn Mada|Ibn Madā|"
            r"al-Jawhari|al-Azhari|Ibn Durayd|al-Akhfash|al-Zamakhshari|al-Raghib|"
            r"Ibn Sayyidih|al-Sakhawi|Ibn al-Sikkit)")
AR_AUTH = (r"(الخليل|سيبويه|ابن جني|ابن مالك|أبو حيان|الشاطبي|ابن عصفور|المبرد|"
           r"ابن فارس|الزجاجي|ابن مضاء|ابن سيده|السخاوي|ابن دريد|الجوهري|الأزهري|"
           r"الزمخشري|الراغب|ابن السكيت|أبو الفتح|ابن مالك)")

WORK_TOKEN = (r"(al-Tashil|al-Tashīl|Tashil|al-Alfiyyah|Alfiyyah|Lamiyyat al-Af'al|"
              r"Lāmiyyat al-Afʿāl|al-Kitab|al-Kitāb|al-Khasais|al-Khasa'is|"
              r"Sirr Sina'at al-I'rab|Sirr|al-Muqtadab|al-Sahibi|al-Sāhibī|"
              r"Maqayis al-Lughah|Maqāyīs|al-Sihah|al-Ṣiḥāḥ|Tahdhib al-Lughah|"
              r"Irtishaf al-Darab|Irtishāf|Tadhyil al-Tashil|Tadhyīl|"
              r"al-Muwafaqat|al-Muwāfaqāt|Sharh al-Alfiyyah|Sharh Jumal al-Zajjaji|"
              r"al-Mumti'|al-Mumtiʿ|Nata'ij al-Fikr|al-Mukhassas|al-Muhkam|"
              r"Jumal al-Zajjaji|al-Idah|al-Ayn|al-ʿAyn)")

RULE_LAT = re.compile(LAT_AUTH + r"[\s,:'’\-]{0,4}" + WORK_TOKEN, re.I)
RULE_LAT_AUTHONLY = re.compile(LAT_AUTH)
RULE_AR = re.compile(AR_AUTH)
RULE_ARWORK = re.compile(AR_AUTH + r"[\s،:]{0,3}" + WORK_TOKEN)


def nearest_claim(pre, post):
    """Return (author_token, work_token_or_None) nearest the quote."""
    best = None
    for m in RULE_LAT.finditer(pre):
        best = (m.group(1), m.group(2), m.start())
    for m in RULE_ARWORK.finditer(pre):
        best = (m.group(1), m.group(2), m.start())
    if best:
        return best[0], best[1]
    # author only, nearest
    cands = []
    for m in RULE_LAT_AUTHONLY.finditer(pre):
        cands.append((m.start(), m.group(1), None))
    for m in RULE_AR.finditer(pre):
        cands.append((m.start(), m.group(1), None))
    if cands:
        cands.sort()
        return cands[-1][1], None
    # fall back to post
    m = RULE_LAT.search(post) or RULE_AR.search(post)
    if m:
        return m.group(1), None
    return None, None


def norm_author(a):
    if not a:
        return None
    a = a.strip().lower()
    table = {
        'sibawayh': 'sibawayh', 'sībawayh': 'sibawayh',
        'ibn jinni': 'ibn jinni', 'ibn jinnī': 'ibn jinni',
        'ibn malik': 'ibn malik', 'ibn mālik': 'ibn malik',
        'abu hayyan': 'abu hayyan', 'abū ḥayyān': 'abu hayyan',
        'al-shatibi': 'shatibi', 'al-shāṭibī': 'shatibi', 'shatibi': 'shatibi',
        'shāṭibī': 'shatibi',
        "ibn 'usfur": 'usfur', 'ibn usfur': 'usfur', 'ibn ʿusfūr': 'usfur',
        'al-khalil': 'khalil', 'al-khalīl': 'khalil', 'khalil': 'khalil',
        'al-mubarrad': 'mubarrad', 'mubarrad': 'mubarrad',
        'ibn faris': 'ibn faris', 'ibn fāris': 'ibn faris',
        'al-zajjaji': 'zajjaji', 'al-zajjājī': 'zajjaji', 'zajjaji': 'zajjaji',
        'al-suhayli': 'suhayli', 'al-suhaylī': 'suhayli', 'suhayli': 'suhayli',
        'ibn mada': 'ibn mada', 'ibn madā': 'ibn mada',
        'الخليل': 'khalil', 'سيبويه': 'sibawayh', 'ابن جني': 'ibn jinni',
        'ابن مالك': 'ibn malik', 'أبو حيان': 'abu hayyan', 'الشاطبي': 'shatibi',
        'ابن عصفور': 'usfur', 'المبرد': 'mubarrad', 'ابن فارس': 'ibn faris',
        'الزجاجي': 'zajjaji', 'ابن مضاء': 'ibn mada', 'ابن سيده': 'ibn sayyidih',
        'السخاوي': 'sakhawi', 'ابن دريد': 'ibn durayd', 'الجوهري': 'jawhari',
        'الأزهري': 'azhari', 'الزمخشري': 'zamakhshari', 'الراغب': 'raghib',
        'ابن السكيت': 'ibn sikkit', 'أبو الفتح': 'ibn jinni',
        'al-jawhari': 'jawhari', 'al-azhari': 'azhari', 'ibn durayd': 'ibn durayd',
        'al-akhfash': 'akhfash', 'al-zamakhshari': 'zamakhshari',
        'al-raghib': 'raghib', 'ibn sayyidih': 'ibn sayyidih',
        'al-sakhawi': 'sakhawi', 'ibn al-sikkit': 'ibn sikkit',
    }
    return table.get(a, a)


CODEY = re.compile(r"(print\s*\(|assert\b|def test|__main__|\breturn\b|JSON|\.json|"
                   r"f['\"]|\.format\(|logging)")


def is_claim(r):
    q = r['arabic']
    if r.get('n_words', 0) < 2:
        return False
    if '{' in q or '}' in q:
        return False
    line = (r.get('pre') or '').split('\n')[-1] + (r.get('post') or '').split('\n')[0]
    if CODEY.search(line) and not r.get('in_docstring'):
        return False
    if re.match(r'^\s*(Lāmiyyah|Alfiyyah|al-Alfiyyah|v\.\d)', q):
        return False
    return True


def main():
    ext, out = sys.argv[1], sys.argv[2]
    files, by_base = V.load_corpus()
    data = json.load(open(ext, encoding='utf-8'))
    results = []

    for r in data['structured']:
        quote = r['arabic']
        src = r.get('src_file') or ''
        base = os.path.basename(src)
        declared_held = base in by_base
        prefer = {base} if declared_held else {
            b for k in V.named((r.get('book') or '') + ' ' + src) for b in V.WORK_MAP.get(k, [])}
        ph, oh = V.find(quote, files, prefer)
        named_keys = V.named((r.get('book') or '') + ' ' + src)
        unheld_named = [k for k in named_keys if k in V.UNHELD]
        if ph is not None:
            cls, hit = ph['cls'], ph          # found in the work it names -> correct
        elif oh is not None and oh['score'] >= 0.9:
            cls, hit = 'MISATTRIBUTED', oh
        elif oh is not None and oh['score'] >= 0.6:
            cls, hit = 'PARAPHRASE', oh
        elif declared_held:
            cls, hit = 'ABSENT', None
        elif unheld_named:
            cls, hit = 'UNCHECKABLE', None
        elif not declared_held and src:
            cls, hit = 'UNCHECKABLE', None     # names a file we do not hold
        else:
            cls, hit = 'ABSENT', None
        dl = r.get('src_line')
        try:
            dl = int(str(dl).strip())
        except (TypeError, ValueError):
            dl = None
        results.append(dict(
            module=r['module'], module_line=r['module_line'], kind='structured',
            claim_author=None, claim_work=r.get('book'),
            declared_file=src, declared_line=dl, declared_held=declared_held,
            cls=cls, found_file=hit['file'] if hit else None,
            found_line=hit['line'] if hit else None,
            level=hit['level'] if hit else None, score=hit['score'] if hit else 0.0,
            line_exact=(hit['line'] == dl) if (hit and dl and declared_held
                                               and os.path.basename(hit['file']) == base) else None,
            snippet=V.snippet(files, hit['file'], hit['line']) if hit else None,
            arabic=quote))

    for r in data['inline']:
        quote = r['arabic']
        author, work = nearest_claim(r.get('pre') or '', r.get('post') or '')
        a = norm_author(author)
        prefer = set(V.WORK_MAP.get(a, []))
        unheld = [a] if (a in V.UNHELD) else []
        ph, oh = V.find(quote, files, prefer)
        if ph is not None:
            cls, hit = ph['cls'], ph
        elif oh is not None and oh['score'] >= 0.9:
            cls, hit = ('MISATTRIBUTED' if a else 'VERBATIM-elsewhere'), oh
        elif oh is not None and oh['score'] >= 0.6:
            cls, hit = 'PARAPHRASE', oh
        else:
            cls, hit = ('UNCHECKABLE' if unheld else 'ABSENT'), None
        results.append(dict(
            module=r['module'], module_line=r['module_line'], kind=r['kind'],
            claim_author=author, claim_work=work, is_claim=is_claim(r),
            in_docstring=r.get('in_docstring'),
            declared_file=None, declared_line=None, declared_held=None,
            cls=cls, found_file=hit['file'] if hit else None,
            found_line=hit['line'] if hit else None, level=hit['level'] if hit else None,
            score=hit['score'] if hit else 0.0, line_exact=None,
            n_words=r.get('n_words'), unheld=unheld,
            snippet=V.snippet(files, hit['file'], hit['line']) if hit else None,
            arabic=quote))

    json.dump({'results': results, 'corpus_files': [f['path'] for f in files]},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    from collections import Counter
    st = [x for x in results if x['kind'] == 'structured']
    il = [x for x in results if x['kind'] != 'structured']
    ic = [x for x in il if x.get('is_claim')]
    print(f"\n=== STRUCTURED registry (n={len(st)}) ===")
    print(Counter(x['cls'] for x in st))
    print("declared file held:", sum(1 for x in st if x['declared_held']), "/", len(st))
    print("declared line == found line:", sum(1 for x in st if x['line_exact'] is True),
          "| mismatch:", sum(1 for x in st if x['line_exact'] is False))
    print("\nlevels:", Counter(x['level'] for x in st))
    print(f"\n=== INLINE claims (n={len(ic)} of {len(il)} raw spans) ===")
    print(Counter(x['cls'] for x in ic))
    print("\nauthors claimed:", Counter(x['claim_author'] for x in ic).most_common(20))
    print(f"\n=== ALL claims (n={len(st)+len(ic)}) ===")
    print(Counter([x['cls'] for x in st] + [x['cls'] for x in ic]))


if __name__ == '__main__':
    main()
