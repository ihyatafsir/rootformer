#!/usr/bin/env python3
"""Mechanically verify every extracted citation against the held primary texts.

Classes (mirroring the audit spec):
  VERBATIM      quoted Arabic is an exact substring of a held text
  NEAR_VERBATIM differs only by diacritics/spacing/orthography, and/or uses
                '...' / '#' elision between fragments that are all present in order
  PARAPHRASE    >=60% of the quote's content tokens occur in order in one text window
  MISATTRIBUTED the passage exists but in a different work than the one named
  ABSENT        not found in any held text
  UNCHECKABLE   names a work that is not held (and not found elsewhere either)

Writes verification.json
"""
import hashlib
import json
import os
import re
import sys
import unicodedata

AR = re.compile(r'[\u0600-\u06FF]')
DIAC = re.compile(r'[\u064B-\u0652\u0670\u0640\u06D6-\u06ED]')

CORPUS_ROOTS = [
    ("corpus",      "/workspace/hf_v20_1_release/corpus"),
    ("heritage",    "/workspace/heritage_foundations"),
    ("scholastic",  "/workspace/scholastic_masters"),
    ("andalusian",  "/workspace/andalusian_canon_raw"),
]

# named worked -> acceptable corpus basenames
WORK_MAP = {
    "sibawayh": ["Sibawayh_Al_Kitab.txt"],
    "al-kitab": ["Sibawayh_Al_Kitab.txt"],
    "sībawayh": ["Sibawayh_Al_Kitab.txt"],
    "khalil": ["Al_Khalil_Al_Ayn.txt"],
    "al-khalil": ["Al_Khalil_Al_Ayn.txt"],
    "al-khalīl": ["Al_Khalil_Al_Ayn.txt"],
    "al-ayn": ["Al_Khalil_Al_Ayn.txt"],
    "khalil_ayn": ["Al_Khalil_Al_Ayn.txt"],
    "ibn jinni": ["Ibn_Jinni_Al_Khasais.txt", "Ibn_Jinni_Sirr_Sinat_Al_Irab.txt"],
    "ibn jinnī": ["Ibn_Jinni_Al_Khasais.txt", "Ibn_Jinni_Sirr_Sinat_Al_Irab.txt"],
    "khasais": ["Ibn_Jinni_Al_Khasais.txt"],
    "khasa'is": ["Ibn_Jinni_Al_Khasais.txt"],
    "sirr": ["Ibn_Jinni_Sirr_Sinat_Al_Irab.txt"],
    "zajjaji": ["Al_Zajjaji_Huruf_Al_Maani.txt", "Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt"],
    "zajjājī": ["Al_Zajjaji_Huruf_Al_Maani.txt", "Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt"],
    "mubarrad": ["Al_Mubarrad_Al_Muqtadab.txt"],
    "ibn faris": ["Ibn_Faris_Mujam_Maqayis_Al_Lughah.txt", "Ibn_Faris_Al_Sahibi.txt"],
    "maqayis": ["Ibn_Faris_Mujam_Maqayis_Al_Lughah.txt"],
    "sahibi": ["Ibn_Faris_Al_Sahibi.txt"],
    "jawhari": ["Al_Jawhari_Al_Sihah.txt"],
    "sihah": ["Al_Jawhari_Al_Sihah.txt"],
    "azhari": ["Al_Azhari_Tahdhib_Al_Lughah.txt"],
    "tahdhib": ["Al_Azhari_Tahdhib_Al_Lughah.txt"],
    "ibn durayd": ["Ibn_Durayd_Jamharat_Al_Lughah.txt", "Ibn_Durayd_Kitab_Al_Ishtiqaq.txt"],
    "jamharat": ["Ibn_Durayd_Jamharat_Al_Lughah.txt"],
    "ishtiqaq": ["Ibn_Durayd_Kitab_Al_Ishtiqaq.txt"],
    "akhfash": ["Al_Akhfash_Maani_Al_Quran.txt"],
    "zamakhshari": ["Zamakhshari_Asas_al_Balaghah.txt"],
    "asas": ["Zamakhshari_Asas_al_Balaghah.txt"],
    "raghib": ["Raghib_Al_Mufradat.txt"],
    "mufradat": ["Raghib_Al_Mufradat.txt"],
    "ibn sikkit": ["Ibn_Al_Sikkit_Islah_Al_Mantiq.txt"],
    "ibn malik": ["03_IbnMalik_Alfiyyah.txt", "04_IbnMalik_Tashil_al_Fawaid.txt",
                  "05_IbnMalik_Sharh_al_Kafiyah.txt", "06_IbnMalik_Lamiyyat_al_Afal.txt"],
    "ibn mālik": ["03_IbnMalik_Alfiyyah.txt", "04_IbnMalik_Tashil_al_Fawaid.txt",
                  "05_IbnMalik_Sharh_al_Kafiyah.txt", "06_IbnMalik_Lamiyyat_al_Afal.txt"],
    "alfiyyah": ["03_IbnMalik_Alfiyyah.txt"],
    "tashil": ["04_IbnMalik_Tashil_al_Fawaid.txt"],
    "tashīl": ["04_IbnMalik_Tashil_al_Fawaid.txt"],
    "sharh al-kafiyah": ["05_IbnMalik_Sharh_al_Kafiyah.txt"],
    "lamiyyat": ["06_IbnMalik_Lamiyyat_al_Afal.txt"],
    "abu hayyan": ["09_AbuHayyan_Irtishaf_al_Darab.txt", "10_AbuHayyan_Tadhyil_al_Tashil.txt"],
    "abū ḥayyān": ["09_AbuHayyan_Irtishaf_al_Darab.txt", "10_AbuHayyan_Tadhyil_al_Tashil.txt"],
    "irtishaf": ["09_AbuHayyan_Irtishaf_al_Darab.txt"],
    "tadhyil": ["10_AbuHayyan_Tadhyil_al_Tashil.txt"],
    "tadhyīl": ["10_AbuHayyan_Tadhyil_al_Tashil.txt"],
    "shatibi": ["12_Shatibi_Al_Muwafaqat.txt", "11_Shatibi_Sharh_Alfiyyah.txt"],
    "shāṭibī": ["12_Shatibi_Al_Muwafaqat.txt", "11_Shatibi_Sharh_Alfiyyah.txt"],
    "muwafaqat": ["12_Shatibi_Al_Muwafaqat.txt"],
    "mashaqqat": ["12_Shatibi_Al_Muwafaqat.txt"],
    "suhayli": ["02_Suhayli_Nataij_al_Fikr.txt"],
    "suhaylī": ["02_Suhayli_Nataij_al_Fikr.txt"],
    "ibn mada": ["01_IbnMada_Radd_ala_Nuhat.txt"],
    "ibn sida": ["08_IbnSayyidih_Al_Muhkam.txt", "07_IbnSayyidih_Al_Mukhassas.txt"],
    "ibn sayyidih": ["08_IbnSayyidih_Al_Muhkam.txt", "07_IbnSayyidih_Al_Mukhassas.txt"],
    "muhkam": ["08_IbnSayyidih_Al_Muhkam.txt"],
    "mukhassas": ["07_IbnSayyidih_Al_Mukhassas.txt"],
}

# works the project cites that we do NOT hold (verified by directory listing)
UNHELD = {
    "ibn usfur": "Ibn 'Usfur, Sharh Jumal al-Zajjaji / al-Mumti' fi al-Tasrif",
    "usfur": "Ibn 'Usfur, Sharh Jumal al-Zajjaji / al-Mumti' fi al-Tasrif",
    "usfūr": "Ibn 'Usfur, Sharh Jumal al-Zajjaji / al-Mumti' fi al-Tasrif",
    "mumti": "Ibn 'Usfur, al-Mumti' fi al-Tasrif",
    "jumal": "Ibn 'Usfur, Sharh Jumal al-Zajjaji",
    "ibn usfūr": "Ibn 'Usfur, Sharh Jumal al-Zajjaji / al-Mumti' fi al-Tasrif",
}


def norm1(s):
    """diacritics + whitespace only"""
    s = DIAC.sub('', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def norm2(s):
    """orthography + punctuation insensitive"""
    s = DIAC.sub('', s)
    s = (s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ٱ', 'ا')
          .replace('ى', 'ي').replace('ة', 'ه').replace('ؤ', 'و').replace('ئ', 'ي'))
    s = re.sub(r'[^\u0600-\u06FF]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


ELIDE = re.compile(r'\s*(?:\.{3,}|…|#)\s*')


def fragments(q):
    parts = [p for p in ELIDE.split(q) if AR.search(p)]
    return [p.strip() for p in parts if norm2(p)]


def load_corpus(verbose=True):
    files = []          # (label, path, raw, norm1, norm2)
    by_base = {}
    seen_md5 = {}
    for label, root in CORPUS_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, names in os.walk(root):
            for n in sorted(names):
                if not n.endswith('.txt'):
                    continue
                p = os.path.join(dirpath, n)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if sz > 120 * 1024 * 1024:
                    continue
                with open(p, encoding='utf-8', errors='replace') as fh:
                    raw = fh.read()
                h = hashlib.md5(raw.encode('utf-8', 'replace')).hexdigest()
                by_base.setdefault(n, []).append(p)
                if h in seen_md5:
                    seen_md5[h]['aliases'].append(p)
                    continue
                rec = {'label': label, 'path': p, 'base': n, 'raw': raw,
                       'n1': norm1(raw), 'n2': norm2(raw), 'aliases': [p]}
                seen_md5[h] = rec
                files.append(rec)
    if verbose:
        print(f"corpus files (deduped): {len(files)}")
        tot = sum(len(f['raw']) for f in files)
        print(f"total chars: {tot/1e6:.1f}M")
    return files, by_base


def line_at(raw, pos):
    return raw.count('\n', 0, pos) + 1


def find_quote(quote, files):
    """Return list of match dicts, best first."""
    fr = fragments(quote)
    hits = []
    for f in files:
        raw, n1, n2 = f['raw'], f['n1'], f['n2']
        # strict
        if quote in raw:
            pos = raw.find(quote)
            hits.append({'cls': 'VERBATIM', 'file': f['path'], 'line': line_at(raw, pos),
                         'score': 1.0, 'level': 'exact'})
            continue
        # level 1 (diacritics) as a single run
        q1 = norm1(quote)
        if len(q1) > 8 and q1 in n1:
            pos = n1.find(q1)
            hits.append({'cls': 'NEAR_VERBATIM', 'file': f['path'],
                         'line': line_at(n1, pos), 'score': 0.99, 'level': 'diacritics'})
            continue
        # level 2 single run
        q2 = norm2(quote)
        if len(q2) > 8 and q2 in n2:
            pos = n2.find(q2)
            hits.append({'cls': 'NEAR_VERBATIM', 'file': f['path'],
                         'line': line_at(n2, pos), 'score': 0.97, 'level': 'orthography'})
            continue
        # elision: all fragments present, in order, inside a bounded span
        if len(fr) >= 2:
            cur = 0
            ok = True
            first = None
            for frag in fr:
                nf = norm2(frag)
                j = n2.find(nf, cur)
                if j < 0:
                    ok = False
                    break
                if first is None:
                    first = j
                cur = j + len(nf)
            if ok and cur - first <= 6000:
                hits.append({'cls': 'NEAR_VERBATIM', 'file': f['path'],
                             'line': line_at(n2, first), 'score': 0.9,
                             'level': f'elision({len(fr)} fragments)'})
                continue
        # paraphrase: token coverage in a window
        if len(fr) == 1:
            toks = norm2(quote).split()
            if len(toks) >= 4:
                needle = toks[:4]
                start = 0
                best = 0
                while True:
                    j = n2.find(' '.join(needle), start)
                    if j < 0:
                        break
                    win = n2[j:j + max(400, len(q2) * 3)]
                    wset = win.split()
                    found = 0
                    for t in toks:
                        if t in wset:
                            found += 1
                    best = max(best, found / len(toks))
                    start = j + 1
                    if best > 0.95:
                        break
                if best >= 0.60:
                    hits.append({'cls': 'PARAPHRASE', 'file': f['path'],
                                 'line': None, 'score': round(best, 3),
                                 'level': 'token-coverage'})
    hits.sort(key=lambda h: -h['score'])
    return hits


NAMED_KEYS = [
    ("sibawayh", "Sibawayh_Al_Kitab.txt"), ("sībawayh", "Sibawayh_Al_Kitab.txt"),
    ("al-kitab", "Sibawayh_Al_Kitab.txt"), ("al-kitāb", "Sibawayh_Al_Kitab.txt"),
    ("khalil", "Al_Khalil_Al_Ayn.txt"), ("khalīl", "Al_Khalil_Al_Ayn.txt"),
    ("ibn jinni", None), ("ibn jinnī", None), ("zajjaji", None), ("zajjājī", None),
    ("mubarrad", "Al_Mubarrad_Al_Muqtadab.txt"), ("ibn faris", None),
    ("jawhari", "Al_Jawhari_Al_Sihah.txt"), ("azhari", "Al_Azhari_Tahdib_Al_Lughah.txt"),
    ("ibn durayd", None), ("akhfash", "Al_Akhfash_Maani_Al_Quran.txt"),
    ("zamakhshari", "Zamakhshari_Asas_al_Balaghah.txt"),
    ("ibn malik", None), ("ibn mālik", None), ("abu hayyan", None),
    ("abū ḥayyān", None), ("shatibi", None), ("shāṭibī", None),
    ("suhayli", "02_Suhayli_Nataij_al_Fikr.txt"), ("ibn mada", "01_IbnMada_Radd_ala_Nuhat.txt"),
    ("usfur", None), ("usfūr", None), ("mumti", None), ("jumal", None),
]


def named_targets(text):
    """Which works does this citation text name?"""
    tl = text.lower()
    out = []
    for k, _ in NAMED_KEYS:
        if k in tl:
            out.append(k)
    return out


def acceptable_bases(books_named):
    acc = set()
    for b in books_named:
        for f in WORK_MAP.get(b, []):
            acc.add(f)
    return acc


def main():
    ext = sys.argv[1] if len(sys.argv) > 1 else '/workspace/grammar_audit/citations2.json'
    out = sys.argv[2] if len(sys.argv) > 2 else '/workspace/grammar_audit/verification.json'
    files, by_base = load_corpus()
    data = json.load(open(ext, encoding='utf-8'))

    results = []

    for r in data['structured']:
        quote = r['arabic']
        src = r.get('src_file')
        named = named_targets((r.get('book') or '') + ' ' + (src or ''))
        # locate the declared file
        declared_path = None
        declared_held = False
        base = os.path.basename(src) if src else None
        if base and base in by_base:
            declared_held = True
            declared_path = by_base[base][0]
        hits = find_quote(quote, files)
        hit = hits[0] if hits else None
        cls = None
        if hit is None:
            cls = 'UNCHECKABLE' if (src and not declared_held) else 'ABSENT'
        else:
            acc = acceptable_bases(named)
            if declared_held and os.path.basename(hit['file']) == base:
                cls = hit['cls']
            else:
                cls = 'MISATTRIBUTED'
        line_ok = None
        if hit and declared_path and os.path.basename(hit['file']) == base:
            if r.get('src_line') is not None:
                line_ok = (hit['line'] == r['src_line'])
        results.append({
            'module': r['module'], 'module_line': r['module_line'],
            'kind': 'structured', 'book': r.get('book'),
            'declared_file': src, 'declared_line': r.get('src_line'),
            'declared_held': declared_held, 'cls': cls,
            'found_file': hit['file'] if hit else None,
            'found_line': hit['line'] if hit else None,
            'match_level': hit['level'] if hit else None,
            'score': hit['score'] if hit else 0.0,
            'line_matches_declared': line_ok,
            'arabic': quote,
            'alternates': hits[1:3],
        })

    for r in data['inline']:
        quote = r['arabic']
        named = named_targets((r.get('book') or '') + ' ' + (r.get('ctx') or ''))
        hits = find_quote(quote, files)
        hit = hits[0] if hits else None
        acc = acceptable_bases(named)
        if hit is None:
            unheld = [b for b in named if b in UNHELD]
            cls = 'UNCHECKABLE' if unheld else 'ABSENT'
        else:
            hb = os.path.basename(hit['file'])
            if acc and hb in acc:
                cls = hit['cls']
            elif hit['score'] < 0.85:
                cls = 'PARAPHRASE'
            else:
                cls = 'MISATTRIBUTED'
        results.append({
            'module': r['module'], 'module_line': r['module_line'],
            'kind': r['kind'], 'book': r.get('book'),
            'declared_file': None, 'declared_line': None,
            'declared_held': None, 'cls': cls,
            'found_file': hit['file'] if hit else None,
            'found_line': hit['line'] if hit else None,
            'match_level': hit['level'] if hit else None,
            'score': hit['score'] if hit else 0.0,
            'line_matches_declared': None,
            'n_words': r.get('n_words'),
            'arabic': quote,
            'alternates': hits[1:3],
        })

    json.dump({'results': results,
               'corpus_files': [f['path'] for f in files]},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    from collections import Counter
    st = [x for x in results if x['kind'] == 'structured']
    il = [x for x in results if x['kind'] != 'structured']
    print("=== STRUCTURED (n=%d) ===" % len(st))
    print(Counter(x['cls'] for x in st))
    print("declared file held:", sum(1 for x in st if x['declared_held']), "/", len(st))
    print("line pointer correct:", sum(1 for x in st if x['line_matches_declared'] is True),
          "wrong:", sum(1 for x in st if x['line_matches_declared'] is False),
          "unknown:", sum(1 for x in st if x['line_matches_declared'] is None))
    print()
    print("=== INLINE (n=%d) ===" % len(il))
    print(Counter(x['cls'] for x in il))
    print()
    print("=== ALL ===")
    print(Counter(x['cls'] for x in results))


if __name__ == '__main__':
    main()
