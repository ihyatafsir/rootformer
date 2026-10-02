#!/usr/bin/env python3
"""Mechanically verify every extracted citation against the held primary texts. v2

Fixes v1: normalization preserves newlines so reported line numbers are real;
the declared source file is preferred before any global search.

Classes: VERBATIM / NEAR_VERBATIM / PARAPHRASE / MISATTRIBUTED / ABSENT / UNCHECKABLE
"""
import hashlib
import json
import os
import re
import sys

AR = re.compile(r'[\u0600-\u06FF]')
DIAC = re.compile(r'[\u064B-\u0652\u0670\u0640\u06D6-\u06ED]')

CORPUS_ROOTS = [
    ("corpus",      "/workspace/hf_v20_1_release/corpus"),
    ("heritage",    "/workspace/heritage_foundations"),
    ("scholastic",  "/workspace/scholastic_masters"),
    ("andalusian",  "/workspace/andalusian_canon_raw"),
]

WORK_MAP = {
    "sibawayh": ["Sibawayh_Al_Kitab.txt"], "sībawayh": ["Sibawayh_Al_Kitab.txt"],
    "al-kitab": ["Sibawayh_Al_Kitab.txt"], "al-kitāb": ["Sibawayh_Al_Kitab.txt"],
    "khalil": ["Al_Khalil_Al_Ayn.txt"], "khalīl": ["Al_Khalil_Al_Ayn.txt"],
    "al-khalil": ["Al_Khalil_Al_Ayn.txt"], "al-khalīl": ["Al_Khalil_Al_Ayn.txt"],
    "al-ayn": ["Al_Khalil_Al_Ayn.txt"], "al-'ayn": ["Al_Khalil_Al_Ayn.txt"],
    "ibn jinni": ["Ibn_Jinni_Al_Khasais.txt", "Ibn_Jinni_Sirr_Sinat_Al_Irab.txt"],
    "ibn jinnī": ["Ibn_Jinni_Al_Khasais.txt", "Ibn_Jinni_Sirr_Sinat_Al_Irab.txt"],
    "khasais": ["Ibn_Jinni_Al_Khasais.txt"], "khasa'is": ["Ibn_Jinni_Al_Khasais.txt"],
    "sirr": ["Ibn_Jinni_Sirr_Sinat_Al_Irab.txt"],
    "zajjaji": ["Al_Zajjaji_Huruf_Al_Maani.txt", "Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt"],
    "zajjājī": ["Al_Zajjaji_Huruf_Al_Maani.txt", "Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt"],
    "mubarrad": ["Al_Mubarrad_Al_Muqtadab.txt"],
    "ibn faris": ["Ibn_Faris_Mujam_Maqayis_Al_Lughah.txt", "Ibn_Faris_Al_Sahibi.txt"],
    "maqayis": ["Ibn_Faris_Mujam_Maqayis_Al_Lughah.txt"],
    "sahibi": ["Ibn_Faris_Al_Sahibi.txt"],
    "jawhari": ["Al_Jawhari_Al_Sihah.txt"], "sihah": ["Al_Jawhari_Al_Sihah.txt"],
    "azhari": ["Al_Azhari_Tahdhib_Al_Lughah.txt"], "tahdhib": ["Al_Azhari_Tahdhib_Al_Lughah.txt"],
    "ibn durayd": ["Ibn_Durayd_Jamharat_Al_Lughah.txt", "Ibn_Durayd_Kitab_Al_Ishtiqaq.txt"],
    "jamharat": ["Ibn_Durayd_Jamharat_Al_Lughah.txt"],
    "ishtiqaq": ["Ibn_Durayd_Kitab_Al_Ishtiqaq.txt"],
    "akhfash": ["Al_Akhfash_Maani_Al_Quran.txt"],
    "zamakhshari": ["Zamakhshari_Asas_al_Balaghah.txt"],
    "asas": ["Zamakhshari_Asas_al_Balaghah.txt"],
    "raghib": ["Raghib_Al_Mufradat.txt"], "mufradat": ["Raghib_Al_Mufradat.txt"],
    "ibn sikkit": ["Ibn_Al_Sikkit_Islah_Al_Mantiq.txt"],
    "ibn malik": ["03_IbnMalik_Alfiyyah.txt", "04_IbnMalik_Tashil_al_Fawaid.txt",
                  "05_IbnMalik_Sharh_al_Kafiyah.txt", "06_IbnMalik_Lamiyyat_al_Afal.txt"],
    "ibn mālik": ["03_IbnMalik_Alfiyyah.txt", "04_IbnMalik_Tashil_al_Fawaid.txt",
                  "05_IbnMalik_Sharh_al_Kafiyah.txt", "06_IbnMalik_Lamiyyat_al_Afal.txt"],
    "alfiyyah": ["03_IbnMalik_Alfiyyah.txt"],
    "tashil": ["04_IbnMalik_Tashil_al_Fawaid.txt"], "tashīl": ["04_IbnMalik_Tashil_al_Fawaid.txt"],
    "sharh al-kafiyah": ["05_IbnMalik_Sharh_al_Kafiyah.txt"],
    "lamiyyat": ["06_IbnMalik_Lamiyyat_al_Afal.txt"],
    "abu hayyan": ["09_AbuHayyan_Irtishaf_al_Darab.txt", "10_AbuHayyan_Tadhyil_al_Tashil.txt"],
    "abū ḥayyān": ["09_AbuHayyan_Irtishaf_al_Darab.txt", "10_AbuHayyan_Tadhyil_al_Tashil.txt"],
    "irtishaf": ["09_AbuHayyan_Irtishaf_al_Darab.txt"],
    "tadhyil": ["10_AbuHayyan_Tadhyil_al_Tashil.txt"], "tadhyīl": ["10_AbuHayyan_Tadhyil_al_Tashil.txt"],
    "shatibi": ["12_Shatibi_Al_Muwafaqat.txt", "11_Shatibi_Sharh_Alfiyyah.txt"],
    "shāṭibī": ["12_Shatibi_Al_Muwafaqat.txt", "11_Shatibi_Sharh_Alfiyyah.txt"],
    "muwafaqat": ["12_Shatibi_Al_Muwafaqat.txt"],
    "suhayli": ["02_Suhayli_Nataij_al_Fikr.txt"], "suhaylī": ["02_Suhayli_Nataij_al_Fikr.txt"],
    "ibn mada": ["01_IbnMada_Radd_ala_Nuhat.txt"],
    "ibn sida": ["08_IbnSayyidih_Al_Muhkam.txt", "07_IbnSayyidih_Al_Mukhassas.txt"],
    "ibn sayyidih": ["08_IbnSayyidih_Al_Muhkam.txt", "07_IbnSayyidih_Al_Mukhassas.txt"],
    "muhkam": ["08_IbnSayyidih_Al_Muhkam.txt"],
    "mukhassas": ["07_IbnSayyidih_Al_Mukhassas.txt"],
}

UNHELD = {
    "usfur": "Ibn 'Usfur: Sharh Jumal al-Zajjaji and al-Mumti' fi al-Tasrif",
    "usfūr": "Ibn 'Usfur: Sharh Jumal al-Zajjaji and al-Mumti' fi al-Tasrif",
    "ibn 'usfur": "Ibn 'Usfur: Sharh Jumal al-Zajjaji and al-Mumti' fi al-Tasrif",
    "mumti": "Ibn 'Usfur, al-Mumti' fi al-Tasrif",
    "jumal": "Ibn 'Usfur, Sharh Jumal al-Zajjaji",
}

NAMED_KEYS = ["sibawayh", "sībawayh", "al-kitab", "al-kitāb", "khalil", "khalīl",
              "al-khalil", "al-khalīl", "al-ayn", "ibn jinni", "ibn jinnī", "zajjaji",
              "zajjājī", "mubarrad", "ibn faris", "jawhari", "azhari", "ibn durayd",
              "akhfash", "zamakhshari", "ibn malik", "ibn mālik", "abu hayyan",
              "abū ḥayyān", "shatibi", "shāṭibī", "suhayli", "ibn mada", "usfur",
              "usfūr", "mumti", "jumal", "alfiyyah", "tashil", "tashīl", "lamiyyat",
              "irtishaf", "tadhyil", "tadhyīl", "muwafaqat", "khasais", "sirr",
              "ishtiqaq", "maqayis", "sahibi", "sihah", "muhkam", "mukhassas"]


def norm1(s):
    s = DIAC.sub('', s)
    s = re.sub(r'[^\S\n]+', ' ', s)
    return s


def norm2(s):
    s = DIAC.sub('', s)
    s = (s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ٱ', 'ا')
          .replace('ى', 'ي').replace('ة', 'ه').replace('ؤ', 'و').replace('ئ', 'ي'))
    s = re.sub(r'[^\u0600-\u06FF\n]+', ' ', s)
    s = re.sub(r'[^\S\n]+', ' ', s)
    return s


ELIDE = re.compile(r'\s*(?:\.{3,}|…|#)\s*')


def fragments(q):
    return [p.strip() for p in ELIDE.split(q) if AR.search(p) and norm2(p).strip()]


def load_corpus():
    files, by_base, seen = [], {}, {}
    for label, root in CORPUS_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, names in os.walk(root):
            for n in sorted(names):
                if not n.endswith('.txt'):
                    continue
                p = os.path.join(dirpath, n)
                try:
                    if os.path.getsize(p) > 120 * 1024 * 1024:
                        continue
                    raw = open(p, encoding='utf-8', errors='replace').read()
                except OSError:
                    continue
                by_base.setdefault(n, []).append(p)
                h = hashlib.md5(raw.encode('utf-8', 'replace')).hexdigest()
                if h in seen:
                    seen[h]['aliases'].append(p)
                    continue
                n1 = norm1(raw)
                n2 = norm2(raw)
                # n1/n2 keep newlines (for line reporting); n1f/n2f are line-joined
                # (the citations join the source's wrapped lines with markup removed).
                rec = {'path': p, 'base': n, 'raw': raw,
                       'n1': n1, 'n2': n2,
                       'n1f': n1.replace('\n', ' '), 'n2f': n2.replace('\n', ' '),
                       'lines': raw.split('\n'), 'aliases': [p]}
                seen[h] = rec
                files.append(rec)
    print(f"corpus files (deduped): {len(files)}  chars={sum(len(f['raw']) for f in files)/1e6:.1f}M")
    return files, by_base


def line_at(s, pos):
    return s.count('\n', 0, pos) + 1


def match_in(f, quote, fr, q2):
    raw, n1, n2 = f['raw'], f['n1'], f['n2']
    n1f, n2f = f['n1f'], f['n2f']
    if quote in raw:
        pos = raw.find(quote)
        return {'cls': 'VERBATIM', 'file': f['path'], 'line': line_at(raw, pos),
                'score': 1.0, 'level': 'exact', 'pos': pos}
    q1 = norm1(quote).strip()
    if len(q1) > 8 and q1 in n1f:
        pos = n1f.find(q1)
        return {'cls': 'NEAR_VERBATIM', 'file': f['path'], 'line': line_at(n1, pos),
                'score': 0.99, 'level': 'diacritics', 'pos': pos}
    if len(q2) > 8 and q2.strip() in n2f:
        pos = n2f.find(q2.strip())
        return {'cls': 'NEAR_VERBATIM', 'file': f['path'], 'line': line_at(n2, pos),
                'score': 0.97, 'level': 'orthography', 'pos': pos}
    if len(fr) >= 2:
        cur, first, ok = 0, None, True
        for frag in fr:
            nf = norm2(frag).strip()
            j = n2f.find(nf, cur)
            if j < 0:
                ok = False
                break
            if first is None:
                first = j
            cur = j + len(nf)
        if ok and cur - first <= 8000:
            return {'cls': 'NEAR_VERBATIM', 'file': f['path'], 'line': line_at(n2, first),
                    'score': 0.9, 'level': f'elision({len(fr)})', 'pos': first}
    # paraphrase: token coverage near an anchor n-gram
    toks = norm2(quote).split()
    if len(toks) >= 4 and len(fr) == 1:
        best, bestpos = 0.0, None
        needles = []
        for s in range(0, max(1, len(toks) - 4), max(1, (len(toks) - 3) // 4)):
            needles.append(' '.join(toks[s:s + 4]))
        needles = needles[:5]
        for nd in needles:
            start = 0
            while True:
                j = n2f.find(nd, start)
                if j < 0:
                    break
                win = n2f[j:j + max(500, len(q2) * 3)].split()
                wset = set(win)
                cov = sum(1 for t in toks if t in wset) / len(toks)
                if cov > best:
                    best, bestpos = cov, j
                start = j + 1
                if best > 0.97:
                    break
            if best > 0.97:
                break
        if best >= 0.60:
            return {'cls': 'PARAPHRASE', 'file': f['path'],
                    'line': line_at(n2, bestpos) if bestpos is not None else None,
                    'score': round(best, 3), 'level': 'token-coverage', 'pos': bestpos}
    return None


def find(quote, files, prefer_bases):
    fr = fragments(quote)
    q2 = norm2(quote)
    preferred, other = [], []
    for f in files:
        h = match_in(f, quote, fr, q2)
        if h is None:
            continue
        (preferred if f['base'] in prefer_bases else other).append(h)
    preferred.sort(key=lambda h: -h['score'])
    other.sort(key=lambda h: -h['score'])
    return (preferred[0] if preferred else None), (other[0] if other else None)


def named(text):
    tl = (text or '').lower()
    return [k for k in NAMED_KEYS if k in tl]


def main():
    ext = sys.argv[1]
    out = sys.argv[2]
    files, by_base = load_corpus()
    data = json.load(open(ext, encoding='utf-8'))
    results = []

    for r in data['structured']:
        quote = r['arabic']
        src = r.get('src_file') or ''
        base = os.path.basename(src)
        declared_held = base in by_base
        prefer = {base} if base else set()
        if not declared_held:
            prefer = {b for k in named((r.get('book') or '') + ' ' + src)
                      for b in WORK_MAP.get(k, [])}
        ph, oh = find(quote, files, prefer)
        if ph is not None:
            cls = ph['cls']
            hit = ph
        elif oh is not None:
            cls = 'MISATTRIBUTED'
            hit = oh
        else:
            cls = 'ABSENT'
            hit = None
        dl = r.get('src_line')
        try:
            dl = int(str(dl).strip())
        except (TypeError, ValueError):
            dl = None
        line_ok = None
        if hit and declared_held and os.path.basename(hit['file']) == base and dl is not None:
            line_ok = (hit['line'] == dl)
        # record whether the quote's own text begins at/below the declared line
        near_line = None
        if hit and declared_held and os.path.basename(hit['file']) == base and dl is not None:
            near_line = (dl <= (hit['line'] or 0) <= dl + 8)
        results.append({
            'module': r['module'], 'module_line': r['module_line'], 'kind': 'structured',
            'book': r.get('book'), 'declared_file': src, 'declared_line': r.get('src_line'),
            'declared_held': declared_held, 'cls': cls,
            'found_file': hit['file'] if hit else None,
            'found_line': hit['line'] if hit else None,
            'match_level': hit['level'] if hit else None,
            'score': hit['score'] if hit else 0.0,
            'line_exact': line_ok, 'line_within_8': near_line,
            'arabic': quote,
        })

    for r in data['inline']:
        quote = r['arabic']
        ctx = (r.get('book') or '') + ' ' + (r.get('ctx') or '')
        prefer = {b for k in named(ctx) for b in WORK_MAP.get(k, [])}
        ph, oh = find(quote, files, prefer)
        unheld = [k for k in named(ctx) if k in UNHELD]
        if ph is not None:
            cls = ph['cls']
            hit = ph
        elif oh is not None:
            cls = oh['cls'] if oh['score'] >= 0.9 else 'PARAPHRASE'
            if oh['score'] >= 0.9:
                cls = 'MISATTRIBUTED'
            hit = oh
        else:
            cls = 'UNCHECKABLE' if unheld else 'ABSENT'
            hit = None
        results.append({
            'module': r['module'], 'module_line': r['module_line'], 'kind': r['kind'],
            'book': r.get('book'), 'declared_file': None, 'declared_line': None,
            'declared_held': None, 'cls': cls,
            'found_file': hit['file'] if hit else None,
            'found_line': hit['line'] if hit else None,
            'match_level': hit['level'] if hit else None,
            'score': hit['score'] if hit else 0.0,
            'line_exact': None, 'line_within_8': None,
            'n_words': r.get('n_words'),
            'unheld': unheld,
            'arabic': quote,
        })

    json.dump({'results': results, 'corpus_files': [f['path'] for f in files]},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    from collections import Counter
    st = [x for x in results if x['kind'] == 'structured']
    il = [x for x in results if x['kind'] != 'structured']
    print(f"\n=== STRUCTURED (n={len(st)}) ===")
    print(Counter(x['cls'] for x in st))
    print("declared file held:", sum(1 for x in st if x['declared_held']), "/", len(st))
    print("declared line == match line:", sum(1 for x in st if x['line_exact'] is True),
          "| within +8 lines:", sum(1 for x in st if x['line_within_8'] is True),
          "| outside:", sum(1 for x in st if x['line_exact'] is False and x['line_within_8'] is False))
    print(f"\n=== INLINE (n={len(il)}) ===")
    print(Counter(x['cls'] for x in il))
    print(f"\n=== ALL (n={len(results)}) ===")
    print(Counter(x['cls'] for x in results))


if __name__ == '__main__':
    main()
