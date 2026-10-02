#!/usr/bin/env python3
"""Mechanical citation verification v3.

Matching levels, strongest first:
  L0  quote is an exact substring of the raw corpus file
  L1  quote is an exact substring of the MARKUP-STRIPPED, line-joined file
      (markup = OpenITI '### ||', '#', '~~' continuations, 'msNNN', 'PageVxxPxxx').
      Nothing else is touched: diacritics and punctuation are preserved.
      -> this is what the project's own docstring claims, so L1 counts as VERBATIM.
  L2  same, diacritics/tatweel removed                       -> NEAR_VERBATIM
  L3  same, + orthographic normalisation (alef/hamza/ta-marbuta) -> NEAR_VERBATIM
  L4  '...' / '#'  elision: every fragment present, in order  -> NEAR_VERBATIM
  L5  >=60% of content tokens present in one window           -> PARAPHRASE
"""
import bisect
import hashlib
import json
import os
import re
import sys

AR = re.compile(r'[\u0600-\u06FF]')
DIAC = re.compile(r'[\u064B-\u0652\u0670\u0640\u06D6-\u06ED]')

CORPUS_ROOTS = [
    ("corpus",     "/workspace/hf_v20_1_release/corpus"),
    ("heritage",   "/workspace/heritage_foundations"),
    ("scholastic", "/workspace/scholastic_masters"),
    ("andalusian", "/workspace/andalusian_canon_raw"),
]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from verify_citations_v2 import WORK_MAP, UNHELD, NAMED_KEYS  # reuse tables
except Exception:
    WORK_MAP, UNHELD, NAMED_KEYS = {}, {}, []


# ---------------------------------------------------------------- normalizers
def strip_markup(line):
    if line.startswith('#META#'):
        return ''
    s = line
    s = re.sub(r'^\s*#+\s*', '', s)
    s = s.replace('||', ' ')
    s = s.replace('~~', ' ')
    s = re.sub(r'\bms\d+\b', ' ', s)
    s = re.sub(r'\bPageV\d+P\d+\b', ' ', s)
    s = re.sub(r'\bPageV\d+\b', ' ', s)
    s = re.sub(r'\bPage\s*\d+\b', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _orth(s):
    return (s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ٱ', 'ا')
             .replace('ى', 'ي').replace('ة', 'ه').replace('ؤ', 'و').replace('ئ', 'ي'))


def lvl_diac(s):
    return DIAC.sub('', s)


def lvl_orth(s):
    s = lvl_diac(s)
    s = _orth(s)
    s = re.sub(r'[^\u0600-\u06FF]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def build_flat(raw, fn):
    """Return (flat, starts) where starts[i] = (offset_in_flat, source_line_1based)."""
    parts, starts, pos = [], [], 0
    for i, line in enumerate(raw.split('\n'), 1):
        t = fn(line)
        if not t:
            continue
        if parts:
            pos += 1                      # the joining space
        starts.append((pos, i))
        parts.append(t)
        pos += len(t)
    return ' '.join(parts), starts


def line_of(starts, p):
    if not starts:
        return None
    offs = [s[0] for s in starts]
    j = bisect.bisect_right(offs, p) - 1
    if j < 0:
        return starts[0][1]
    return starts[j][1]


# ---------------------------------------------------------------- corpus
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
                m1, s1 = build_flat(raw, strip_markup)
                m2, s2 = build_flat(raw, lambda L: lvl_diac(strip_markup(L)))
                m3, s3 = build_flat(raw, lambda L: lvl_orth(strip_markup(L)))
                rec = {'path': p, 'base': n, 'raw': raw,
                       'm1': m1, 's1': s1, 'm2': m2, 's2': s2, 'm3': m3, 's3': s3,
                       'raw_lines': raw.split('\n'), 'aliases': [p]}
                seen[h] = rec
                files.append(rec)
    print(f"corpus files (deduped): {len(files)}  chars={sum(len(f['raw']) for f in files)/1e6:.1f}M")
    return files, by_base


ELIDE = re.compile(r'\s*(?:\.{3,}|…|#)\s*')


def fragments(q):
    return [p.strip() for p in ELIDE.split(q) if AR.search(p) and lvl_orth(p)]


def m_orth(s):
    return lvl_orth(s)


def match_in(f, quote):
    raw = f['raw']
    if quote in raw:
        pos = raw.find(quote)
        return dict(cls='VERBATIM', level='L0 raw', file=f['path'],
                    line=raw.count('\n', 0, pos) + 1, score=1.0)
    q1 = strip_markup(quote).strip()
    if len(q1) > 8 and q1 in f['m1']:
        pos = f['m1'].find(q1)
        return dict(cls='VERBATIM', level='L1 markup-stripped', file=f['path'],
                    line=line_of(f['s1'], pos), score=1.0, pos=pos)
    q2 = lvl_diac(q1)
    if len(q2) > 8 and q2 in f['m2']:
        pos = f['m2'].find(q2)
        return dict(cls='NEAR_VERBATIM', level='L2 diacritics', file=f['path'],
                    line=line_of(f['s2'], pos), score=0.99, pos=pos)
    q3 = m_orth(quote)
    if len(q3) > 8 and q3 in f['m3']:
        pos = f['m3'].find(q3)
        return dict(cls='NEAR_VERBATIM', level='L3 orthography', file=f['path'],
                    line=line_of(f['s3'], pos), score=0.97, pos=pos)
    fr = fragments(quote)
    if len(fr) >= 2:
        cur, first, ok = 0, None, True
        for frag in fr:
            nf = m_orth(frag)
            j = f['m3'].find(nf, cur)
            if j < 0:
                ok = False
                break
            if first is None:
                first = j
            cur = j + len(nf)
        if ok and cur - first <= 8000:
            return dict(cls='NEAR_VERBATIM', level=f'L4 elision({len(fr)})', file=f['path'],
                        line=line_of(f['s3'], first), score=0.9, pos=first)
    toks = q3.split()
    # L4b: greedy IN-ORDER token match.  Catches quotes the modules weld together
    # across OpenITI markup / '#' comment artefacts, where no contiguous substring
    # exists but the source words are all present in sequence.
    if len(toks) >= 6:
        cur, nh, first = 0, 0, None
        for t in toks:
            j = f['m3'].find(t, cur)
            if j >= 0:
                nh += 1
                cur = j + len(t)
                if first is None:
                    first = j
                if cur - first > 20000:
                    break
        cov = nh / len(toks)
        if cov >= 0.88:
            return dict(cls='NEAR_VERBATIM', level=f'L4b in-order({cov:.2f})',
                        file=f['path'], line=line_of(f['s3'], first) if first is not None else None,
                        score=0.86, pos=first)
    if len(toks) >= 4 and len(fr) == 1:
        best, bestpos = 0.0, None
        step = max(1, (len(toks) - 3) // 5)
        needles = [' '.join(toks[s:s + 4]) for s in range(0, max(1, len(toks) - 3), step)][:6]
        for nd in needles:
            start = 0
            while True:
                j = f['m3'].find(nd, start)
                if j < 0:
                    break
                win = set(f['m3'][j:j + max(500, len(q3) * 3)].split())
                cov = sum(1 for t in toks if t in win) / len(toks)
                if cov > best:
                    best, bestpos = cov, j
                start = j + 1
                if best > 0.97:
                    break
            if best > 0.97:
                break
        if best >= 0.60:
            return dict(cls='PARAPHRASE', level='L5 token-coverage', file=f['path'],
                        line=line_of(f['s3'], bestpos), score=round(best, 3), pos=bestpos)
    return None


def named(text):
    tl = (text or '').lower()
    return [k for k in NAMED_KEYS if k in tl]


def find(quote, files, prefer_bases):
    preferred, other = None, None
    for f in files:
        h = match_in(f, quote)
        if h is None:
            continue
        if f['base'] in prefer_bases:
            if preferred is None or h['score'] > preferred['score']:
                preferred = h
        else:
            if other is None or h['score'] > other['score']:
                other = h
    return preferred, other


def snippet(files, path, line, n=2):
    for f in files:
        if f['path'] == path and line:
            ls = f['raw_lines']
            a = max(0, line - 1)
            return ' '.join(x.strip() for x in ls[a:a + n])[:400]
    return None


def main():
    ext, out = sys.argv[1], sys.argv[2]
    files, by_base = load_corpus()
    data = json.load(open(ext, encoding='utf-8'))
    results = []

    for r in data['structured']:
        quote = r['arabic']
        src = r.get('src_file') or ''
        base = os.path.basename(src)
        declared_held = base in by_base
        prefer = {base} if declared_held else {
            b for k in named((r.get('book') or '') + ' ' + src) for b in WORK_MAP.get(k, [])}
        ph, oh = find(quote, files, prefer)
        if ph is not None and declared_held:
            cls, hit = ph['cls'], ph
        elif oh is not None and oh['score'] >= 0.9:
            cls, hit = 'MISATTRIBUTED', oh
        elif ph is not None:
            cls, hit = ph['cls'], ph
        elif oh is not None:
            cls, hit = 'PARAPHRASE', oh
        else:
            cls, hit = ('ABSENT' if declared_held else 'UNCHECKABLE'), None
        dl = r.get('src_line')
        try:
            dl = int(str(dl).strip())
        except (TypeError, ValueError):
            dl = None
        results.append(dict(
            module=r['module'], module_line=r['module_line'], kind='structured',
            book=r.get('book'), declared_file=src, declared_line=dl,
            declared_held=declared_held, cls=cls,
            found_file=hit['file'] if hit else None, found_line=hit['line'] if hit else None,
            level=hit['level'] if hit else None, score=hit['score'] if hit else 0.0,
            line_exact=(hit['line'] == dl) if (hit and dl and declared_held
                                               and os.path.basename(hit['file']) == base) else None,
            snippet=snippet(files, hit['file'], hit['line']) if hit else None,
            arabic=quote))

    for r in data['inline']:
        quote = r['arabic']
        ctx = (r.get('book') or '') + ' ' + (r.get('ctx') or '')
        prefer = {b for k in named(ctx) for b in WORK_MAP.get(k, [])}
        unheld = [k for k in named(ctx) if k in UNHELD]
        ph, oh = find(quote, files, prefer)
        if ph is not None:
            cls, hit = ph['cls'], ph
        elif oh is not None and oh['score'] >= 0.9:
            cls, hit = 'MISATTRIBUTED', oh
        elif oh is not None and oh['score'] >= 0.6:
            cls, hit = 'PARAPHRASE', oh
        else:
            cls, hit = ('UNCHECKABLE' if unheld else 'ABSENT'), None
        results.append(dict(
            module=r['module'], module_line=r['module_line'], kind=r['kind'],
            book=r.get('book'), declared_file=None, declared_line=None, declared_held=None,
            cls=cls, found_file=hit['file'] if hit else None,
            found_line=hit['line'] if hit else None, level=hit['level'] if hit else None,
            score=hit['score'] if hit else 0.0, line_exact=None,
            n_words=r.get('n_words'), unheld=unheld,
            snippet=snippet(files, hit['file'], hit['line']) if hit else None,
            arabic=quote))

    json.dump({'results': results, 'corpus_files': [f['path'] for f in files]},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    from collections import Counter
    st = [x for x in results if x['kind'] == 'structured']
    il = [x for x in results if x['kind'] != 'structured']
    print(f"\n=== STRUCTURED (n={len(st)}) ===")
    print(Counter(x['cls'] for x in st))
    print("declared file held:", sum(1 for x in st if x['declared_held']), "/", len(st))
    print("declared line == found line:", sum(1 for x in st if x['line_exact'] is True),
          "| mismatch:", sum(1 for x in st if x['line_exact'] is False))
    print("levels:", Counter(x['level'] for x in st))
    print(f"\n=== INLINE (n={len(il)}) ===")
    print(Counter(x['cls'] for x in il))
    print(f"\n=== ALL (n={len(results)}) ===")
    print(Counter(x['cls'] for x in results))


if __name__ == '__main__':
    main()
