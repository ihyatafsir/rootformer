#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rootformer_analyzer_v2.py -- recover the ~30% of tokens the v12 morphological analyzer drops.

Design constraints
------------------
The v19.2 checkpoint has a FIXED root vocabulary (9114 entries). We therefore may only map a
word to a root that is already in `roots_set`; introducing new roots would invalidate the
checkpoint's root_embed/root_head. This module is a pure *fallback*: it is consulted only when
`PureArabicMorphemicTokenizerV12.decompose_arabic_word` returns no root, so it cannot regress
any word the original analyzer already handles.

Strategy
--------
1. Strip candidate prefixes and suffixes (longest-first, bounded combinations).
2. On the residual stem, try progressively looser reductions:
     exact -> drop weak radicals -> collapse gemination -> hamza normalisation
3. Additionally match against a WEAK-NORMALISED index of the blueprint roots, so that
   e.g. تعالى (root علو), معنى (root عني), قيل (root قول), صلى (root صلو) resolve.
4. Accept a normalised match only when it is unambiguous (exactly one root shares the key).

Self-test:  python rootformer_analyzer_v2.py --bench 20000
"""
import argparse
import collections
import glob
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

WEAK = 'اويى'
HAMZA_MAP = {'أ': 'ء', 'إ': 'ء', 'آ': 'ء', 'ؤ': 'ء', 'ئ': 'ء', 'ٱ': 'ا'}

# Only affixes that are genuinely separable orthographically. Verbal prefixes
# (ي/ت/ن/أ/س/است/مست...) are deliberately excluded: stripping them invents stems and
# produced demonstrably wrong roots (استوت -> ستي, أبهى -> بها).
PREFIXES = ['وال', 'فال', 'بال', 'كال', 'لل', 'ال', 'و', 'ف', 'ب', 'ل', 'ك']
SUFFIXES = ['هما', 'هم', 'هن', 'كما', 'كم', 'كن', 'ونا', 'ينا', 'تها', 'تهم', 'تنا',
            'ات', 'ون', 'ين', 'ان', 'وا', 'تم', 'تن', 'نا', 'ها', 'هم', 'ه', 'ك',
            'ي', 'ت', 'ة', 'ا', 'ه']


def _norm_hamza(s):
    return ''.join(HAMZA_MAP.get(c, c) for c in s)


def _canon(s):
    """Canonical radical shape: weak radicals and hamza collapse to placeholders."""
    s = _norm_hamza(s)
    out = []
    for c in s:
        out.append('W' if c in WEAK else c)
    return ''.join(out)


def build_root_index(roots_set):
    """canon-key -> set of real roots sharing that key."""
    idx = collections.defaultdict(set)
    for r in roots_set:
        if len(r) in (3, 4) and not r.startswith('<'):
            idx[_canon(r)].add(r)
    return idx


def _stem_candidates(stem):
    """Progressively looser reductions of a stem, most-specific first."""
    seen, out = set(), []

    def push(x):
        if x and x not in seen and 2 <= len(x) <= 5:
            seen.add(x)
            out.append(x)

    push(stem)
    push(_norm_hamza(stem))
    # collapse gemination (e.g. مدد -> مد)
    collapsed = re.sub(r'(.)\1', r'\1', stem)
    push(collapsed)
    push(_norm_hamza(collapsed))
    # drop weak radicals (all subsets of 1 or 2 weak letters)
    pos = [i for i, c in enumerate(stem) if c in WEAK]
    for i in pos:
        push(stem[:i] + stem[i + 1:])
    for a in range(len(pos)):
        for b in range(a + 1, len(pos)):
            i, j = pos[a], pos[b]
            push(stem[:i] + stem[i + 1:j] + stem[j + 1:])
    # doubled-letter expansion is not attempted (cannot invent radicals)
    return out


def _affix_splits(word, max_prefix=4, max_suffix=4):
    """Yield (prefix, stem, suffix), longest affixes first, bounded."""
    pre_cands = ['']
    for p in sorted(PREFIXES, key=len, reverse=True):
        if p and word.startswith(p) and len(p) <= max_prefix and word != p:
            pre_cands.append(p)
    suf_cands = ['']
    for s in sorted(SUFFIXES, key=len, reverse=True):
        if s and word.endswith(s) and len(s) <= max_suffix and word != s:
            suf_cands.append(s)
    seen = set()
    for p in sorted(set(pre_cands), key=len, reverse=True):
        for s in sorted(set(suf_cands), key=len, reverse=True):
            stem = word[len(p):len(word) - len(s) if s else len(word)]
            if stem and stem not in seen and 2 <= len(stem) <= 5:
                seen.add(stem)
                yield p, stem, s


def _reconstructs(word, prefix, root, suffix):
    """The recovered tuple must realise back to the surface word (weak-letter tolerant)."""
    rec = (prefix or '') + root + (suffix or '')
    if rec == word:
        return True
    return _norm_hamza(rec) == _norm_hamza(word)


def recover_root(word, roots_set, index, strict=True):
    """
    Returns (prefix, root, wazn, suffix, how) or None.
    Only ever returns a root already present in `roots_set`.
    With strict=True the tuple must reconstruct the surface word exactly.
    """
    for p, stem, s in _affix_splits(word):
        for cand in _stem_candidates(stem):
            if cand in roots_set and (not strict or _reconstructs(word, p, cand, s)):
                return p, cand, None, s, 'exact'
        key = _canon(stem)
        hit = index.get(key)
        if hit and len(hit) == 1:
            cand = next(iter(hit))
            if not strict or _reconstructs(word, p, cand, s):
                return p, cand, None, s, 'canon'
    return None


# --------------------------------------------------------------------------------------
# Integration shim
# --------------------------------------------------------------------------------------
def patch_vocab(vocab, verbose=False):
    """
    Wrap a FarahidianMorphemicVocab instance so encode_word falls back to recover_root()
    when the base analyzer yields no root. Returns the same object (mutated).
    """
    roots_set = set(vocab.base_tok.roots_set)
    index = build_root_index(roots_set)
    vocab._v2_index = index
    vocab._v2_roots_set = roots_set
    return vocab


def encode_word_v2(vocab, word):
    """Drop-in replacement for FarahidianMorphemicVocab.encode_word with recovery."""
    clean_w = word.strip()
    if not clean_w:
        return (vocab.NONE_PREFIX, vocab.UNK_ROOT, vocab.NONE_WAZN, vocab.NONE_SUFFIX)

    particle_tag = f'<P:{clean_w}>'
    if particle_tag in vocab.root2id:
        return (vocab.NONE_PREFIX, vocab.root2id[particle_tag], vocab.NONE_WAZN, vocab.NONE_SUFFIX)

    p, r, wz, s = vocab.base_tok.decompose_arabic_word(clean_w)
    if r:
        # mirror FarahidianMorphemicVocab.encode_word exactly, including the <P:...> remap,
        # otherwise words whose analyzed root coincides with a particle name regress to <UNK>
        p_id = vocab.NONE_PREFIX if not p else vocab.prefix2id.get(p, vocab.prefix2id['<UNK>'])
        w_id = vocab.NONE_WAZN if not wz else vocab.wazn2id.get(wz, vocab.wazn2id['<UNK>'])
        s_id = vocab.NONE_SUFFIX if not s else vocab.suffix2id.get(s, vocab.suffix2id['<UNK>'])
        particle_cand = f'<P:{r}>'
        if particle_cand in vocab.root2id:
            r_id = vocab.root2id[particle_cand]
        else:
            r_id = vocab.root2id.get(r, vocab.UNK_ROOT)
        return (p_id, r_id, w_id, s_id)

    # ---- fallback ----
    rec = recover_root(clean_w, vocab._v2_roots_set, vocab._v2_index)
    if rec:
        pre, root, _wazn, suf, _how = rec
        p_id = vocab.prefix2id.get(pre, vocab.NONE_PREFIX) if pre else vocab.NONE_PREFIX
        s_id = vocab.suffix2id.get(suf, vocab.NONE_SUFFIX) if suf else vocab.NONE_SUFFIX
        return (p_id, vocab.root2id.get(root, vocab.UNK_ROOT), vocab.NONE_WAZN, s_id)

    if clean_w in vocab.COMMON_PARTICLES:
        return (vocab.NONE_PREFIX, vocab.root2id.get(f'<P:{clean_w}>', vocab.root2id['<PARTICLE>']),
                vocab.NONE_WAZN, vocab.NONE_SUFFIX)
    return (vocab.NONE_PREFIX, vocab.UNK_ROOT, vocab.NONE_WAZN, vocab.NONE_SUFFIX)


def _load_vocab():
    import nrmp_vocab as nv
    cls = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    return cls(str(ROOT_DIR / 'data/rootformer_v12_arabic_blueprint.json'))


def bench(n_tokens=20000):
    vocab = _load_vocab()
    patch_vocab(vocab)
    words = []
    for f in (sorted(glob.glob('/workspace/andalusian_canon_sanitized/*.txt'))[:3]
              + sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))[:3]):
        txt = Path(f).read_text(encoding='utf-8', errors='ignore')
        for s in re.split(r'[\n.!?؟]+', txt)[:6000]:
            for w in s.split():
                w2 = re.sub(r'[^\u0600-\u06FF]', '', w)
                if len(w2) > 1:
                    words.append(w2)
                if len(words) >= n_tokens:
                    break
            if len(words) >= n_tokens:
                break
        if len(words) >= n_tokens:
            break

    PART, UNK = vocab.root2id['<PARTICLE>'], vocab.root2id['<UNK>']
    old_junk = new_junk = recovered = particle_hits = 0
    how = collections.Counter()
    samples = []
    strict_rec = loose_rec = consistent = fails = 0
    for w in words:
        o = vocab.encode_word(w)
        n = encode_word_v2(vocab, w)
        o_j = o[1] in (PART, UNK)
        n_j = n[1] in (PART, UNK)
        if o_j:  # only words the base analyzer actually dropped
            rec_s = recover_root(w, vocab._v2_roots_set, vocab._v2_index, strict=True)
            rec_l = recover_root(w, vocab._v2_roots_set, vocab._v2_index, strict=False)
            strict_rec += bool(rec_s)
            loose_rec += bool(rec_l)
            if rec_l and _reconstructs(w, rec_l[0], rec_l[1], rec_l[3]):
                consistent += 1
            fails += 1
        old_junk += o_j
        new_junk += n_j
        if o_j and not n_j:
            recovered += 1
            rid = n[1]
            rname = vocab.id2root.get(rid, '?')
            if n[1] == vocab.UNK_ROOT:
                how['unk'] += 1
            elif rname.startswith('<P:'):
                particle_hits += 1
                how['particle'] += 1
            else:
                how['radical'] += 1
                if len(samples) < 40:
                    samples.append((w, n[0], rname, n[3]))

    print(f'tokens                    : {len(words)}')
    print(f'junk roots BEFORE         : {old_junk} ({100*old_junk/len(words):.2f}%)')
    print(f'junk roots AFTER          : {new_junk} ({100*new_junk/len(words):.2f}%)')
    print(f'tokens recovered          : {recovered} ({100*recovered/len(words):.2f} pp)')
    print(f'  -> mapped to REAL root  : {how["radical"]}')
    print(f'  -> mapped to particle   : {how["particle"]}')
    print(f'  -> still <UNK>          : {how["unk"]}')
    print(f'words the base analyzer dropped: {fails}')
    print(f'  loose fallback hits on those  : {loose_rec}')
    print(f'  ...reconstructing exactly     : {consistent}'
          f' ({100*consistent/max(loose_rec,1):.1f}% precision proxy)')
    print(f'  STRICT hits on those          : {strict_rec}'
          f' ({100*strict_rec/max(fails,1):.1f}% of failures repaired)')
    print('\nsample recoveries (word -> prefix + ROOT + suffix):')
    for w, p, r, s in samples:
        print(f'  {w:<16} -> {p or "-":<5} {r:<8} {s or "-"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', type=int, default=0)
    args = ap.parse_args()
    if args.bench:
        bench(args.bench)
    else:
        ap.print_help()
