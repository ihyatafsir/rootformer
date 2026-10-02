#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_heldout.py -- the FIRST honest held-out evaluation of the NRMP morphemic tokenizer
(FarahidianMorphemicVocab) on real classical Arabic text.

WHY THIS FILE EXISTS
--------------------
Every previous verification of this tokenizer was either (a) a primary-source citation check or
(b) a self-test on data the tokenizer's own lexicon/frequency table had already seen.  The
al-'Ayn round-trip figure (70.63% on a 15,265-form sample) was never reproduced against a
held-out corpus.  This harness measures the tokenizer on corpora the frequency table never read.

HELD-OUT CORPORA (evaluated)
    scholastic_sunni_sanitized/       21 .txt   -- held out
    scholastic_falsafa_sanitized/     40 .txt   -- held out
    scholastic_masters/               17 .txt   -- held out (see CONTAMINATION NOTE below)
    chronological_mujtahid_corpus/    sanitized/*.jsonl ('arabic' field) -- held out

CONTAMINATED CORPORA (NEVER evaluated here; hard-asserted out)
    scholastic_sanitized/             kathra frequency table source
    andalusian_canon_sanitized/       kathra frequency table source
    heritage_foundations/             kathra frequency table source (incl. Al-Khalil_Al_Ayn.txt)
  The kathra table (data/kathra_counts.json) was built with exactly those three globs; its own
  meta.corpus_globs_or_paths records this.  Evaluating on them would be circular -- the same
  circularity that already forced one withdrawal in this project.  They are excluded by
  assertion, not by convention.

CONTAMINATION NOTE ON scholastic_masters/
  kathra's globs do NOT include scholastic_masters/, so it is formally held out.  BUT it is the
  same SET OF WORKS as the contaminated scholastic_sanitized/: 16 of its 17 files share a
  filename with a kathra source (only Hilyat_al_Awliya.txt is unique), and the bytes differ only
  by sanitisation.  This harness therefore also measures, and reports, the measured word-type and
  word-occurrence OVERLAP between scholastic_masters and the kathra table, so the reader can see
  how much of that corpora's mass the frequency table had effectively already seen.

METRICS (all need no hand annotation)
  1. round-trip rate        decode_word(encode_word(w)) == w          [headline]
     round-trip (skeleton)  decode_word(encode_word(w)) == clean(w)   (diacritics/tatweel removed,
                            i.e. the tokenizer's own clean_arabic contract; reported separately
                            because the exact-match headline is otherwise penalised for
                            vocalisation the tokenizer deliberately discards)
  2. UNK rate               any of the 4 slots is <UNK>; root-UNK reported separately
  3. morphemic coverage     real lexical root AND real wazn (not <PARTICLE>, <NONE>, <UNK>)
  4. root-attestation rate  of the real roots assigned, the share al-'Ayn attests
                            (data/khalil_attest_v4.json); muhmal (unused) share reported too --
                            a non-zero muhmal share is a bug signal
  5. the above three times: ROOTFORMER_VALIDATED_SEG = 0 (greedy), 1 (shipped default,
     closed-class only), 2 (root re-segmentation; needs the kathra table)

METHOD
  * one deterministic scan of every held-out file, capped at --per-file-cap word FORMS per file
    by uniform stride over that file's token stream (so a whole work is represented, not just its
    header).  The scan is cached so the three mode runs evaluate the identical sample.
  * the SAME token sample is scored in each mode, so mode deltas are paired and noise-free.
  * every metric is computed occurrence-weighted (token level) AND type-weighted (distinct forms).
  * per-file, per-corpus, per-author (filename prefix) and global scopes.

USAGE (this is the exact three-command invocation)
  cd /workspace
  ROOTFORMER_VALIDATED_SEG=0 /workspace/venvs/rootformer/bin/python eval_heldout.py
  ROOTFORMER_VALIDATED_SEG=1 /workspace/venvs/rootformer/bin/python eval_heldout.py
  ROOTFORMER_VALIDATED_SEG=2 /workspace/venvs/rootformer/bin/python eval_heldout.py
  /workspace/venvs/rootformer/bin/python eval_heldout.py --report

  Each run merges its mode into /workspace/eval_heldout_results.json (per-file results included).
  --report prints the three-way comparison table and the top failures, and writes
  /workspace/EVAL_HELDOUT_REPORT.md.

  The tokenizer is loaded exactly as the other scripts do: nrmp_vocab -> FarahidianMorphemicVocab
  -> .base_tok, blueprint data/rootformer_v12_arabic_blueprint.json.  No existing module is
  modified.  The mode is read from the environment before the vocab is constructed, which is when
  validated_segmentation latches it.

This file is CPU-only (single process) and does not touch the GPU.
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys
import time

RELEASE = '/workspace/hf_v19_2_release'
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)
if RELEASE + '/models' not in sys.path:
    sys.path.insert(0, RELEASE + '/models')

BLUEPRINT = RELEASE + '/data/rootformer_v12_arabic_blueprint.json'
ATTEST = RELEASE + '/data/khalil_attest_v4.json'
KATHRA = RELEASE + '/data/kathra_counts.json'
RESULTS = '/workspace/eval_heldout_results.json'
SAMPLE_CACHE = '/workspace/eval_heldout_sample.json'
REPORT_MD = '/workspace/EVAL_HELDOUT_REPORT.md'

# --------------------------------------------------------------------------------------------
# corpora
# --------------------------------------------------------------------------------------------
HELD_OUT = [
    ('scholastic_sunni_sanitized',
     '/workspace/scholastic_sunni_sanitized/*.txt', 'txt'),
    ('scholastic_falsafa_sanitized',
     '/workspace/scholastic_falsafa_sanitized/*.txt', 'txt'),
    ('scholastic_masters',
     '/workspace/scholastic_masters/*.txt', 'txt'),
    ('chronological_mujtahid_corpus',
     '/workspace/chronological_mujtahid_corpus/sanitized/*.jsonl', 'jsonl'),
]

# the kathra sources -- NEVER evaluated.  Kept as resolved prefixes for the assertion.
CONTAMINATED_DIRS = (
    '/workspace/scholastic_sanitized',
    '/workspace/andalusian_canon_sanitized',
    '/workspace/heritage_foundations',
)

# --------------------------------------------------------------------------------------------
# word extraction
#
# A token is a maximal run of Arabic word characters, then kept only if it contains at least one
# Arabic LETTER.  That drops OpenITI/Shamela markers (#META#, ######OpenITI#, ::, 010.BookURI),
# Latin, and bare numbers, while keeping real Arabic prose -- including the Arabic content of
# lines that begin with '#' (in the sanitised files '#' is a heading marker, not a discard).
#
# The character classes deliberately EXCLUDE Arabic punctuation U+060C..U+061F, the Arabic-Indic
# digits U+0660..0669, U+06D4, U+06DD..U+06E9, so those act as token boundaries instead of
# smuggling punctuation into a word (which decode_word could never reproduce).
WORD_CHARS = '\u0621-\u065f\u0670-\u06d3\u06d6-\u06ed\u06fa-\u06ff'
LETTER_CHARS = '\u0621-\u063a\u0641-\u064a\u0671-\u06d3\u06fa-\u06ff'
TOKEN_RE = re.compile('[' + WORD_CHARS + ']+')
LETTER_RE = re.compile('[' + LETTER_CHARS + ']')
DIAC_RE = re.compile('[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')


def strip_diac(s):
    return DIAC_RE.sub('', s)


def iter_tokens(path, kind):
    """Yield every Arabic word form in a file, in order."""
    if kind == 'txt':
        with open(path, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                for m in TOKEN_RE.finditer(line):
                    t = m.group(0)
                    if LETTER_RE.search(t):
                        yield t
    else:
        with open(path, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get('arabic') or ''
                for m in TOKEN_RE.finditer(text):
                    t = m.group(0)
                    if LETTER_RE.search(t):
                        yield t


def jsonl_author(path):
    try:
        with open(path, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                a = obj.get('author')
                if a:
                    return a
    except Exception:
        pass
    return None


def author_key(fname, corpus, full_author):
    """Author label from the filename.  'Ghazali_Ihya_Ulum_al_Din.txt' -> 'Ghazali';
    'chronological ... shafici_risala.jsonl' -> 'shafici'.  Filenames are the only author source
    the task guarantees, so they win; the jsonl 'author' field is carried alongside as metadata.
    Leading numeric ordering prefixes (01_IbnMada_...) are dropped."""
    stem = os.path.basename(fname).rsplit('.', 1)[0]
    stem = re.sub(r'^\d+_', '', stem)
    head = stem.split('_')[0]
    return head or stem


# --------------------------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------------------------
def scan(cap, force=False):
    if os.path.exists(SAMPLE_CACHE) and not force:
        with open(SAMPLE_CACHE, encoding='utf-8') as fh:
            cache = json.load(fh)
        print('[*] scan cache loaded: %d files, %d sampled forms (%s)'
              % (len(cache['files']), cache['meta']['sampled_forms'], SAMPLE_CACHE))
        return cache

    t0 = time.time()
    files = []
    for corpus, pattern, kind in HELD_OUT:
        paths = sorted(glob.glob(pattern))
        assert paths, 'no files matched %s' % pattern
        for p in paths:
            real = os.path.realpath(p)
            for bad in CONTAMINATED_DIRS:
                assert not real.startswith(os.path.realpath(bad) + os.sep), \
                    'CONTAMINATION: %s is under kathra source %s' % (p, bad)
            files.append((corpus, p, kind))

    print('[*] scanning %d held-out files, cap %d forms/file ...' % (len(files), cap))
    out, total_forms, total_types = [], 0, 0
    for i, (corpus, path, kind) in enumerate(files, 1):
        n = sum(1 for _ in iter_tokens(path, kind))
        step = max(1, -(-n // cap)) if n else 1
        if step == 1:
            toks = list(iter_tokens(path, kind))
        else:
            toks = [t for j, t in enumerate(iter_tokens(path, kind)) if j % step == 0]
        counts = collections.Counter(toks)
        del toks
        fa = jsonl_author(path) if kind == 'jsonl' else None
        rec = {
            'corpus': corpus,
            'file': os.path.basename(path),
            'path': path,
            'kind': kind,
            'author_key': author_key(path, corpus, fa),
            'author_full': fa,
            'n_tokens_in_file': n,
            'stride': step,
            'n_sampled': sum(counts.values()),
            'n_types': len(counts),
            'counts': dict(counts),
        }
        total_forms += rec['n_sampled']
        total_types += rec['n_types']
        out.append(rec)
        print('  [%2d/%d] %-42s n=%8d stride=%3d sampled=%6d types=%6d'
              % (i, len(files), rec['file'], n, step, rec['n_sampled'], rec['n_types']))

    cache = {
        'meta': {
            'built': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'per_file_cap': cap,
            'n_files': len(out),
            'sampled_forms': total_forms,
            'sampled_types_summed_over_files': total_types,
            'held_out_corpora': [c for c, _, _ in HELD_OUT],
            'contaminated_dirs_never_evaluated': list(CONTAMINATED_DIRS),
            'token_re': 'maximal [%s]+ containing >=1 [%s]' % (WORD_CHARS, LETTER_CHARS),
            'sampling': 'uniform stride over the file token stream, capped at per_file_cap',
            'elapsed_s': round(time.time() - t0, 1),
        },
        'files': out,
    }
    with open(SAMPLE_CACHE, 'w', encoding='utf-8') as fh:
        json.dump(cache, fh, ensure_ascii=False)
    print('[*] scan done: %d forms, %d file-type entries, %.1fs -> %s'
          % (total_forms, total_types, time.time() - t0, SAMPLE_CACHE))
    return cache


# --------------------------------------------------------------------------------------------
# attestation record
# --------------------------------------------------------------------------------------------
def load_attest():
    d = json.load(open(ATTEST, encoding='utf-8'))
    attested = {tuple(t) for t in d.get('attested', [])}
    unused = {tuple(t) for t in d.get('unused', [])}
    ambiguous = {tuple(t) for t in d.get('ambiguous', [])}
    return attested, unused, ambiguous, d.get('counts', {})


def root_tuple(r):
    return tuple(c for c in strip_diac(r) if LETTER_RE.match(c))


# --------------------------------------------------------------------------------------------
# per-word evaluation, cached
# --------------------------------------------------------------------------------------------
REAL_ROOT = 0        # a lexical root
ROOT_PARTICLE = 1    # <P:..> or <PARTICLE>
ROOT_UNK = 2
ROOT_NONE = 3        # <PAD>/<BOS>/<EOS>


def vocab_class():
    import nrmp_vocab as nv
    if hasattr(nv, 'FarahidianMorphemicVocab'):
        return nv.FarahidianMorphemicVocab
    return next(v for k, v in vars(nv).items()
                if isinstance(v, type) and 'MorphemicVocab' in k)


class Evaluator(object):
    """Scores word forms in one ROOTFORMER_VALIDATED_SEG mode."""

    def __init__(self):
        self.V = vocab_class()
        self.vocab = self.V(BLUEPRINT)
        v = self.vocab
        self.attested, self.unused, self.ambiguous, self.attest_counts = load_attest()
        self.cache = {}
        self.detail = {}            # word -> (decoded, root, wazn, suffix) for every scored word
        self.attest_hits = 0        # type-level diagnostics only (reporting uses Acc)
        self.muhmal_hits = 0
        self.ambiguous_hits = 0
        self.unrecorded = 0
        self.real_root_tokens = 0

    # -- one word -> packed ints -------------------------------------------------------------
    def score(self, w):
        c = self.cache.get(w)
        if c is not None:
            return c
        v = self.vocab
        pid, rid, wid, sid = v.encode_word(w)
        dec = v.decode_word(pid, rid, wid, sid)
        clean = DIAC_RE.sub('', w)

        exact = 1 if dec == w else 0
        skel = 1 if dec == clean else 0

        rstr = v.id2root.get(rid, '<UNK>')
        if rstr.startswith('<P:') or rstr == '<PARTICLE>':
            rclass = ROOT_PARTICLE
        elif rstr in ('<UNK>',):
            rclass = ROOT_UNK
        elif rstr in ('<PAD>', '<BOS>', '<EOS>'):
            rclass = ROOT_NONE
        else:
            rclass = REAL_ROOT

        unk_any = 1 if (rid == v.UNK_ROOT
                        or wid == v.wazn2id['<UNK>']
                        or pid == v.prefix2id['<UNK>']
                        or sid == v.suffix2id['<UNK>']) else 0
        unk_root = 1 if rclass == ROOT_UNK else 0
        real_wazn = 1 if wid not in (v.NONE_WAZN, v.wazn2id['<UNK>'], v.PAD_WAZN) else 0
        morphemic = 1 if (rclass == REAL_ROOT and real_wazn) else 0
        particle = 1 if rclass == ROOT_PARTICLE else 0
        none_wazn = 1 if wid == v.NONE_WAZN else 0

        # attestation, only meaningful for a real root
        att = None
        if rclass == REAL_ROOT:
            rt = root_tuple(rstr)
            if rt in self.attested:
                att = 1
            elif rt in self.unused:
                att = 2
            elif rt in self.ambiguous:
                att = 3
            else:
                att = 4      # not in al-'Ayn's record at all (record is partial)

        c = (exact, skel, unk_any, unk_root, morphemic, particle, none_wazn, rclass, att)
        self.cache[w] = c
        self.detail[w] = (dec, rstr, v.id2wazn.get(wid), v.id2suffix.get(sid))
        if att == 1:
            self.attest_hits += 1
        elif att == 2:
            self.muhmal_hits += 1
        elif att == 3:
            self.ambiguous_hits += 1
        elif att == 4:
            self.unrecorded += 1
        if rclass == REAL_ROOT:
            self.real_root_tokens += 1
        return c


# --------------------------------------------------------------------------------------------
# metric accumulation
# --------------------------------------------------------------------------------------------
MKEYS = ('forms', 'types', 'rt', 'rt_skel', 'unk_any', 'unk_root', 'morphemic',
         'particle', 'none_wazn', 'root_forms', 'root_types',
         'att_ok', 'att_muhmal', 'att_ambiguous', 'att_unrecorded', 'diacritic_forms')


class Acc(object):
    __slots__ = MKEYS

    def __init__(self):
        for k in MKEYS:
            setattr(self, k, 0)

    def add(self, c, cnt):
        (exact, skel, unk_any, unk_root, morphemic, particle, none_wazn, rclass, att) = c
        self.forms += cnt
        self.types += 1
        self.rt += exact * cnt
        self.rt_skel += skel * cnt
        self.unk_any += unk_any * cnt
        self.unk_root += unk_root * cnt
        self.morphemic += morphemic * cnt
        self.particle += particle * cnt
        self.none_wazn += none_wazn * cnt
        if rclass == REAL_ROOT:
            self.root_forms += cnt
            self.root_types += 1
            if att == 1:
                self.att_ok += cnt
            elif att == 2:
                self.att_muhmal += cnt
            elif att == 3:
                self.att_ambiguous += cnt
            else:
                self.att_unrecorded += cnt

    def to_dict(self):
        f = max(self.forms, 1)
        t = max(self.types, 1)
        rf = max(self.root_forms, 1)
        return {
            'forms': self.forms,
            'types': self.types,
            'rt_forms': self.rt, 'rt_rate': round(100.0 * self.rt / f, 4),
            # rt_type_rate / *_type_rate are supplied by TypeAcc via .update() at every call site
            'rt_skel_forms': self.rt_skel, 'rt_skel_rate': round(100.0 * self.rt_skel / f, 4),
            'unk_any_forms': self.unk_any, 'unk_rate': round(100.0 * self.unk_any / f, 4),
            'unk_root_forms': self.unk_root, 'unk_root_rate': round(100.0 * self.unk_root / f, 4),
            'morphemic_forms': self.morphemic,
            'morphemic_rate': round(100.0 * self.morphemic / f, 4),
            'particle_forms': self.particle,
            'particle_rate': round(100.0 * self.particle / f, 4),
            'none_wazn_forms': self.none_wazn,
            'none_wazn_rate': round(100.0 * self.none_wazn / f, 4),
            'root_forms': self.root_forms, 'root_types': self.root_types,
            'att_ok_forms': self.att_ok,
            'attest_rate': round(100.0 * self.att_ok / rf, 4),
            'att_muhmal_forms': self.att_muhmal,
            'muhmal_rate': round(100.0 * self.att_muhmal / rf, 4),
            'att_ambiguous_forms': self.att_ambiguous,
            'ambiguous_rate': round(100.0 * self.att_ambiguous / rf, 4),
            'att_unrecorded_forms': self.att_unrecorded,
            'unrecorded_rate': round(100.0 * self.att_unrecorded / rf, 4),
        }


class TypeAcc(object):
    """type-level (distinct-form) accumulation, deduplicated across the whole sample."""
    __slots__ = ('seen', 'rt', 'rt_skel', 'unk_any', 'unk_root', 'morphemic', 'n')

    def __init__(self):
        self.seen = set()
        self.rt = self.rt_skel = self.unk_any = self.unk_root = self.morphemic = 0
        self.n = 0

    def add(self, w, c):
        if w in self.seen:
            return
        self.seen.add(w)
        exact, skel, unk_any, unk_root, morphemic = c[0], c[1], c[2], c[3], c[4]
        self.n += 1
        self.rt += exact
        self.rt_skel += skel
        self.unk_any += unk_any
        self.unk_root += unk_root
        self.morphemic += morphemic

    def to_dict(self):
        n = max(self.n, 1)
        return {
            'types': self.n,
            'rt_type_rate': round(100.0 * self.rt / n, 4),
            'rt_skel_type_rate': round(100.0 * self.rt_skel / n, 4),
            'unk_type_rate': round(100.0 * self.unk_any / n, 4),
            'unk_root_type_rate': round(100.0 * self.unk_root / n, 4),
            'morphemic_type_rate': round(100.0 * self.morphemic / n, 4),
        }


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def kathra_overlap(files):
    """How much of the held-out sample the kathra frequency table had already seen."""
    if not os.path.exists(KATHRA):
        return None
    d = json.load(open(KATHRA, encoding='utf-8'))
    counts = d['counts']
    meta = d['meta']
    # kathra normalisation: diacritics+tatweel stripped; alef hamza forms folded to bare alef;
    # ى and ة preserved.
    fold = str.maketrans({'\u0623': '\u0627', '\u0625': '\u0627', '\u0622': '\u0627',
                          '\u0671': '\u0627'})

    def norm(w):
        return DIAC_RE.sub('', w).translate(fold)

    out = {}
    for corpus in sorted({f['corpus'] for f in files}):
        sub = [f for f in files if f['corpus'] == corpus]
        tf = tt = hf = ht = 0
        for f in sub:
            for w, n in f['counts'].items():
                tf += n
                tt += 1
                if norm(w) in counts:
                    hf += n
                    ht += 1
        out[corpus] = {
            'types': tt, 'types_seen_by_kathra': ht,
            'type_overlap_pct': round(100.0 * ht / max(tt, 1), 4),
            'forms': tf, 'forms_seen_by_kathra': hf,
            'form_overlap_pct': round(100.0 * hf / max(tf, 1), 4),
        }
    out['_kathra_built_from'] = meta.get('corpus_globs_or_paths')
    out['_kathra_n_files'] = meta.get('n_files')
    out['_kathra_tokens'] = meta.get('tokens')
    out['_kathra_distinct_kept'] = meta.get('distinct_kept')
    out['_kathra_min_count'] = meta.get('min_count')
    out['_caveat'] = ('hapax pruned at min_count, so absence from kathra is uninformative for '
                      'rare forms; overlap is a LOWER bound on prior exposure')
    return out


def run_mode(cap, force_scan, dump_path=None):
    mode_env = os.environ.get('ROOTFORMER_VALIDATED_SEG', '1')
    try:
        mode_int = int(mode_env)
    except ValueError:
        mode_int = {'off': 0, 'closed': 1, 'root': 2, 'full': 2, 'on': 1}.get(mode_env.lower(), 1)

    print('=' * 90)
    print('ROOTFORMER_VALIDATED_SEG=%s  (mode %d)' % (mode_env, mode_int))
    print('=' * 90)

    sample = scan(cap, force=force_scan)
    files = sample['files']

    ev = Evaluator()
    print('[*] vocab: %d roots | %d awzan | %d prefixes | %d suffixes'
          % (ev.vocab.num_roots, ev.vocab.num_awzan, ev.vocab.num_prefixes, ev.vocab.num_suffixes))
    try:
        import validated_segmentation as vs
        print('[*] validated_segmentation.mode()=%s enabled()=%s record=%s'
              % (vs.mode(), vs.enabled(), vs._record_path()))
    except Exception as e:
        print('[*] validated_segmentation introspection failed: %r' % (e,))
    print('[*] al-\'Ayn record: attested=%d unused(muhmal)=%d ambiguous=%d'
          % (len(ev.attested), len(ev.unused), len(ev.ambiguous)))

    total = Acc()
    by_corpus = collections.defaultdict(Acc)
    by_author = collections.defaultdict(Acc)
    by_author_files = collections.defaultdict(set)
    types_all = TypeAcc()
    types_corpus = collections.defaultdict(TypeAcc)
    types_author = collections.defaultdict(TypeAcc)
    per_file = []
    diac_forms = 0
    fail_counts = collections.Counter()
    dump = {} if dump_path else None

    t0 = time.time()
    for fi, f in enumerate(files, 1):
        acc = Acc()
        ta = TypeAcc()
        for w, n in f['counts'].items():
            c = ev.score(w)
            if not c[0]:
                fail_counts[w] += n
            if dump is not None and w not in dump:
                dec, r, wz, s = ev.detail[w]
                dump[w] = [c[0], c[7], c[8], dec, r, wz, s]
            acc.add(c, n)
            ta.add(w, c)
            total.add(c, n)
            types_all.add(w, c)
            by_corpus[f['corpus']].add(c, n)
            types_corpus[f['corpus']].add(w, c)
            ak = f['author_key']
            by_author[ak].add(c, n)
            types_author[ak].add(w, c)
            by_author_files[ak].add(f['file'])
            if w != strip_diac(w):
                diac_forms += n
        d = acc.to_dict()
        d.update(ta.to_dict())
        d.update({'corpus': f['corpus'], 'file': f['file'], 'path': f['path'],
                  'author_key': f['author_key'], 'author_full': f['author_full'],
                  'n_tokens_in_file': f['n_tokens_in_file'], 'stride': f['stride'],
                  'n_sampled': f['n_sampled'], 'n_types_sampled': f['n_types']})
        per_file.append(d)
        if fi % 10 == 0 or fi == len(files):
            print('  scored %2d/%d files, %d words-through-cache, %.0fs'
                  % (fi, len(files), len(ev.cache), time.time() - t0))
    elapsed = time.time() - t0

    overall = total.to_dict()
    overall.update(types_all.to_dict())
    overall['distinct_forms'] = types_all.n
    overall['diacritic_forms'] = diac_forms
    overall['diacritic_form_rate'] = round(100.0 * diac_forms / max(total.forms, 1), 4)
    overall['words_decomposed'] = len(ev.cache)

    corpus_out = {}
    for c, a in by_corpus.items():
        d = a.to_dict()
        d.update(types_corpus[c].to_dict())
        corpus_out[c] = d

    author_out = {}
    for a, acc in by_author.items():
        d = acc.to_dict()
        d.update(types_author[a].to_dict())
        d['files'] = sorted(by_author_files[a])
        author_out[a] = d

    # top failures, occurrence-weighted over the WHOLE held-out sample
    top = []
    for w, n in fail_counts.most_common(10):
        dec, r, wz, s = ev.detail[w]
        top.append({'word': w, 'count': n, 'decoded': dec,
                    'root': r, 'wazn': wz, 'suffix': s})

    res = {
        'mode': mode_int,
        'mode_env': mode_env,
        'label': {0: '0 greedy (validator OFF)',
                  1: '1 shipped DEFAULT (closed-class only)',
                  2: '2 root re-segmentation (needs kathra)'}.get(mode_int, mode_env),
        'elapsed_s': round(elapsed, 1),
        'sample': {'forms': total.forms, 'files': len(files),
                   'distinct_forms': types_all.n,
                   'per_file_cap': sample['meta']['per_file_cap']},
        'overall': overall,
        'by_corpus': corpus_out,
        'by_author': author_out,
        'per_file': per_file,
        'top_failures': top,
        'n_failing_types': len(fail_counts),
    }

    print('\n[*] mode %d done in %.1fs: forms=%d distinct=%d roundtrip=%.2f%% '
          '(skeleton %.2f%%) morphemic=%.2f%% unk=%.2f%% muhmal=%.2f%%'
          % (mode_int, elapsed, total.forms, types_all.n,
             overall['rt_rate'], overall['rt_skel_rate'], overall['morphemic_rate'],
             overall['unk_rate'], overall['muhmal_rate']))

    blob = {}
    if os.path.exists(RESULTS):
        try:
            blob = json.load(open(RESULTS, encoding='utf-8'))
        except Exception:
            blob = {}
    blob.setdefault('meta', {})
    blob['meta'] = {
        'harness': 'eval_heldout.py',
        'built': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'release': RELEASE,
        'blueprint': BLUEPRINT,
        'blueprint_md5': md5(BLUEPRINT),
        'attest_record': ATTEST,
        'attest_counts': ev.attest_counts,
        'held_out_corpora': [c for c, _, _ in HELD_OUT],
        'contaminated_dirs_never_evaluated': list(CONTAMINATED_DIRS),
        'contamination_basis': ('kathra_counts.json meta.corpus_globs_or_paths == '
                               '/workspace/heritage_foundations/*.txt, '
                               '/workspace/andalusian_canon_sanitized/*.txt, '
                               '/workspace/scholastic_sanitized/*.txt'),
        'kathra_overlap_of_sample': kathra_overlap(files),
        'sample': sample['meta'],
        'python': sys.version.split()[0],
        'note': ('per-file and per-author results are in this JSON; the sample word counts are '
                 'cached in ' + SAMPLE_CACHE),
    }
    blob.setdefault('modes', {})
    blob['modes'][str(mode_int)] = res
    with open(RESULTS, 'w', encoding='utf-8') as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=1)
    print('[*] wrote %s (mode %d)' % (RESULTS, mode_int))

    if dump_path:
        with open(dump_path, 'w', encoding='utf-8') as fh:
            json.dump({'mode': mode_int, 'words': dump}, fh, ensure_ascii=False)
        print('[*] wrote per-word dump %s (%d words)' % (dump_path, len(dump)))
    return blob


# --------------------------------------------------------------------------------------------
# paired comparison of two modes over the SAME word sample
# --------------------------------------------------------------------------------------------
RCLASS_NAME = {REAL_ROOT: 'real-root', ROOT_PARTICLE: 'particle', ROOT_UNK: 'UNK',
               ROOT_NONE: 'PAD/BOS/EOS'}


def word_occurrences():
    """Global occurrence count per distinct form, from the cached sample."""
    with open(SAMPLE_CACHE, encoding='utf-8') as fh:
        sample = json.load(fh)
    c = collections.Counter()
    for f in sample['files']:
        for w, n in f['counts'].items():
            c[w] += n
    return c


def compare(dump_path_a, dump_path_b, write=True):
    A_ = json.load(open(dump_path_a, encoding='utf-8'))
    B_ = json.load(open(dump_path_b, encoding='utf-8'))
    ma, mb = A_['mode'], B_['mode']
    a, b = A_['words'], B_['words']
    occ = word_occurrences()
    words = [w for w in a if w in b]
    print('=== PAIRED COMPARISON over %d distinct forms (%d occurrences) ==='
          % (len(words), sum(occ[w] for w in words)))

    # 1. round-trip transition, form counts and occurrence-weighted
    trans = collections.Counter()
    for w in words:
        trans[(a[w][0], b[w][0])] += 1
    tforms = collections.Counter()
    for w in words:
        tforms[(a[w][0], b[w][0])] += occ[w]

    # 2. root-class transition
    ctrans = collections.Counter()
    for w in words:
        ctrans[(a[w][1], b[w][1])] += occ[w]

    # 3. newly broken / newly fixed, ranked by occurrences
    broken = sorted((w for w in words if a[w][0] == 1 and b[w][0] == 0),
                    key=lambda w: -occ[w])
    fixed = sorted((w for w in words if a[w][0] == 0 and b[w][0] == 1),
                   key=lambda w: -occ[w])

    # 4. the mechanism: REAL -> PARTICLE reassignments, and their round-trip fate
    rp = collections.Counter()
    for w in words:
        if a[w][1] == REAL_ROOT and b[w][1] == ROOT_PARTICLE:
            rp[(a[w][0], b[w][0])] += occ[w]

    print('\n  round-trip transition mode%d -> mode%d' % (ma, mb))
    print('    %-22s %12s %12s' % ('transition', 'distinct', 'occurrences'))
    for k, lbl in (((1, 1), 'ok -> ok'), ((1, 0), 'ok -> FAIL (newly broken)'),
                   ((0, 1), 'FAIL -> ok (newly fixed)'), ((0, 0), 'FAIL -> FAIL')):
        print('    %-22s %12d %12d' % (lbl, trans[k], tforms[k]))
    net = tforms[(0, 1)] - tforms[(1, 0)]
    print('    NET occurrences: %+d (%s)' % (net, 'gained' if net >= 0 else 'lost'))
    print('    NET distinct forms: %+d' % (trans[(0, 1)] - trans[(1, 0)]))

    print('\n  root-class transition (occurrence-weighted)')
    for (ka, kb), n in sorted(ctrans.items(), key=lambda x: -x[1])[:10]:
        print('    %-10s -> %-10s %10d' % (RCLASS_NAME.get(ka, ka), RCLASS_NAME.get(kb, kb), n))

    print('\n  REAL-root -> PARTICLE reassignments: %d occurrences' % sum(rp.values()))
    for k, lbl in (((0, 1), 'were FAILING, now decode'), ((1, 0), 'were decoding, now FAIL'),
                   ((1, 1), 'still ok'), ((0, 0), 'still FAIL')):
        if rp[k]:
            print('    %-28s %10d' % (lbl, rp[k]))

    def show(title, ws):
        print('\n  %s (top 12 by occurrences)' % title)
        print('    %-14s %9s   %-14s -> %-14s' % ('word', 'count', 'mode%d' % ma, 'mode%d' % mb))
        for w in ws[:12]:
            print('    %-14s %9d   %-14s -> %-14s' % (w, occ[w], a[w][3], b[w][3]))

    show('NEWLY BROKEN by mode %d' % mb, broken)
    show('NEWLY FIXED by mode %d' % mb, fixed)

    out = {
        'mode_a': ma, 'mode_b': mb,
        'distinct_forms': len(words),
        'occurrences': sum(occ[w] for w in words),
        'roundtrip_transition_forms': {'%d->%d' % k: v for k, v in trans.items()},
        'roundtrip_transition_occurrences': {'%d->%d' % k: v for k, v in tforms.items()},
        'net_occurrences': net,
        'net_distinct_forms': trans[(0, 1)] - trans[(1, 0)],
        'root_class_transition_occurrences': {'%d->%d' % k: v for k, v in ctrans.items()},
        'real_to_particle_occurrences': {'%d->%d' % k: v for k, v in rp.items()},
        'newly_broken': [{'word': w, 'count': occ[w], 'before': a[w][3], 'after': b[w][3],
                          'after_root': b[w][4], 'after_wazn': b[w][5],
                          'after_suffix': b[w][6]} for w in broken[:25]],
        'newly_fixed': [{'word': w, 'count': occ[w], 'before': a[w][3], 'after': b[w][3],
                         'after_root': b[w][4], 'after_wazn': b[w][5],
                         'after_suffix': b[w][6]} for w in fixed[:25]],
    }
    if write:
        blob = json.load(open(RESULTS, encoding='utf-8'))
        blob.setdefault('comparisons', {})['%d_vs_%d' % (ma, mb)] = out
        with open(RESULTS, 'w', encoding='utf-8') as fh:
            json.dump(blob, fh, ensure_ascii=False, indent=1)
        print('\n[*] comparison written into %s' % RESULTS)
    return out


# --------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------
ROWS = [
    ('round-trip rate (exact)', 'rt_rate', 2),
    ('round-trip rate (skeleton)', 'rt_skel_rate', 2),
    ('round-trip rate (types, exact)', 'rt_type_rate', 2),
    ('UNK rate (any slot)', 'unk_rate', 2),
    ('UNK rate (root slot)', 'unk_root_rate', 2),
    ('morphemic coverage', 'morphemic_rate', 2),
    ('collapse to <PARTICLE>', 'particle_rate', 2),
    ('collapse to <NONE> wazn', 'none_wazn_rate', 2),
    ('root attestation (al-\'Ayn)', 'attest_rate', 2),
    ('root muhmal (condemned)', 'muhmal_rate', 2),
    ('root unrecorded', 'unrecorded_rate', 2),
]

# which direction is an improvement, for the mode-1-vs-mode-0 verdict
HIGHER_IS_BETTER = {
    'rt_rate': True, 'rt_skel_rate': True, 'rt_type_rate': True,
    'morphemic_rate': True, 'attest_rate': True,
    'unk_rate': False, 'unk_root_rate': False,
    'particle_rate': False, 'none_wazn_rate': False,
    'muhmal_rate': False, 'unrecorded_rate': False,
}


def report():
    blob = json.load(open(RESULTS, encoding='utf-8'))
    modes = blob['modes']
    order = [m for m in ('0', '1', '2') if m in modes]
    L = []
    A = L.append

    A('# Held-out evaluation of the NRMP morphemic tokenizer')
    A('')
    A('Harness: `/workspace/eval_heldout.py`  |  results: `%s`  |  built %s'
      % (RESULTS, blob['meta'].get('built')))
    A('')
    A('## Sample')
    A('')
    sm = blob['meta']['sample']
    distinct = max((modes[m]['overall'].get('distinct_forms', 0) for m in order), default=0)
    A('- %d held-out files, cap %d word forms/file by uniform stride over each file'
      % (sm['n_files'], sm['per_file_cap']))
    A('- **%d word forms sampled**, %d distinct forms scored in every mode'
      % (sm['sampled_forms'], distinct))
    A('- corpora: %s' % ', '.join(sm['held_out_corpora']))
    A('- NOT evaluated (kathra sources): %s'
      % ', '.join(blob['meta']['contaminated_dirs_never_evaluated']))
    A('')
    A('## Metrics by mode (occurrence-weighted over the whole held-out sample)')
    A('')
    hdr = '| metric | ' + ' | '.join('mode %s' % m for m in order) + ' | mode1 - mode0 |'
    A(hdr)
    A('|---|' + '---|' * (len(order) + 1))
    for label, key, nd in ROWS:
        vals = []
        for m in order:
            v = modes[m]['overall'].get(key)
            vals.append('n/a' if v is None else ('%.*f%%' % (nd, v)))
        if '0' in modes and '1' in modes:
            d = modes['1']['overall'].get(key, 0) - modes['0']['overall'].get(key, 0)
            dv = '%+.*f pp' % (nd, d)
        else:
            dv = '-'
        A('| %s | %s | %s |' % (label, ' | '.join(vals), dv))
    A('')
    A('| denominator | ' + ' | '.join('mode %s' % m for m in order) + ' | |')
    A('|---|' + '---|' * (len(order) + 1))
    for label, key in (('word forms', 'forms'), ('distinct forms', 'distinct_forms'),
                       ('forms given a real root', 'root_forms'),
                       ('diacritic-bearing forms', 'diacritic_forms'),
                       ('distinct forms failing the round trip', 'n_failing_types')):
        A('| %s | %s | |' % (label, ' | '.join(str(modes[m]['overall'].get(key, modes[m].get(key, '-')))
                                               for m in order)))
    A('')
    A('## Per corpus')
    A('')
    for corpus in blob['meta']['sample']['held_out_corpora']:
        A('### %s' % corpus)
        A('')
        A('| metric | ' + ' | '.join('mode %s' % m for m in order) + ' |')
        A('|---|' + '---|' * len(order))
        for label, key, nd in ROWS:
            A('| %s | %s |' % (label, ' | '.join(
                'n/a' if modes[m]['by_corpus'].get(corpus, {}).get(key) is None
                else '%.*f%%' % (nd, modes[m]['by_corpus'][corpus][key]) for m in order)))
        A('| word forms | %s |' % ' | '.join(
            str(modes[m]['by_corpus'].get(corpus, {}).get('forms', '-')) for m in order))
        A('')
    A('## Per author (filename prefix; chronological corpus uses the file-stem prefix)')
    A('')
    authors = sorted({a for m in order for a in modes[m]['by_author']})
    cols = [a for a in authors
            if any(a in modes[m]['by_author'] for m in order)]
    key_authors = [a for a in ('Ghazali', 'Razi', 'Raghib') if a in cols]
    A('| author | mode | forms | distinct | round-trip | morphemic | unk | muhmal |')
    A('|---|---|---|---|---|---|---|---|')
    show = key_authors + [a for a in cols if a not in key_authors]
    for a in show:
        for m in order:
            d = modes[m]['by_author'].get(a)
            if not d:
                continue
            A('| %s | %s | %d | %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% |'
              % (a, m, d['forms'], d['types'], d['rt_rate'], d['morphemic_rate'],
                 d['unk_rate'], d['muhmal_rate']))
    A('')
    A('## Top-10 round-trip failures (occurrence-weighted)')
    A('')
    for m in order:
        A('### mode %s -- %s' % (m, modes[m]['label']))
        A('')
        A('| # | word | count | decoded | root | wazn | suffix |')
        A('|---|---|---|---|---|---|---|')
        for i, f in enumerate(modes[m]['top_failures'], 1):
            A('| %d | %s | %d | %s | %s | %s | %s |'
              % (i, f['word'], f['count'], f['decoded'], f['root'], f['wazn'], f['suffix']))
        A('')
    cmps = blob.get('comparisons') or {}
    if cmps:
        A('## Paired comparison (same word sample, mode A -> mode B)')
        A('')
        A('Marginal rates can hide offsetting changes. These are paired: each distinct form is')
        A('scored in both modes and its round-trip fate compared.')
        A('')
        A('| pair | ok -> ok | ok -> FAIL (newly broken) | FAIL -> ok (newly fixed) | '
          'FAIL -> FAIL | NET occ | NET forms |')
        A('|---|---|---|---|---|---|---|')
        for k, c in cmps.items():
            tf = c['roundtrip_transition_occurrences']
            A('| mode %d -> mode %d | %d | %d | %d | %d | %+d | %+d |'
              % (c['mode_a'], c['mode_b'],
                 tf.get('1->1', 0), tf.get('1->0', 0), tf.get('0->1', 0), tf.get('0->0', 0),
                 c['net_occurrences'], c['net_distinct_forms']))
        A('')
        for k, c in cmps.items():
            ma, mb = c['mode_a'], c['mode_b']
            A('### mode %d -> mode %d detail' % (ma, mb))
            A('')
            if c['newly_broken']:
                A('Newly broken (top by occurrences):')
                A('')
                A('| word | occurrences | mode %d decode | mode %d decode | mode %d root / wazn / suffix |'
                  % (ma, mb, mb))
                A('|---|---|---|---|---|')
                for x in c['newly_broken'][:10]:
                    A('| %s | %d | %s | %s | %s / %s / %s |'
                      % (x['word'], x['count'], x['before'], x['after'], x['after_root'],
                         x['after_wazn'], x['after_suffix']))
                A('')
            if c['newly_fixed']:
                A('Newly fixed (top by occurrences):')
                A('')
                A('| word | occurrences | mode %d decode | mode %d decode | mode %d root / wazn / suffix |'
                  % (ma, mb, mb))
                A('|---|---|---|---|---|')
                for x in c['newly_fixed'][:10]:
                    A('| %s | %d | %s | %s | %s / %s / %s |'
                      % (x['word'], x['count'], x['before'], x['after'], x['after_root'],
                         x['after_wazn'], x['after_suffix']))
                A('')
            rp = c.get('real_to_particle_occurrences') or {}
            if rp:
                A('REAL-root -> PARTICLE reassignments: %d occurrences '
                  '(FAIL->ok %d, ok->FAIL %d, still ok %d, still FAIL %d)'
                  % (sum(rp.values()), rp.get('0->1', 0), rp.get('1->0', 0),
                     rp.get('1->1', 0), rp.get('0->0', 0)))
                A('')

    ko = blob['meta'].get('kathra_overlap_of_sample')
    if ko:
        A('## Contamination check')
        A('')
        A('The kathra frequency table was built from %s (min_count=%s, %s tokens).'
          % (ko.get('_kathra_built_from'), ko.get('_kathra_min_count'), ko.get('_kathra_tokens')))
        A('')
        A('| corpus | form overlap with kathra | type overlap with kathra |')
        A('|---|---|---|')
        for c, d in sorted(ko.items()):
            if c.startswith('_'):
                continue
            A('| %s | %.2f%% | %.2f%% |' % (c, d['form_overlap_pct'], d['type_overlap_pct']))
        A('')
        A('Caveat: %s' % ko.get('_caveat'))
        A('')

    text = '\n'.join(L)
    open(REPORT_MD, 'w', encoding='utf-8').write(text + '\n')

    # console: the headline table + comparison verdict
    print('\n'.join(L))
    print('\n[*] wrote %s' % REPORT_MD)
    if '0' in modes and '1' in modes:
        d0 = modes['0']['overall']
        d1 = modes['1']['overall']
        print('\n=== DEFAULT (mode 1) vs GREEDY (mode 0) -- +pp = mode 1 larger ===')
        for label, key, nd in ROWS:
            delta = d1.get(key, 0) - d0.get(key, 0)
            good = HIGHER_IS_BETTER.get(key)
            mark = ''
            if good is True:
                mark = 'DEFAULT BETTER' if delta > 1e-9 else ('DEFAULT WORSE' if delta < -1e-9
                                                              else 'same')
            elif good is False:
                mark = 'DEFAULT BETTER' if delta < -1e-9 else ('DEFAULT WORSE' if delta > 1e-9
                                                               else 'same')
            print('  %-32s %8.2f%% -> %8.2f%%   %+7.2f pp  %s'
                  % (label, d0.get(key, 0), d1.get(key, 0), delta, mark))
    if '1' in modes and '2' in modes:
        d1 = modes['1']['overall']
        d2 = modes['2']['overall']
        print('\n=== MODE 2 (root re-seg) vs MODE 1 (default) -- +pp = mode 2 larger ===')
        for label, key, nd in ROWS:
            delta = d2.get(key, 0) - d1.get(key, 0)
            good = HIGHER_IS_BETTER.get(key)
            mark = ''
            if good is True:
                mark = 'M2 BETTER' if delta > 1e-9 else ('M2 WORSE' if delta < -1e-9 else 'same')
            elif good is False:
                mark = 'M2 BETTER' if delta < -1e-9 else ('M2 WORSE' if delta > 1e-9 else 'same')
            print('  %-32s %8.2f%% -> %8.2f%%   %+7.2f pp  %s'
                  % (label, d1.get(key, 0), d2.get(key, 0), delta, mark))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-file-cap', type=int, default=50000)
    ap.add_argument('--force-scan', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--dump-words', metavar='PATH', default=None,
                    help='also write per-word tuples/decodes for the current mode')
    ap.add_argument('--compare', nargs=2, metavar=('DUMP_A', 'DUMP_B'), default=None,
                    help='paired comparison of two --dump-words files')
    a = ap.parse_args()
    if a.compare:
        compare(a.compare[0], a.compare[1])
        return
    if a.report:
        report()
        return
    run_mode(a.per_file_cap, a.force_scan, dump_path=a.dump_words)


if __name__ == '__main__':
    main()
