#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
taxonomy.py -- CLOSED, EXHAUSTIVE taxonomy of every way the NRMP morphemic tokenizer
(mode 1 = ROOTFORMER_VALIDATED_SEG=1, the shipped default) fails to round-trip a word.

INPUTS (read-only):
  /workspace/du_m1.json                per-word dump from eval_heldout.py --dump-words (mode 1)
  /workspace/eval_heldout_sample.json  the cached scan; per-file "counts" {word: n} -- the ONLY
                                       source of occurrence weights.
  /workspace/hf_v19_2_release          the release under test

OUTPUTS (under /workspace/taxonomy_out/):
  failures_m1.jsonl   one line per failing distinct form: weight, tuple, decode, cause, stratum
  taxonomy.json       cause -> {types, occurrences, side, responsible, examples, strata}
  mode_delta.json     m0/m1/m2 paired deltas (root re-assignment regressions etc.)
  resources.json      (with --dump-resources) the tokenizer's own tables, for audit

Every failing form is assigned by a FIRST-MATCH cascade of mechanical tests.  The number of
forms that reach the cascade's final `E9_unclassified` guard is reported; it must be 0.
"""

import argparse
import ast
import collections
import inspect
import json
import os
import re
import sys
import textwrap
import time

RELEASE = '/workspace/hf_v19_2_release'
BLUEPRINT = RELEASE + '/data/rootformer_v12_arabic_blueprint.json'
SAMPLE = '/workspace/eval_heldout_sample.json'
DUMP_M1 = '/workspace/du_m1.json'
DUMP_M0 = '/workspace/du_m0.json'
DUMP_M2 = '/workspace/du_m2.json'
OUTDIR = '/workspace/taxonomy_out'
MODE = os.environ.get('ROOTFORMER_VALIDATED_SEG', '1')

if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)
if RELEASE + '/models' not in sys.path:
    sys.path.insert(0, RELEASE + '/models')

DIAC_RE = re.compile('[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
NONEISH = ('<NONE>', '<PAD>', '<UNK>', None)
HAMZA_LETTERS = 'ءأإآٱئؤ'
# conservative normalisation for LEXICON MEMBERSHIP: only the variants that are pure spelling
# of the same phoneme.  Deliberately does NOT map ئ/ؤ/ء onto ي/و/nothing -- doing that made
# كائن collide with ك+أين.
MEM_NORM = str.maketrans({'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', 'ى': 'ي', 'ة': 'ه'})
# wider normalisation, only used to name the EDIT TYPE of a decode/encode disagreement
HAMZA_NORM = str.maketrans({'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', 'ى': 'ي', 'ة': 'ه',
                            'ؤ': 'و', 'ئ': 'ي', 'ء': ''})
WIDE_NORM = str.maketrans({'ؤ': 'و', 'ئ': 'ي', 'ء': ''})


def mnorm(s):
    return s.translate(MEM_NORM)


def collapse_doubles(s):
    out = []
    for ch in s:
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return ''.join(out)


def implemented_wazn(fn):
    """The wazn names the realizer's top-level if/elif CHAIN is keyed on.

    Only the chain counts.  A `wazn in [...]` guard nested inside a `root == ...` special case
    (e.g. `if root == 'خلا' and wazn in ['يَفْعَلُ','يَفْعُلُ']`) is NOT an implementation.
    """
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    fndef = tree.body[0]

    def mentions(node, name):
        return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))

    def harvest(t, out):
        for node in ast.walk(t):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == 'wazn':
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(op, ast.Eq) and isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                        out.add(comp.value)
                    elif isinstance(op, ast.In) and isinstance(comp, (ast.List, ast.Tuple)):
                        for e in comp.elts:
                            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                                out.add(e.value)

    out = set()
    for st in fndef.body:
        if not isinstance(st, ast.If):
            continue
        node = st
        while node is not None:
            if mentions(node.test, 'wazn') and not mentions(node.test, 'root'):
                harvest(node.test, out)
            nxt = node.orelse
            node = nxt[0] if len(nxt) == 1 and isinstance(nxt[0], ast.If) else None
    return out


class Ctx(object):
    pass


def build(release):
    for p in (release + '/models', release):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    import zipfile  # noqa
    import nrmp_vocab as nv
    import morphemic_tokenizer_v12_arabic as mt
    cls = next(c for k, c in vars(nv).items() if isinstance(c, type) and 'MorphemicVocab' in k)
    v = cls(release + '/data/rootformer_v12_arabic_blueprint.json')
    tok = v.base_tok
    c = Ctx()
    c.v, c.tok = v, tok
    c.release = release
    c.module_files = {'nrmp_vocab': nv.__file__, 'tokenizer': mt.__file__}
    import hashlib
    c.module_md5 = {k: hashlib.md5(open(f, 'rb').read()).hexdigest()
                    for k, f in c.module_files.items()}
    c.validator_file = None
    c.particle_tags = {k[3:-1] for k in v.root2id if k.startswith('<P:')}
    c.func_inv = set(v.COMMON_PARTICLES) | c.particle_tags | set(mt.CLOSED_PARTICLES)
    c.compound = set(mt.COMPOUND_PARTICLES)
    c.proc = list(mt.PROCLITICS)
    c.enc = list(mt.ENCLITICS)
    c.prefixes = list(v.canonical_prefixes)
    c.suffixes = list(v.canonical_suffixes)
    c.roots = set(tok.roots_set)
    c.tokens = set(tok.token_to_id)
    c.nfunc = {mnorm(x) for x in c.func_inv}
    c.nroots = {mnorm(x) for x in c.roots}
    c.ntokens = {mnorm(x) for x in c.tokens}
    c.nclosed = {mnorm(x) for x in mt.CLOSED_PARTICLES}
    c.impl = implemented_wazn(tok.realize_root_and_wazn)
    try:
        import validated_segmentation as vs
        c.vs = vs
        c.validator = vs.validator(tok)
        c.validator_file = vs.__file__
        c.module_files['validated_segmentation'] = vs.__file__
        import hashlib
        c.module_md5['validated_segmentation'] = hashlib.md5(open(vs.__file__, 'rb').read()).hexdigest()
    except Exception:
        c.vs = None
        c.validator = None
    return c


# --------------------------------------------------------------------------------------------
# E1 / E2 strata (mechanical, ordered)
# --------------------------------------------------------------------------------------------
def _has_root_ngram(C, S, n=3):
    for i in range(max(0, len(C) - n + 1)):
        if C[i:i + n] in S:
            return True
    return False


def _has_clitic(C, ctx):
    for pre in ctx.proc:
        if C.startswith(pre) and len(C) > len(pre) + 1:
            return True
    for suf in ctx.enc:
        if C.endswith(suf) and len(C) > len(suf) + 1:
            return True
    return False


def stratum_particle_echo(C, ctx):
    if mnorm(C) in ctx.nclosed or C in ctx.compound or mnorm(C) in {mnorm(x) for x in ctx.compound}:
        return 'closed_particle_orthographic_variant'
    for pre in [''] + ctx.proc:
        if pre and not C.startswith(pre):
            continue
        rest = C[len(pre):]
        if len(rest) >= 2 and (rest in ctx.func_inv or mnorm(rest) in ctx.nfunc):
            return 'clitic_plus_closed_particle'
    for pre in [''] + ctx.proc:
        if pre and not C.startswith(pre):
            continue
        rest = C[len(pre):]
        if len(rest) >= 2 and (rest in ctx.tokens or mnorm(rest) in ctx.ntokens):
            return 'clitic_plus_known_token'
    # exact (unnormalised) clitic+root(+enclitic): the shipped loop should have found it
    for pre in [''] + ctx.proc:
        if pre and not C.startswith(pre):
            continue
        rest = C[len(pre):]
        for suf in [''] + ctx.enc:
            if suf and not rest.endswith(suf):
                continue
            stem = rest[:len(rest) - len(suf)] if suf else rest
            if len(stem) >= 2 and stem in ctx.roots:
                return 'clitic_root_enclitic_exact_match_missed'
    if C.startswith('ال') and C[2:] in ctx.roots:
        return 'definite_article_plus_bare_root_exact'
    # normalised-only matches -> orthographic normalisation gap
    for pre in [''] + ctx.proc:
        if pre and not C.startswith(pre):
            continue
        rest = C[len(pre):]
        for suf in [''] + ctx.enc:
            if suf and not rest.endswith(suf):
                continue
            stem = rest[:len(rest) - len(suf)] if suf else rest
            if len(stem) >= 2 and mnorm(stem) in ctx.nroots:
                return 'clitic_root_enclitic_normalized_match'
    if C.startswith('ال') and mnorm(C[2:]) in ctx.nroots:
        return 'definite_article_plus_bare_root_normalized'
    if len(C) <= 2:
        return 'short_or_single_char'
    if _has_root_ngram(C, ctx.roots):
        kind = ('root_ngram_hamza_or_weak_surface'
                if any(ch in HAMZA_LETTERS for ch in C) or C[-1] in 'ىء' or C[0] in 'يتنأ'
                else 'root_ngram_plain_surface')
        return kind + ('_clitic' if _has_clitic(C, ctx) else '_bare')
    if C.startswith('ال'):
        return 'no_root_evidence_oov_definite_article'
    return 'no_root_evidence_oov'


def stratum_unk(C, ptx, ctx):
    if C in ctx.func_inv or C in ctx.compound:
        return 'closed_particle_without_P_tag'
    if mnorm(C) in ctx.nfunc:
        return 'closed_particle_orthographic_variant'
    for pre in [''] + ctx.proc:
        if pre and not C.startswith(pre):
            continue
        rest = C[len(pre):]
        if len(rest) >= 2 and (rest in ctx.func_inv or mnorm(rest) in ctx.nfunc):
            return 'clitic_plus_closed_particle'
    if len(C) == 1:
        return 'single_char_token'
    if ptx:
        return 'clitic_plus_unknown_residue'
    if C in ctx.tokens:
        return 'vocab_token_not_a_root'
    for pre in [''] + ctx.proc:
        if pre and not C.startswith(pre):
            continue
        rest = C[len(pre):]
        if len(rest) >= 2 and (rest in ctx.tokens or mnorm(rest) in ctx.ntokens):
            return 'clitic_plus_known_token'
    return 'oov_unanalysable'


# --------------------------------------------------------------------------------------------
# the cascade
# --------------------------------------------------------------------------------------------
def label(w, p_str, r_str, w_str, s_str, dec, clean, ctx):
    """Return (cause, stratum, side, resp)."""
    ptx = '' if p_str in NONEISH else p_str
    stx = '' if s_str in NONEISH else s_str
    C = clean

    # 0. surface characters the codec never emits -------------------------------------------
    if DIAC_RE.search(w):
        return ('S1_surface_diacritic_loss',
                'pure_diacritic_only' if dec == DIAC_RE.sub('', w) else 'diacritic_plus_structural_error',
                'DECODE', 'nrmp_vocab.py:290-321 decode_word (no harakat in the realizer)')

    # 1. decode echoes a control token -------------------------------------------------------
    if r_str == '<PARTICLE>':
        st = stratum_particle_echo(C, ctx)
        side = 'DATA' if st == 'no_root_evidence_oov' else 'ENCODE'
        return ('E1_particle_echo_no_analysis', st, side,
                'nrmp_vocab.py:264-268 encode_word + nrmp_vocab.py:312-313 decode_word')
    if r_str == '<UNK>':
        st = stratum_unk(C, ptx, ctx)
        side = 'DATA' if st in ('oov_unanalysable',) else 'ENCODE'
        return ('E2_unk_echo_unknown_root', st, side,
                'nrmp_vocab.py:270-274 encode_word + nrmp_vocab.py:312-313 decode_word')
    if r_str in ('<PAD>', '<BOS>', '<EOS>'):
        return ('E3_pad_root_empty_decode', 'pad_root', 'ENCODE', 'nrmp_vocab.py:296-297')

    # 2. a suffix the vocabulary does not contain -> decode silently drops it ----------------
    if s_str == '<UNK>':
        return ('E10_unknown_suffix_dropped', 'suffix_not_in_vocabulary', 'ENCODE',
                'nrmp_vocab.py:283-286 encode_word + nrmp_vocab.py:304-305 decode_word')

    # 3. closed-class particle root with a clitic -------------------------------------------
    if r_str.startswith('<P:'):
        core = r_str[3:-1]
        contract = ptx + core[:-1] + 'ي' + stx
        if stx and core and core[-1] in ('ى', 'ا') and C == contract:
            return ('D3_particle_pronoun_alif_to_ya', 'alif_maqsura_plus_pronoun', 'DECODE',
                    'nrmp_vocab.py:308-310 decode_word (assimilation exists only at '
                    'morphemic_tokenizer_v12_arabic.py:849-850)')
        if core == 'ما' and ptx in ('من', 'عن') and C == ptx[:-1] + 'ما':
            return ('D4_preposition_ma_merger', 'prep_plus_ma', 'DECODE',
                    'nrmp_vocab.py:308-310 decode_word')
        if C != ptx + core + stx:
            st = 'particle_root_with_wazn_substituted' if w_str not in NONEISH \
                else 'particle_root_substituted_letters_lost'
            return ('E7_particle_root_substituted_for_longer_word', st, 'ENCODE',
                    'validated_segmentation.py:459-499 _closed_split / '
                    'morphemic_tokenizer_v12_arabic.py:340-344 layer 0')
        return ('D5_particle_concat_unclassified', 'equal_concat', 'DECODE',
                'nrmp_vocab.py:308-310')

    # 4. real root ---------------------------------------------------------------------------
    inter = ptx + r_str + stx
    # 4a. the encoder dropped a clitic the surface carries (strict affix tests)
    MUDARIA = ('ي', 'ت', 'ن', 'أ', 'است', 'مست', 'يست', 'تست')
    if C != dec:
        if dec and C.startswith(dec):
          extra_s = C[len(dec):]
          if extra_s in set(ctx.enc) | set(ctx.suffixes) and not (r_str and r_str.endswith(extra_s)):
            return ('E4_clitic_dropped_by_encoder', 'suffix_dropped', 'ENCODE',
                    'nrmp_vocab.py:283-286 encode_word (tokenizer greedy layer 1 / '
                    'COMPOUND_PARTICLES table)')
        if dec and C.endswith(dec):
            extra = C[:len(C) - len(dec)]
            if extra in ctx.proc and extra not in MUDARIA:
                return ('E4_clitic_dropped_by_encoder', 'prefix_dropped', 'ENCODE',
                        'nrmp_vocab.py:258-261 encode_word')
            if extra in MUDARIA and not (r_str and r_str[:1] == extra):
                return ('E4_clitic_dropped_by_encoder', 'person_prefix_letter_lost', 'ENCODE',
                        'nrmp_vocab.py:258-261 encode_word (prefix slot not filled)')
    # 4b. a clitic the decoder emits that the surface does not contain
    if ptx and not C.startswith(ptx):
        return ('D6_realizer_emits_clitic_not_in_surface', 'prefix_not_in_surface', 'ENCODE',
                'nrmp_vocab.py:258-261 encode_word')
    if stx and not C.endswith(stx):
        return ('D6_realizer_emits_clitic_not_in_surface', 'suffix_invented_by_realizer', 'DECODE',
                'morphemic_tokenizer_v12_arabic.py:761-767 (فِعَال + نهي special case)')

    # 4c. the root is the assimilated divine name -> the realizer cannot rebuild الله
    if r_str == 'اله':
        return ('E13_allah_root_assimilation', 'root=اله', 'DECODE',
                'morphemic_tokenizer_v12_arabic.py:366-368 (encode special case) + '
                'morphemic_tokenizer_v12_arabic.py:613-646 (realizer)')

    # 4d. the person prefix of an implemented wazn is not carried anywhere in the tuple
    if (ptx == '' and len(C) > 2 and len(dec) == len(C) and C[1:] == dec[1:]
            and C[0] in 'يتنأ' and dec[0] in 'يتنأ' and C[0] != dec[0]
            and w_str not in NONEISH and w_str in ctx.impl):
        return ('E12_person_prefix_letter_lost', 'mudaria_prefix_letter|' + str(w_str), 'ENCODE',
                'nrmp_vocab.py:240-288 encode_word (no slot carries the person letter)')

    # 4e. wazn not implemented by the realizer -> falls through to the bare root
    if w_str not in NONEISH and w_str not in ctx.impl:
        return ('D1_wazn_unimplemented_fallthrough', 'wazn=' + str(w_str), 'DECODE',
                'morphemic_tokenizer_v12_arabic.py:805 (final `return r`)')

    # 4f. no wazn at all
    if w_str in NONEISH:
        return ('E5_no_wazn_assigned_bare_root',
                'root_equals_surface_stem' if dec == C else 'root_not_surface_stem', 'ENCODE',
                'nrmp_vocab.py:316-319 decode_word')

    # 4g. wazn implemented: can the tuple's own reading generate the stem?
    stem = C[len(ptx):len(C) - len(stx)] if stx else C[len(ptx):]
    gen = None
    if ctx.validator is not None and stem:
        try:
            gen = bool(ctx.validator._generates(stem, r_str, w_str))
        except Exception:
            gen = None
    edit = _edit_type(dec, w)
    if gen:
        return ('D2_realizer_mismatch_generates_true', edit + '|' + str(w_str), 'DECODE',
                'morphemic_tokenizer_v12_arabic.py:613-805 realize_root_and_wazn')
    if gen is False:
        return ('E6_analysis_inconsistent_generates_false', edit + '|' + str(w_str), 'ENCODE',
                'morphemic_tokenizer_v12_arabic.py:334-541 _decompose_greedy / '
                'validated_segmentation.py:501-606 decompose')
    return ('E6_analysis_inconsistent_generates_false', 'generates_error|' + str(w_str), 'ENCODE',
            'morphemic_tokenizer_v12_arabic.py:334-541 / tasrif engine')


def _edit_type(dec, w):
    if dec == w:
        return 'equal'
    if dec.translate(HAMZA_NORM) == w.translate(HAMZA_NORM):
        return 'hamza_or_ta_marbuta'
    if collapse_doubles(dec) == collapse_doubles(w):
        return 'gemination'
    if dec.translate(WIDE_NORM) == w.translate(WIDE_NORM):
        return 'hamza_seat'
    if len(dec) < len(w) and (w.startswith(dec) or w.endswith(dec)):
        return 'truncation'
    if len(dec) > len(w) and (dec.startswith(w) or dec.endswith(w)):
        return 'insertion'
    return 'substitution'


# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=OUTDIR)
    ap.add_argument('--dump-resources', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--release', default=RELEASE)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print('[*] ROOTFORMER_VALIDATED_SEG=%s  release=%s' % (MODE, args.release))
    ctx = build(args.release)
    v, tok = ctx.v, ctx.tok
    if args.dump_resources:
        import morphemic_tokenizer_v12_arabic as mt
        json.dump({'impl': sorted(ctx.impl), 'roots': sorted(ctx.roots),
                   'tokens': sorted(ctx.tokens), 'particles': v.COMMON_PARTICLES,
                   'awzan': sorted(tok.awzan_set), 'prefixes': v.canonical_prefixes,
                   'suffixes': v.canonical_suffixes,
                   'closed_particles': sorted(mt.CLOSED_PARTICLES),
                   'compound_particles': {k: list(x) for k, x in mt.COMPOUND_PARTICLES.items()},
                   'proclitics': list(mt.PROCLITICS), 'enclitics': list(mt.ENCLITICS),
                   'awzan_templates': [[p.pattern, w, sp] for p, w, sp in mt.AWZAN_TEMPLATES],
                   'quad_templates': [[p.pattern, w] for p, w in mt.QUADRILITERAL_AWZAN]},
                  open(os.path.join(args.outdir, 'resources.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False)
        print('[*] resources.json written: impl=%d roots=%d tokens=%d'
              % (len(ctx.impl), len(ctx.roots), len(ctx.tokens)))
        return

    if ctx.vs is not None:
        print('[*] validated_segmentation: %s mode=%s enabled=%s'
              % (ctx.vs.__file__, ctx.vs.mode(), ctx.vs.enabled()))
    print('[*] realize implements %d wazn names; vocab has %d' % (len(ctx.impl), v.num_awzan))

    print('[*] loading sample counts ...')
    sample = json.load(open(SAMPLE, encoding='utf-8'))
    cnt = collections.Counter()
    for f in sample['files']:
        for w, n in f['counts'].items():
            cnt[w] += n
    total_occ = sum(cnt.values())
    total_types = len(cnt)

    du1 = json.load(open(DUMP_M1, encoding='utf-8'))['words']
    failing = [w for w, rec in du1.items() if not rec[0]]
    fail_occ = sum(cnt[w] for w in failing)
    print('[*] sample: %d occ / %d types; mode-1 failing: %d types / %d occ (%.4f%% / %.4f%%)'
          % (total_occ, total_types, len(failing), fail_occ,
             100.0 * len(failing) / total_types, 100.0 * fail_occ / total_occ))
    if args.limit:
        failing = failing[:args.limit]

    mism = 0
    for w in failing[:2000]:
        p, r, wz, s = v.encode_word(w)
        if v.decode_word(p, r, wz, s) == w:
            mism += 1
    print('[*] sanity: %d/2000 dumped-failing forms actually round-trip live' % mism)

    causes = collections.defaultdict(lambda: {'types': 0, 'occ': 0,
                                              'strata': collections.Counter(),
                                              'strata_occ': collections.Counter(),
                                              'strata_side': {},
                                              'side': None, 'resp': None, 'ex': []})
    jsonl = open(os.path.join(args.outdir, 'failures_m1.jsonl'), 'w', encoding='utf-8')
    t0 = time.time()
    for i, w in enumerate(failing):
        c = cnt[w]
        p_id, r_id, w_id, s_id = v.encode_word(w)
        dec = v.decode_word(p_id, r_id, w_id, s_id)
        p_str = v.id2prefix.get(p_id)
        r_str = v.id2root.get(r_id)
        w_str = v.id2wazn.get(w_id)
        s_str = v.id2suffix.get(s_id)
        clean = tok.clean_arabic(w)
        cause, stratum, side, resp = label(w, p_str, r_str, w_str, s_str, dec, clean, ctx)
        slot = causes[cause]
        slot['types'] += 1
        slot['occ'] += c
        slot['strata'][stratum] += 1
        slot['strata_occ'][stratum] += c
        slot['strata_side'][stratum] = side
        slot['side'] = side
        slot['resp'] = resp
        slot['ex'].append({'word': w, 'count': c, 'prefix': p_str, 'root': r_str,
                           'wazn': w_str, 'suffix': s_str, 'decoded': dec,
                           'stratum': stratum})
        jsonl.write(json.dumps({'w': w, 'count': c, 'p': p_str, 'r': r_str, 'wazn': w_str,
                                's': s_str, 'dec': dec, 'cause': cause, 'stratum': stratum,
                                'side': side, 'resp': resp}, ensure_ascii=False) + '\n')
        if (i + 1) % 20000 == 0:
            print('    %d/%d  %.0fs' % (i + 1, len(failing), time.time() - t0))
    jsonl.close()

    print('[*] covered %d/%d types, %d/%d occ'
          % (sum(s['types'] for s in causes.values()), len(failing),
             sum(s['occ'] for s in causes.values()), fail_occ))

    out = {'meta': {'mode': 1, 'mode_env': MODE, 'built': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'sample_occurrences': total_occ, 'sample_types': total_types,
                    'failing_types': len(failing), 'failing_occurrences': fail_occ,
                    'sample_source': SAMPLE, 'dump_source': DUMP_M1,
                    'implemented_wazn_n': len(ctx.impl),
                    'release': args.release, 'module_files': ctx.module_files,
                    'module_md5': ctx.module_md5,
                    'validator': (None if ctx.vs is None else ctx.vs.__file__)},
           'causes': {}}
    for k, s in sorted(causes.items(), key=lambda x: -x[1]['occ']):
        ex = sorted(s['ex'], key=lambda e: -e['count'])[:10]
        out['causes'][k] = {
            'types': s['types'], 'occurrences': s['occ'],
            'type_share_pct': round(100.0 * s['types'] / len(failing), 4),
            'occ_share_pct': round(100.0 * s['occ'] / fail_occ, 4),
            'type_share_of_sample_pct': round(100.0 * s['types'] / total_types, 4),
            'occ_share_of_sample_pct': round(100.0 * s['occ'] / total_occ, 4),
            'side': s['side'], 'responsible': s['resp'],
            'strata': dict(s['strata'].most_common()),
            'strata_occurrences': dict(s['strata_occ'].most_common()),
            'strata_side': s['strata_side'],
            'examples': ex}
    json.dump(out, open(os.path.join(args.outdir, 'taxonomy.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    delta = {}
    try:
        du0 = json.load(open(DUMP_M0, encoding='utf-8'))['words']
        du2 = json.load(open(DUMP_M2, encoding='utf-8'))['words']
        for name, du in (('m0', du0), ('m2', du2)):
            only_fail = [w for w in du if not du[w][0] and du1[w][0]]
            only_pass = [w for w in du if du[w][0] and not du1[w][0]]
            delta[name + '_only_fail_vs_m1'] = {'types': len(only_fail),
                                                'occurrences': sum(cnt[w] for w in only_fail)}
            delta[name + '_only_pass_vs_m1'] = {'types': len(only_pass),
                                                'occurrences': sum(cnt[w] for w in only_pass)}
            delta[name + '_examples_only_fail'] = [
                {'word': w, 'count': cnt[w], 'dec': du[w][3], 'root': du[w][4],
                 'wazn': du[w][5], 'suffix': du[w][6]} for w in
                sorted(only_fail, key=lambda x: -cnt[x])[:20]]
            delta[name + '_examples_only_pass'] = [
                {'word': w, 'count': cnt[w], 'dec': du[w][3], 'root': du[w][4],
                 'wazn': du[w][5], 'suffix': du[w][6]} for w in
                sorted(only_pass, key=lambda x: -cnt[x])[:20]]
    except Exception as e:
        delta['error'] = repr(e)
    json.dump(delta, open(os.path.join(args.outdir, 'mode_delta.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    print()
    print('%-46s %8s %10s %8s %8s' % ('cause', 'types', 'occ', 't%', 'o%'))
    for k, s in sorted(causes.items(), key=lambda x: -x[1]['occ']):
        print('%-46s %8d %10d %7.2f%% %7.2f%%' % (k, s['types'], s['occ'],
              100.0 * s['types'] / len(failing), 100.0 * s['occ'] / fail_occ))


if __name__ == '__main__':
    main()
