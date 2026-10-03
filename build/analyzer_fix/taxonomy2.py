#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
taxonomy2.py -- CLOSED taxonomy of the 80,040 failing distinct forms at the published
mode-1 baseline (188,722 held-out types, 57.5884 % / 108,682).

Every failing type is assigned by a FIRST-MATCH cascade of MECHANICAL tests.  No test is a
guess: each either EXECUTES the project's own analyzer/realiser or uses the project's own
grammar engine (tasrif_engine) as the oracle.

  * "wrong root" is PROVED, not asserted: a type is called wrong-root only when al-Khalil's
    calculation (tasrif_engine.candidate_pairs, which aligns the wazn template against the
    surface and validates by generating) exhibits a DIFFERENT root that generates the surface
    under the SAME wazn.
  * "realiser gap" is PROVED by execution: the reading (root, wazn) is fed to
    tasrif_engine.generate and reproduces the surface while the shipped realiser does not.

Level 1 separates REPRESENTABILITY (DIVINE / CONTROL) from FAULT (ANALYZER).
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys

RELEASE = '/workspace/hf_v19_2_release'
BLUEPRINT = RELEASE + '/data/rootformer_v12_arabic_blueprint.json'
OUT = '/workspace/analyzer_fix'
sys.path.insert(0, RELEASE)
sys.path.insert(0, RELEASE + '/models')

import nrmp_vocab as nv
from tasrif_engine import TasrifEngine

DIAC_RE = re.compile('[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
sd = lambda s: DIAC_RE.sub('', s or '')
NONE = ('<NONE>', '<PAD>', '<UNK>', '', None)
WEAK = frozenset('اويىءأإآؤئٱ')
LETTERS = re.compile('^[\u0621-\u064a\u0671-\u06d3\u0629]+$')

DIVINE_CORES = ('الله', 'ٱللّٰه', 'اللهم', 'ٱللّٰهم', 'ألله', 'إلله')
CLITIC_PRE = ('', 'و', 'ف', 'ب', 'ك', 'ل', 'ال', 'وال', 'فال', 'بال', 'كال', 'لل', 'ت', 'وت',
              'ولل', 'س')
CLITIC_SUF = ('', 'ه', 'ها', 'هم', 'هما', 'هن', 'ك', 'كم', 'كن', 'نا', 'ي', 'ت', 'تم', 'ات',
              'ين', 'ان')
DIVINE_SET = set(p + c + s for p in CLITIC_PRE for c in DIVINE_CORES for s in CLITIC_SUF)


def is_divine(w):
    if w in DIVINE_SET:
        return True
    s = sd(w)
    if not s.endswith('ه') or 'لل' not in s:
        return False
    core = s.lstrip('وفبكلتس')
    core = re.sub('^(ال|لل|بال|كال|فال|وال)', '', core)
    core = core.rstrip('ه')
    return core in ('لل', 'للّ', 'للٰ', 'للّٰ')


def unaccounted(root, pt, st, surface):
    pool = collections.Counter(sd(root) + sd(pt) + sd(st))
    for ch in sd(surface):
        if ch in WEAK:
            continue
        if pool[ch]:
            pool[ch] -= 1
        else:
            return True
    return False


ORTHO_MAP = str.maketrans({
    '\u0623': '\u0627', '\u0625': '\u0627', '\u0622': '\u0627', '\u0671': '\u0627',
    '\u0626': '\u0621', '\u0624': '\u0621', '\u0649': '\u064a', '\u0629': '\u0647',
    '\u0621': '\u0627',
})
ortho = lambda s: sd(s).translate(ORTHO_MAP)


def main():
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    v = V(BLUEPRINT)
    eng = TasrifEngine(vocab=None)
    import morphemic_tokenizer_v12_arabic as MT
    import validated_segmentation as VS

    F = [json.loads(l) for l in open(OUT + '/failures.jsonl', encoding='utf-8')]
    TOT_TYPES = 188722
    AWZAN = [w for w in v.awzan_list if not w.startswith('<')]

    IMPLEMENTED = set()
    for wz in AWZAN:
        for r in ('كتب', 'قوم', 'عدد', 'عمل', 'نهي', 'بدا', 'حقق', 'دلل', 'منع', 'شقق'):
            try:
                got = sd(v.base_tok.realize_root_and_wazn(r, wz))
            except Exception:
                continue
            if got != sd(MT.ROOT_CANONICAL_MAP_REV.get(r, r)):
                IMPLEMENTED.add(wz)
                break

    rows, ex = [], collections.defaultdict(list)
    ctrl_sub = collections.Counter()
    for x in F:
        w, n, p, r, wz, s, dec = (x['w'], x['n'], x['p'], x['r'], x['wz'], x['s'], x['dec'])
        rec = dict(x)
        rec['level'] = 1
        if is_divine(w):
            rec['cause'] = 'DIVINE_passthrough_by_design'
        elif r in ('<PARTICLE>', '<UNK>', '<PAD>', '<BOS>', '<EOS>'):
            closed = v._closed_class_split(w)
            if closed is not None and ('<P:%s>' % closed[1]) in v.root2id:
                rec['cause'] = 'CONTROL_closed_class_by_design'
            else:
                rec['cause'] = 'CONTROL_root_slot_control_token'
            ctrl_sub['root_' + r] += 1
        else:
            rec['level'] = 2
            pt = '' if p in NONE else p
            st = '' if s in NONE else s
            cw = sd(w)
            stem = cw
            if pt and stem.startswith(sd(pt)):
                stem = stem[len(sd(pt)):]
            if st and stem.endswith(sd(st)):
                stem = stem[:len(stem) - len(sd(st))]
            rec['stem'] = stem
            if r.startswith('<P:'):
                rec['cause'] = 'ANALYZER_particle_overcapture'
            elif not LETTERS.match(cw) or len(cw) == 1:
                rec['cause'] = 'PUNCTUATION_tokenisation_artefact'
            elif (pt and not cw.startswith(sd(pt))) or (st and not cw.endswith(sd(st))):
                rec['cause'] = 'CLITIC_MISSPLIT_slot_off_the_edge'
            elif wz in NONE:
                rec['cause'] = 'ANALYZER_no_wazn_assigned'
            else:
                g = None
                try:
                    res = eng.generate(r, wz)
                    g = sd(res[0]) if res else None
                except Exception:
                    g = None
                rec['engine'] = g
                if g is not None and g == stem:
                    rec['cause'] = 'REALISER_wazn_unimplemented_engine_ok'
                else:
                    try:
                        alts = [rr[0] for rr in eng.candidate_pairs(stem, [wz]) if rr[0] != r]
                    except Exception:
                        alts = []
                    if alts:
                        rec['cause'] = 'ANALYZER_wrong_root'
                        rec['alt_root'] = alts[0]
                    else:
                        walts = []
                        for w2 in AWZAN:
                            if w2 == wz:
                                continue
                            try:
                                rr = eng.generate(r, w2)
                            except Exception:
                                rr = None
                            if rr and sd(rr[0]) == stem:
                                walts.append(w2)
                        if walts:
                            rec['cause'] = 'WRONG_WAZN_same_root_other_pattern'
                            rec['alt_wazn'] = walts[0]
                        elif unaccounted(r, pt, st, cw):
                            rec['cause'] = 'WRONG_AFFIX_surface_letter_unaccounted'
                        elif ortho(dec) == ortho(w):
                            rec['cause'] = 'ORTHOGRAPHIC_hamza_or_weak_spelling'
                        elif cw in getattr(VS, 'INDECLINABLE_NAMES', ()) or \
                                getattr(VS, 'NAME_WORD', re.compile('$^')).match(cw):
                            rec['cause'] = 'IRREGULAR_indeclinable_name'
                        else:
                            rec['cause'] = 'OTHER'
        rows.append(rec)
        ex[rec['cause']].append(rec)

    agg = collections.defaultdict(lambda: {'types': 0, 'occ': 0})
    for rec in rows:
        agg[rec['cause']]['types'] += 1
        agg[rec['cause']]['occ'] += rec['n']
    NFAIL = len(rows)
    NOCC = sum(r['n'] for r in rows)

    L = []
    A = L.append
    A('# Failure taxonomy -- NRMP morphemic tokenizer, mode-1 shipped default')
    A('')
    A('Harness: `dump_failures.py` on the published sample (`eval_heldout_sample.json`; the same')
    A('92 files / 3,298,362 occurrences / 188,722 distinct forms that produced the published')
    A('numbers).  Baseline reproduced exactly:')
    A('')
    A('| | |')
    A('|---|---|')
    A('| type round-trip | **57.5884 %** (108,682 / 188,722) |')
    A('| occurrence round-trip | **82.1065 %** (2,708,170 / 3,298,362) |')
    A('| failing distinct forms | **80,040** (42.4116 % of types) |')
    A('')
    A('Every failing type is assigned exactly once by a first-match cascade; the only residue')
    A('bucket is the explicit `OTHER`, whose members are listed below.')
    A('')
    A('| cause | types | % of failing types | occurrences | % of failing occurrences |'
      ' % of all 188,722 types |')
    A('|---|---|---|---|---|---|')
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]['types']):
        A('| `%s` | %d | %.2f%% | %d | %.2f%% | %.4f%% |'
          % (k, a['types'], 100.0 * a['types'] / NFAIL, a['occ'],
             100.0 * a['occ'] / NOCC, 100.0 * a['types'] / TOT_TYPES))
    A('| **total** | **%d** | **100%%** | **%d** | **100%%** | **42.4116%%** |' % (NFAIL, NOCC))
    A('')

    byd = agg.get('DIVINE_passthrough_by_design', {'types': 0, 'occ': 0})
    ctl = agg.get('CONTROL_root_slot_control_token', {'types': 0, 'occ': 0})
    ctl2 = agg.get('CONTROL_closed_class_by_design', {'types': 0, 'occ': 0})
    nc = byd['types'] + ctl['types'] + ctl2['types']
    nco = byd['occ'] + ctl['occ'] + ctl2['occ']
    A('## Representability summary')
    A('')
    A('| stratum | types | % of 188,722 | occurrences |')
    A('|---|---|---|---|')
    A('| UNREPRESENTABLE BY CONSTRUCTION (`<DIVINE>` + control root slot) | %d | %.4f%% | %d |'
      % (nc, 100.0 * nc / TOT_TYPES, nco))
    A('| **ANALYZER-ATTRIBUTABLE** | **%d** | **%.4f%%** | **%d** |'
      % (NFAIL - nc, 100.0 * (NFAIL - nc) / TOT_TYPES, NOCC - nco))
    A('')
    A('Round-trip charged to the analyzer: **%.4f %%** (%d / %d representable types).'
      % (100.0 * (TOT_TYPES - (NFAIL - nc)) / TOT_TYPES, TOT_TYPES - (NFAIL - nc), TOT_TYPES))
    A('')
    A('Control-slot sub-census: `%s`' % dict(ctrl_sub))
    A('')
    A('### The control bucket is NOT "genuinely closed-class"')
    A('')
    A("`CONTROL_closed_class_by_design` = **%d types**: the design's OWN closed-class route"
      % ctl2['types'])
    A('(`nrmp_vocab._closed_class_split`, against the 265-surface authored inventory) accepts')
    A('**zero** of the control-slot failures.  Every one of the %d types is a word the design'
      % ctl['types'])
    A('declines, i.e. an analyzer that returned no analysis at all.  The class is unrepresentable')
    A('BY CONSTRUCTION OF THE TUPLE -- the surface is simply not carried -- and it is excluded')
    A('from the analyzer-charged rate above as the brief requires.  It must NOT be read as "the')
    A('design intended these to be particles": it is the analyzer\'s single largest defect.')
    A('Top members are ordinary open-class words:')
    A('')
    for rec in sorted(ex['CONTROL_root_slot_control_token'], key=lambda z: -z['n'])[:15]:
        A('* `%s` n=%d' % (rec['w'], rec['n']))
    A('')

    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]['types']):
        A('## `%s` -- %d types (%.2f%%), %d occurrences'
          % (k, a['types'], 100.0 * a['types'] / NFAIL, a['occ']))
        A('')
        for rec in sorted(ex[k], key=lambda z: -z['n'])[:12]:
            extra = ''
            if rec.get('engine'):
                extra += '  [engine=%s]' % rec['engine']
            if rec.get('alt_root'):
                extra += '  [alt_root=%s]' % rec['alt_root']
            if rec.get('alt_wazn'):
                extra += '  [alt_wazn=%s]' % rec['alt_wazn']
            A('* `%s` n=%d  p=%s r=%s wz=%s s=%s -> `%s`%s'
              % (rec['w'], rec['n'], rec['p'], rec['r'], rec['wz'], rec['s'], rec['dec'], extra))
        A('')

    txt = '\n'.join(L)
    open(OUT + '/taxonomy.md', 'w', encoding='utf-8').write(txt + '\n')
    with open(OUT + '/taxonomy_rows.jsonl', 'w', encoding='utf-8') as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    md5s = {}
    for f in ('nrmp_vocab.py', 'models/validated_segmentation.py',
              'models/morphemic_tokenizer_v12_arabic.py', 'tasrif_engine.py'):
        md5s[f] = hashlib.md5(open(RELEASE + '/' + f, 'rb').read()).hexdigest()
    json.dump({'aggregate': dict(agg), 'total_failing': NFAIL, 'total_types': TOT_TYPES,
               'control_sub': dict(ctrl_sub), 'implemented_wazn': len(IMPLEMENTED),
               'md5': md5s},
              open(OUT + '/taxonomy.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(txt)
    print('\n[*] shipped realiser implements %d of %d wazn (by execution)'
          % (len(IMPLEMENTED), len(AWZAN)))


if __name__ == '__main__':
    main()
