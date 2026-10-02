#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sim_fixes.py -- evaluate concrete candidate fixes against the EXACT pinned failure set.

Each fix is a pure function of the (word, tuple, decode) that answers: if this fix were applied
to the release, would the form round-trip?  We never edit the release; we simulate.

The point is to separate three very different kinds of work:
  * mechanical DECODE fixes (the tuple already contains the answer, the renderer mangles it)
  * ENCODE/analysis fixes (the tuple is lossy; a decoder change cannot recover the word)
  * DATA fixes (the lexicon/vocabulary simply lacks the item)
"""
import json, os, re, sys, collections

DIAC = re.compile('[\u064b-\u0652\u0670\u0640]')

os.environ.setdefault('ROOTFORMER_VALIDATED_SEG', '1')
R = '/workspace/taxpin'
for p in (R + '/models', R):
    if p not in sys.path:
        sys.path.insert(0, p)

import nrmp_vocab as nv
import validated_segmentation as vs
import morphemic_tokenizer_v12_arabic as mt
cls = next(c for k, c in vars(nv).items() if isinstance(c, type) and 'MorphemicVocab' in k)
v = cls(R + '/data/rootformer_v12_arabic_blueprint.json')
tok = v.base_tok
val = vs.validator(tok)
eng = val.engine
AWZAN = sorted(tok.awzan_set)

rows = [json.loads(l) for l in open('/workspace/taxonomy_out/failures_m1.jsonl', encoding='utf-8')]
print('[*] failing forms: %d, occurrences %d' % (len(rows), sum(r['count'] for r in rows)))

# --- helpers ---------------------------------------------------------------------------------
def generated(root, wazn):
    if not root or root.startswith('<') or not wazn or wazn.startswith('<'):
        return None
    try:
        g = eng.generate(root, wazn)
        if g:
            return DIAC.sub('', g[0])
    except Exception:
        pass
    return None


def root_of(r):
    return r['r'][3:-1] if r['r'].startswith('<P:') else r['r']


def fix_wazn_branches(r):
    """Add the missing wazn branches to realize_root_and_wazn (use the tasrif generator)."""
    g = generated(r['r'], r['wazn'])
    if not g:
        return False
    p = '' if r['p'] in ('<NONE>', '<PAD>', '<UNK>') else r['p']
    s = '' if r['s'] in ('<NONE>', '<PAD>', '<UNK>') else r['s']
    return p + g + s == r['w']


def fix_wazn_search(r):
    """E5: no wazn was assigned; would any wazn in the inventory generate the stem?"""
    p = '' if r['p'] in ('<NONE>', '<PAD>', '<UNK>') else r['p']
    s = '' if r['s'] in ('<NONE>', '<PAD>', '<UNK>') else r['s']
    stem = r['w'][len(p):len(r['w']) - len(s)] if s else r['w'][len(p):]
    for wz in AWZAN:
        g = generated(r['r'], wz)
        if g and p + g + s == r['w']:
            return True
    return False


def fix_realizer_generate(r):
    """D2/E6: the tuple's own (root,wazn) pairs -- render with the tasrif generator."""
    g = generated(root_of(r), r['wazn'])
    if not g:
        return False
    p = '' if r['p'] in ('<NONE>', '<PAD>', '<UNK>') else r['p']
    s = '' if r['s'] in ('<NONE>', '<PAD>', '<UNK>') else r['s']
    return p + g + s == r['w']


def fix_span_canonical(r):
    """Every case where the surface IS exactly prefix+root+suffix and only the renderer differs:
    the correct decode is the concatenation (this is the fix for geminate truncation, the نهي
    special case, clitic dropping, the particle-tag collision, ...)."""
    p = '' if r['p'] in ('<NONE>', '<PAD>', '<UNK>') else r['p']
    s = '' if r['s'] in ('<NONE>', '<PAD>', '<UNK>') else r['s']
    return p + r['r'] + s == r['w']


def fix_realize_direct(r):
    """Call the realizer with the tuple's own (root, wazn), ignoring the particle-tag detour."""
    try:
        stem = tok.realize_root_and_wazn(root_of(r), r['wazn'])
    except Exception:
        return False
    p = '' if r['p'] in ('<NONE>', '<PAD>', '<UNK>') else r['p']
    s = '' if r['s'] in ('<NONE>', '<PAD>', '<UNK>') else r['s']
    return p + stem + s == r['w']


def fix_contraction(r):
    """decode_word applies على+ه -> عليه (the rule already exists in tokenizer.decode)."""
    return r['cause'] in ('D3_particle_pronoun_alif_to_ya', 'D4_preposition_ma_merger')


def fix_closed_particle_list(r):
    """COMMON_PARTICLES extended to CLOSED_PARTICLES (and orthographic variants normalised)."""
    return r['cause'] == 'E2_unk_echo_unknown_root' and r['stratum'] in (
        'closed_particle_without_P_tag', 'closed_particle_orthographic_variant',
        'clitic_plus_closed_particle')


def fix_allah(r):
    return r['cause'] == 'E13_allah_root_assimilation'


def fix_person_prefix(r):
    return r['cause'] == 'E12_person_prefix_letter_lost'


def fix_unknown_suffix(r):
    return r['cause'] == 'E10_unknown_suffix_dropped'


def fix_hamza_alif(r):
    if r['cause'] != 'E6_analysis_inconsistent_generates_false':
        return False
    if not r['stratum'].startswith('hamza_or_ta_marbuta|'):
        return False
    # the realizer writes إ/أ where the corpus writes the bare alif
    return r['dec'].replace('إ', 'ا').replace('أ', 'ا') == r['w']


def fix_gemination(r):
    if not r['stratum'].startswith('gemination|'):
        return False
    return fix_span_canonical(r)


FIXES = [
    ('F1_contraction_D3_D4', fix_contraction),
    ('F2_allah_E13', fix_allah),
    ('F3_person_prefix_E12', fix_person_prefix),
    ('F4_unknown_suffix_E10', fix_unknown_suffix),
    ('F5_closed_particle_list_E2', fix_closed_particle_list),
    ('F6_hamza_alif_E6', fix_hamza_alif),
    ('F7_gemination_truncation', fix_gemination),
    ('F8_wazn_branches_D1', lambda r: r['cause'] == 'D1_wazn_unimplemented_fallthrough' and fix_wazn_branches(r)),
    ('F9_span_canonical_E4_E7_D6', lambda r: r['cause'] in (
        'E4_clitic_dropped_by_encoder', 'E7_particle_root_substituted_for_longer_word',
        'D6_realizer_emits_clitic_not_in_surface') and fix_span_canonical(r)),
    ('F10_realize_direct_E7', lambda r: r['cause'] == 'E7_particle_root_substituted_for_longer_word' and (fix_realize_direct(r) or fix_realizer_generate(r))),
    ('F11_clitic_restore_E4', lambda r: r['cause'] == 'E4_clitic_dropped_by_encoder'),
    ('F12_realizer_generate_D2_E6', lambda r: r['cause'] in (
        'D2_realizer_mismatch_generates_true', 'E6_analysis_inconsistent_generates_false') and fix_realizer_generate(r)),
    ('F13_wazn_search_E5', lambda r: r['cause'] == 'E5_no_wazn_assigned_bare_root' and fix_wazn_search(r)),
]

# per-fix exact counts
occ = collections.Counter(); typ = collections.Counter()
for r in rows:
    for name, fn in FIXES:
        if fn(r):
            occ[name] += r['count']; typ[name] += 1
print('\nper-fix simulated gains (forms that would then round-trip)')
for name, _ in FIXES:
    print('  %-34s types=%-7d occ=%-8d' % (name, typ[name], occ[name]))

# stacked curve: apply the fixes cumulatively in the order above
covered = set()
cum = []
for name, fn in FIXES:
    newly = 0; newo = 0
    for i, r in enumerate(rows):
        if i in covered:
            continue
        if fn(r):
            covered.add(i); newly += 1; newo += r['count']
    cum.append({'fix': name, 'new_types': newly, 'new_occ': newo,
                'cum_types': len(covered), 'cum_occ': sum(rows[i]['count'] for i in covered)})
print('\ncumulative (this fix order)')
for c in cum:
    print('  %-34s +t=%-7d +o=%-8d cum t=%-7d o=%d' % (c['fix'], c['new_types'], c['new_occ'],
                                                       c['cum_types'], c['cum_occ']))

# what CANNOT be fixed by any decode-side change we can simulate: the tuple lost the surface
lost = [r for r in rows if not any(fn(r) for _, fn in FIXES)]
lo = sum(r['count'] for r in lost)
print('\n[*] not reachable by any simulated fix: %d types / %d occ (%.2f%% of failures)'
      % (len(lost), lo, 100.0 * lo / sum(r['count'] for r in rows)))
by = collections.Counter(r['cause'] for r in lost)
byo = collections.Counter()
for r in lost:
    byo[r['cause']] += r['count']
for k, n in by.most_common():
    print('    %-46s t=%-7d o=%d' % (k, n, byo[k]))

json.dump({'per_fix': {n: {'types': typ[n], 'occurrences': occ[n]} for n, _ in FIXES},
           'cumulative': cum,
           'unfixable': {'types': len(lost), 'occurrences': lo,
                         'by_cause': {k: {'types': by[k], 'occurrences': byo[k]} for k in by}},
           'fixable_total': {'types': len(covered),
                             'occurrences': sum(rows[i]['count'] for i in covered)}},
          open('/workspace/taxonomy_out/sim_fixes.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('[*] wrote /workspace/taxonomy_out/sim_fixes.json')
