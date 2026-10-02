#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmp_vocab.py
Farāhīdian Next-Root/Morph Vocabulary and Disentangled Representation.
Maps Arabic words to structured tuples: (prefix, root, wazn, suffix)
and provides deterministic morphological realization.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Iterable

# ---------------------------------------------------------------------------
# AWZAN ID SPACE -- APPEND-ONLY.  DO NOT SORT.
#
# The awzan id space is PERSISTENT: saved checkpoints (morphemic_embed.wazn_embed,
# nrmp_head.wazn_head), cached NRMP training tensors and hardcoded id lists
# (past_awzan_ids, the mudari' window 114..129, ...) index it directly.
#
# History.  Until 2026-10-01 this list was built with `sorted(awzan_set)`.  When 12
# nominal patterns were added to the blueprint on 2026-10-01, that SORTED insert
# moved 124 of the 130 existing ids: the 16 mudari' slid 114..129 -> 126..141 and
# all 28 past_awzan_ids silently came to mean a different pattern.  Every saved
# checkpoint and every cached tensor was invalidated.
#
# The order below is therefore AUTHORED, not derived:
#   raw ids 0..1     : the two boundary markers, matching the historical layout
#   raw ids 2..126   : AWZAN_CANONICAL_125 -- the original patterns, historical order
#   raw ids 127..138 : AWZAN_APPENDED_12  -- added 2026-10-01, in blueprint order
# (raw ids are offset by 3 by `special_awzan`, so raw id r is nrmp wazn id r + 3.)
#
# `canonical_awzan_order()` only ever APPENDS a pattern that is new to the
# blueprint.  If one disappears it raises, because a removal would compact the id
# space -- which is the same silent-renumbering failure.
# ---------------------------------------------------------------------------
AWZAN_BOUNDARY_MARKERS = ('end', 'start')

# The 125 patterns of the original (pre-2026-10-01) inventory, in their
# historical order.  Recovered from the archived 130-awzan vocabulary at
# /workspace/rootformer_v12/v18_next_root_morph/data/nrmp_vocab.json and pinned
# here so that no future blueprint edit can move them.
AWZAN_CANONICAL_125 = [
    'أَفَاعِل', 'أَفْعَال', 'أَفْعَل', 'أَفْعَلَ', 'أَفْعُل',
    'أَفْعِلَاء', 'أَفْعِلَة', 'أُفْعِلَ', 'إِفْعَال', 'اُسْتُفْعِلَ',
    'اُفْتُعِلَ', 'اِسْتَفْعَلَ', 'اِسْتِفْعَال', 'اِسْتِفْعَالَة', 'اِفْتَعَلَ',
    'اِفْتِعَال', 'اِفْتِعَالَة', 'اِفْعَالَّ', 'اِفْعَلَلَّ', 'اِفْعَلَّ',
    'اِفْعَنْلَلَ', 'اِفْعَوَّلَ', 'اِفْعَوْعَلَ', 'اِفْعِلَال', 'اِنْفَعَلَ',
    'اِنْفِعَال', 'تَفَاعَلَ', 'تَفَاعُل', 'تَفَعَّلَ', 'تَفَعُّل',
    'تَفَعْلَلَ', 'تَفْعِلَة', 'تَفْعِيل', 'تُفُعِّلَ', 'تُفُوعِلَ',
    'فَاعَلَ', 'فَاعُول', 'فَاعِل', 'فَاعِلَة', 'فَعَل',
    'فَعَلَ', 'فَعَلَة', 'فَعَّال', 'فَعَّالَة', 'فَعَّلَ',
    'فَعُلَ', 'فَعُول', 'فَعُولَة', 'فَعِل', 'فَعِلَ',
    'فَعِيل', 'فَعِيلَة', 'فَعْل', 'فَعْلَان', 'فَعْلَب',
    'فَعْلَل', 'فَعْلَلَ', 'فَعْلَن', 'فَعْلَى', 'فَعْوَل',
    'فَوْعَل', 'فَوْعَلَة', 'فَيْعُول', 'فُعَل', 'فُعَلَاء',
    'فُعَلَة', 'فُعَيْل', 'فُعَيْلِل', 'فُعَّال', 'فُعُول',
    'فُعِلَ', 'فُعِّلَ', 'فُعْل', 'فُعْلَان', 'فُعْلَة',
    'فُعْلَى', 'فُعْلُل', 'فُعْلِلَ', 'فُوعِلَ', 'فِعَال',
    'فِعَل', 'فِعْل', 'فِعْلَان', 'فِعْلَة', 'فِعْلِل',
    'مَفَاعِل', 'مَفَاعِيل', 'مَفْعَل', 'مَفْعَلَة', 'مَفْعُول',
    'مَفْعُولَة', 'مَفْعِل', 'مَفْعِلَة', 'مُتَفَاعِل', 'مُتَفَعِّل',
    'مُتَفَعْلِل', 'مُسْتَفْعَل', 'مُسْتَفْعَلَة', 'مُسْتَفْعِل', 'مُفَاعَلَة',
    'مُفَعْلَل', 'مُفَعْلِل', 'مُفْتَعَل', 'مُفْتَعَلَة', 'مُفْتَعِل',
    'مُفْعَل', 'مُفْعِل', 'مُنْفَعِل', 'مِفْعَال', 'يَتَفَاعَلُ',
    'يَتَفَعَّلُ', 'يَسْتَفْعِلُ', 'يَفْتَعِلُ', 'يَفْعَلُ', 'يَفْعَلُّ',
    'يَفْعُلُ', 'يَفْعِلُ', 'يَنْفَعِلُ', 'يُفَاعَلُ', 'يُفَاعِلُ',
    'يُفَعَّلُ', 'يُفَعِّلُ', 'يُفَعْلِلُ', 'يُفْعَلُ', 'يُفْعِلُ',
]

# Appended 2026-10-01 (blueprint `wazn_extension.added`, blueprint token ids
# 353..364 -> nrmp wazn ids 130..141).  The first two -- فَعَائِل and فَوَاعِل --
# were added deliberately by patch_awzan_labels.py for the sound plural
# (حقائق -> فَعَائِل, حوادث -> فَوَاعِل); the other ten close the sighat muntaha
# al-jumu' gaps and arrived with the same blueprint regeneration.
AWZAN_APPENDED_12 = [
    'فَعَائِل', 'فَعَالِيل', 'فَعَالِل', 'فَوَاعِل',
    'فَعَائِيل', 'أَفَاعِيل', 'فَعَاوِيل', 'فُعُل',
    'فِعْلَى', 'أَفْعِل', 'فُعَّل', 'فِعَّال',
]

AWZAN_CANONICAL_RAW = list(AWZAN_BOUNDARY_MARKERS) + AWZAN_CANONICAL_125
AWZAN_APPENDED = tuple(AWZAN_APPENDED_12)

#: nrmp wazn ids of the appended block (130..141) -- asserted by test_awzan_order.py
AWZAN_APPENDED_IDS = tuple(range(len(AWZAN_CANONICAL_RAW) + 3,
                                 len(AWZAN_CANONICAL_RAW) + 3 + len(AWZAN_APPENDED_12)))


def canonical_awzan_order(present: Iterable[str]) -> List[str]:
    """Return the append-only awzan order for `present` (the blueprint's wazn set).

    Ids already handed out are never moved:

    * a pattern that is NEW to the blueprint is APPENDED at the end;
    * a pattern missing from the END of the canonical order (an older blueprint
      that predates the 2026-10-01 addition) is tolerated and reported -- no
      surviving id moves, the vocabulary is simply shorter;
    * a pattern missing from the MIDDLE (a hole) RAISES, because closing a hole
      would compact the id space and silently renumber every later awzan.
    """
    present = set(present)
    canonical = AWZAN_CANONICAL_RAW + list(AWZAN_APPENDED_12)
    missing = [w for w in canonical if w not in present]
    if missing:
        keep = len(canonical) - len(missing)
        if canonical[keep:] != missing:
            raise RuntimeError(
                'awzan inventory has a HOLE: %r are missing from the middle of the canonical '
                'append-only order.  Closing the hole would compact the id space and silently '
                'renumber every later awzan (invalidating the checkpoints).  If the removal is '
                'intentional, rewrite AWZAN_CANONICAL_125 / AWZAN_APPENDED_12 explicitly -- '
                'do not let the ids shift.' % (missing,))
        print('[Farahidian Vocab] NOTE: the blueprint predates %d canonical awzan pattern(s); '
              'using the %d-pattern prefix so that no id moves: %r'
              % (len(missing), keep, missing), file=sys.stderr)
    order = [w for w in canonical if w in present]
    unknown = sorted(present - set(order))
    if unknown:
        order = order + unknown
        print('[Farahidian Vocab] WARNING: %d awzan pattern(s) present in the blueprint but '
              'absent from the authored append-only order were APPENDED at ids %d..%d so that '
              'no existing id moved: %r.  Add them to AWZAN_APPENDED_12 to pin the order.'
              % (len(unknown), len(order) - len(unknown) + 3, len(order) + 2, unknown),
              file=sys.stderr)
    return order


# ============================================================================================
# CLOSED-CLASS SURFACE RECOVERY
#
# Both round-trip defects this block repairs have one cause: a closed-class word whose surface
# the encoder cannot represent is written into the ROOT slot as a generic control token
# (<PARTICLE> / <UNK>), and decode_word then writes that tag into the output TEXT as if it were
# Arabic.  A word that reaches either tag fails `decode(encode(w)) == w` BY CONSTRUCTION, before
# any morphology runs.
#
# MEASURED on the held-out sample (mode 1, 3,298,362 forms): 665,930 occurrences (20.19%) decode
# with a control token in them.  They are not one bug:
#   * the particle IS in the project's authored closed inventory but has no <P:..> id (75,603);
#   * the validator's closed-class rule is never REACHED, because the greedy analyser returned no
#     root at all and validated_segmentation.decompose() bails out first -- so the whole surface
#     is discarded (547,717);
#   * the analyser's SLOTS are wrong: the closed-class host is in PREFIX and the clitic in ROOT,
#     so neither reaches the <P:..> or suffix vocabulary (16,654);
#   * single letters / stray markers (25,955).
#
# The inventory below is AUTHORED FROM GRAMMAR THE PROJECT ALREADY CARRIES -- the base
# tokenizer's CLOSED_PARTICLES, the validator's SUPPLEMENT_CLOSED and its Sibawayhian blueprint
# partition.  It is NEVER read off an evaluation corpus.  That distinction is the point: a
# fallback that registers whatever it is shown would make the round-trip metric a tautology.
# ============================================================================================

# AL-QALB (القلب), the GENERATION direction.  The analysis direction already exists in
# models/validated_segmentation.py::_closed_split (`core[:-1] + 'ى' in self._closed`).
#
# Ibn 'Usfur, Sharh Jumal al-Zajjaji: «إن العرب قد تقلب الألف ياء مع المضمر في نحو: عليه وإليه
# ولديه».  Al-Zajjaji, Huruf al-Ma'ani (لدى): «ومع المضمر تنقلب ياء تقول لدى زيد ولديك».  Abu
# Hayyan: «وقلبت ألفه ياء لإضافته إلى المضمر، كما قلبوا في عليك ولديك».
#
# The alif is TURNED INTO ya', not deleted, and only where the sources say the alif is MUTTASIL
# (شديد الاتصال) with the pronoun.  The particles the sources name are exactly these (with the
# bare-alif spelling الى of إلى, which the corpus also writes); the rule is NOT applied to ما
# (ماه), لا (لاه), هذا, كذا or متى (متاك).
AL_QALB_PARTICLES = frozenset({'على', 'إلى', 'الى', 'لدى', 'علي', 'إلي', 'لدي'})

# the pronominal clitics, as the validator enumerates them (Sibawayh, bab 'alamat al-mudmarin)
PRONOUN_CLITICS = ('هما', 'كما', 'هن', 'هم', 'كم', 'كن', 'نا', 'ني', 'ها', 'ه', 'ك', 'ي')
PRONOUN_CLITIC_SET = frozenset(PRONOUN_CLITICS)

# prefix / suffix orders copied from models/validated_segmentation.py so that the two directions
# of the same closed-class rule cannot drift apart.
_PRE_CLOSED = ['وال', 'فال', 'بال', 'كال', 'لل', 'ال', 'و', 'ف', 'ب', 'ك', 'ل']
_SINGLE_PREPS = {'ب', 'ك', 'ل', 'س'}

_DIAC_RE = re.compile('[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')


def _strip_diac(s):
    return _DIAC_RE.sub('', s) if s else s


def _ya_alif_variants(s):
    """Word-final ي and ى are two spellings of the same letter in this orthography.

    The lexicon writes في/على/إلى/لدى/متى/الذي/التي; the corpus also writes فى/علي/إلي/لدي/متي/
    الذى/التى.  Both are the same word, so both spellings get an id.  Only the ي<->ى pair is
    swapped -- NOT the bare alif ا, which would manufacture verb-shaped surfaces (على -> علا) and
    divert words the analyser already reads as roots.  Orthography, not corpus statistics.
    """
    if not s or len(s) < 2 or s[-1] not in 'يى':
        return ()
    return (s[:-1] + ('ى' if s[-1] == 'ي' else 'ي'),)


def authored_closed_inventory(common_particles=()):
    """The project's closed-class surface inventory, assembled from sources it already carries.

    NOTHING here is derived from an evaluated corpus.  Every item comes from an authored
    inventory: this module's COMMON_PARTICLES, morphemic_tokenizer_v12_arabic's CLOSED_PARTICLES
    and COMPOUND_PARTICLES, validated_segmentation's SUPPLEMENT_CLOSED, the 4x12 bare-preposition
    + pronominal-clitic grid that design treats as one closed token, and the blueprint's own
    Sibawayhian particle partition (added by the caller).  Every source is optional; a failure
    only shrinks the inventory, never raises.
    """
    closed = set(x for x in common_particles if isinstance(x, str))
    try:
        from morphemic_tokenizer_v12_arabic import CLOSED_PARTICLES, COMPOUND_PARTICLES
    except Exception:
        try:
            from models.morphemic_tokenizer_v12_arabic import CLOSED_PARTICLES, COMPOUND_PARTICLES
        except Exception:
            CLOSED_PARTICLES, COMPOUND_PARTICLES = (), {}
    try:
        closed |= set(p for p in CLOSED_PARTICLES if isinstance(p, str))
    except Exception:
        pass
    try:
        closed |= set(k for k in COMPOUND_PARTICLES if isinstance(k, str))
    except Exception:
        pass
    try:
        from validated_segmentation import SUPPLEMENT_CLOSED
    except Exception:
        try:
            from models.validated_segmentation import SUPPLEMENT_CLOSED
        except Exception:
            SUPPLEMENT_CLOSED = ()
    try:
        closed |= set(p for p in SUPPLEMENT_CLOSED if isinstance(p, str))
    except Exception:
        pass
    # a bare preposition + pronominal clitic is ONE closed token in this design (به، له); the
    # compound grid is authored, not fitted.
    closed |= {pp + sfx for pp in _SINGLE_PREPS for sfx in PRONOUN_CLITICS}
    # the bare conjunction, which the corpus uses as a word and no list carried
    closed.add('و')
    out = set()
    for p in closed:
        p = _strip_diac(p)
        if p and not p.startswith('<'):
            out.add(p)
    return out


def blueprint_particle_surfaces(blueprint_path):
    """The blueprint's SIBawayhian particle partition as surfaces (ids 117..228 by default).

    This is the inventory the project calls authoritative and treats the hand-written lists as
    supplements to.  Absent or unreadable -> empty set, never an exception.
    """
    try:
        with open(blueprint_path, encoding='utf-8') as fh:
            bp = json.load(fh)
        i2t = bp['vocab_id_to_token']
        lo, hi = bp['partitions']['sibawayh_particles']
        return {_strip_diac(i2t[str(i)]) for i in range(lo, hi)
                if not i2t[str(i)].startswith('<')}
    except Exception:
        return set()


class FarāhīdianMorphemicVocab:
    def __init__(self, blueprint_path: str):
        self.blueprint_path = blueprint_path
        
        # Ensure parent models path is accessible
        models_dir = str(Path(blueprint_path).parent.parent / 'models')
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
            
        from morphemic_tokenizer_v12_arabic import PureArabicMorphemicTokenizerV12
        self.base_tok = PureArabicMorphemicTokenizerV12(blueprint_path)

        # Basran Syntactic Realizer Engine (محرك الضبط النحوي والإعرابي)
        try:
            from farahidian_syntactic_realizer import BasranSyntacticRealizer
            self.syntactic_realizer = BasranSyntacticRealizer(self)
        except ImportError:
            self.syntactic_realizer = None
        
        # Build Special & Particle Lists
        self.COMMON_PARTICLES = [
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'حتى', 'منذ', 'مذ',
            'إن', 'أن', 'كأن', 'لكن', 'ليت', 'لعل', 'لا', 'ما', 'لم', 'لن',
            'ليس', 'كان', 'هو', 'هي', 'هما', 'هم', 'هن', 'أنت', 'أنا', 'نحن',
            'هذا', 'هذه', 'هؤلاء', 'ذلك', 'تلك', 'أولئك', 'الذي', 'التي', 'الذين',
            'ثم', 'أو', 'بل', 'أم', 'إما', 'إذا', 'إذ', 'كل', 'بعض', 'غير',
            'سوى', 'عند', 'بين', 'فوق', 'تحت', 'قبل', 'بعد', 'حيث', 'كيف',
            'أين', 'متى', 'كم', 'أي', 'نعم', 'بلى', 'إذن', 'قد', 'سوف'
        ]
        
        # 1. ROOT VOCABULARY (~9,200)
        self.special_roots = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<PARTICLE>']
        raw_roots = sorted(list(self.base_tok.roots_set))
        
        # Dedicated particles in root slot
        particle_roots = [f'<P:{p}>' for p in self.COMMON_PARTICLES]
        self.roots_list = self.special_roots + particle_roots + raw_roots
        # ---- CLOSED-CLASS SURFACE RECOVERY: APPEND-ONLY --------------------------------------
        # The <P:..> block above covers only COMMON_PARTICLES (66 words), but the project's
        # authored closed inventory is much larger.  Every closed word outside COMMON_PARTICLES
        # therefore has no id, so encode_word discards it as <UNK>/<PARTICLE> and decode_word
        # writes that tag into the text -- the surface is destroyed.  The missing closed surfaces
        # are APPENDED AFTER raw_roots: appending is the only insertion that cannot move an
        # existing root id (the same append-only rule the awzan id space above is under).  Every
        # pre-existing root keeps its id, including all 9,114 that a checkpoint embeds.
        self._closed_inventory = authored_closed_inventory(self.COMMON_PARTICLES)
        self._closed_inventory |= blueprint_particle_surfaces(blueprint_path)
        self._closed_inventory = {p for p in self._closed_inventory if p}
        self._closed_cache = {}
        _have = set(self.roots_list)
        self.appended_particle_roots = sorted(
            f'<P:{p}>' for p in self._closed_inventory if f'<P:{p}>' not in _have)
        self._appended_particle_set = frozenset(self.appended_particle_roots)
        self.roots_list = self.roots_list + self.appended_particle_roots
        self.root2id = {r: i for i, r in enumerate(self.roots_list)}
        self.id2root = {i: r for i, r in enumerate(self.roots_list)}
        
        # 2. WAZN VOCABULARY (~142) -- APPEND-ONLY id space, see canonical_awzan_order()
        # above.  NEVER reintroduce sorted(): a sorted list renumbers every id after an
        # insertion, which silently invalidates the checkpoints and every hardcoded id list.
        self.special_awzan = ['<PAD>', '<NONE>', '<UNK>']
        raw_awzan = canonical_awzan_order(self.base_tok.awzan_set)
        _hist = min(len(raw_awzan), len(AWZAN_CANONICAL_RAW))
        assert raw_awzan[:_hist] == AWZAN_CANONICAL_RAW[:_hist], \
            'awzan ids 3..%d MUST stay byte-identical to the historical vocabulary' % (_hist + 2)
        self.awzan_list = self.special_awzan + raw_awzan
        self.wazn2id = {w: i for i, w in enumerate(self.awzan_list)}
        self.id2wazn = {i: w for i, w in enumerate(self.awzan_list)}
        
        # 3. PREFIX VOCABULARY (~26)
        self.canonical_prefixes = [
            '<PAD>', '<NONE>', '<UNK>', 'ال', 'و', 'وال', 'ف', 'فال',
            'ب', 'بال', 'ل', 'لل', 'ك', 'كال', 'كل', 'في', 'من', 'على',
            'عن', 'إلى', 'أن', 'إن', 'كأن', 'ليس', 'مع', 'س'
        ]
        self.prefix2id = {p: i for i, p in enumerate(self.canonical_prefixes)}
        self.id2prefix = {i: p for i, p in enumerate(self.canonical_prefixes)}
        
        # 4. SUFFIX VOCABULARY (~24)
        self.canonical_suffixes = [
            '<PAD>', '<NONE>', '<UNK>', 'ه', 'ها', 'هم', 'هما', 'هن',
            'ك', 'كم', 'كن', 'نا', 'ي', 'ت', 'تم', 'ا', 'ون', 'ين',
            'ان', 'ات', 'ة', 'ية', 'هم'
        ]
        # Deduplicate while preserving order
        seen = set()
        dedup_suffixes = []
        for s in self.canonical_suffixes:
            if s not in seen:
                seen.add(s)
                dedup_suffixes.append(s)
        self.canonical_suffixes = dedup_suffixes
        self.suffix2id = {s: i for i, s in enumerate(self.canonical_suffixes)}
        self.id2suffix = {i: s for i, s in enumerate(self.canonical_suffixes)}
        
        # Constants
        self.PAD_ROOT = self.root2id['<PAD>']
        self.BOS_ROOT = self.root2id['<BOS>']
        self.EOS_ROOT = self.root2id['<EOS>']
        self.UNK_ROOT = self.root2id['<UNK>']
        
        self.PAD_WAZN = self.wazn2id['<PAD>']
        self.NONE_WAZN = self.wazn2id['<NONE>']
        
        self.PAD_PREFIX = self.prefix2id['<PAD>']
        self.NONE_PREFIX = self.prefix2id['<NONE>']
        
        self.PAD_SUFFIX = self.suffix2id['<PAD>']
        self.NONE_SUFFIX = self.suffix2id['<NONE>']

    @property
    def num_roots(self) -> int:
        return len(self.roots_list)
        
    @property
    def num_awzan(self) -> int:
        return len(self.awzan_list)
        
    @property
    def num_prefixes(self) -> int:
        return len(self.canonical_prefixes)
        
    @property
    def num_suffixes(self) -> int:
        return len(self.canonical_suffixes)

    def _closed_class_split(self, clean):
        """(prefix, particle, suffix) if the whole word is closed-class material, else None.

        The GENERATION-side counterpart of ValidatedSegmentation._closed_split().  That routine
        only runs once the greedy analyser has already found a root, so a function word the
        analyser cannot analyse AT ALL (ولو، بأن، وإذا، وأما، الى، إلا) never reaches the
        closed-class rule and its whole surface is discarded.  This applies the same
        decomposition, against the same authored inventory, to the surface directly.

        Slightly more permissive than the analysis direction: a bare preposition IS allowed before
        an authored particle here.  `_closed_split` refuses it so that كان is not read as ك+ان
        when the analyser HAS a reading; this routine is only consulted after the analyser failed,
        and because the pieces are concatenated back, any split of an authored word reproduces the
        surface -- the refusal would only throw away بأن/بأنه.
        """
        s = _strip_diac(clean)
        if not s:
            return None
        cached = self._closed_cache.get(s)
        if cached is not None or s in self._closed_cache:
            return cached
        result = None
        for pre in [''] + _PRE_CLOSED:
            if pre and not s.startswith(pre):
                continue
            rest = s[len(pre):]
            if not rest:
                continue
            if rest in self._closed_inventory:
                result = (pre or None, rest, None)
                break
            for sfx in PRONOUN_CLITICS:
                if not rest.endswith(sfx) or len(rest) <= len(sfx):
                    continue
                core = rest[:-len(sfx)]
                if core in self._closed_inventory:
                    result = (pre or None, core, sfx)
                    break
                # the surface may spell the particle with a bare alif where the lexicon has
                # alif maqsura (لدا -> لدى); read it in the direction the surface is written.
                if core[-1] in 'يىا' and core[:-1] + 'ى' in self._closed_inventory:
                    result = (pre or None, core[:-1] + 'ى', sfx)
                    break
            if result is not None:
                break
        self._closed_cache[s] = result
        return result

    def encode_word(self, word: str) -> Tuple[int, int, int, int]:
        """
        Decomposes a single Arabic word into a 4-tuple of categorical integer IDs:
        (prefix_id, root_id, wazn_id, suffix_id)
        """
        clean_w = word.strip()
        if not clean_w:
            return (self.NONE_PREFIX, self.UNK_ROOT, self.NONE_WAZN, self.NONE_SUFFIX)
            
        # Check if entire word is a direct particle
        particle_tag = f'<P:{clean_w}>'
        if particle_tag in self.root2id:
            return (self.NONE_PREFIX, self.root2id[particle_tag], self.NONE_WAZN, self.NONE_SUFFIX)
            
        # Decompose using morphological analyzer
        p, r, wz, s = self.base_tok.decompose_arabic_word(clean_w)
        
        # Map Prefix
        if p is None or p == '':
            p_id = self.NONE_PREFIX
        else:
            p_id = self.prefix2id.get(p, self.prefix2id['<UNK>'])
            
        # Map Root
        if r is None or r == '':
            if clean_w in self.COMMON_PARTICLES or f'<P:{clean_w}>' in self.root2id:
                r_id = self.root2id.get(f'<P:{clean_w}>', self.root2id['<PARTICLE>'])
            else:
                # SURFACE RECOVERY 1 -- the analyser found nothing at all.  Try the closed-class
                # decomposition before discarding the word as a bare <PARTICLE>.  Without this
                # the whole surface is lost before decode ever runs.
                closed = self._closed_class_split(clean_w)
                if closed is not None and f'<P:{closed[1]}>' in self.root2id:
                    return (self.prefix2id.get(closed[0], self.NONE_PREFIX) if closed[0]
                            else self.NONE_PREFIX,
                            self.root2id[f'<P:{closed[1]}>'], self.NONE_WAZN,
                            self.suffix2id.get(closed[2], self.NONE_SUFFIX) if closed[2]
                            else self.NONE_SUFFIX)
                r_id = self.root2id['<PARTICLE>']
        else:
            particle_cand = f'<P:{r}>'
            # A ROOT WITH A WAZN IS A ROOT, NOT A PARTICLE.  This check was unguarded and only
            # appeared safe because no COMMON_PARTICLES surface happened to equal a root the
            # analyser returns for another word.  Expanding the <P:..> inventory exposes the
            # collision: خلاف analyses as root خلف + فِعَال, and 'خلف' is also a closed adverb,
            # so the particle branch discarded the wazn and decoded خلاف as خلف (947 occ), and
            # the same for اختلاف، بخلاف، مختلفة، يختلف، مخالفة، الخلاف.  The guard is applied
            # ONLY to the ids this patch APPENDED, so the 66 historical particle ids keep their
            # established behaviour exactly.
            if particle_cand in self.root2id and (
                    particle_cand not in self._appended_particle_set
                    or wz is None or wz == ''):
                r_id = self.root2id[particle_cand]
            else:
                r_id = self.root2id.get(r, self.root2id['<UNK>'])
                if r_id == self.UNK_ROOT:
                    # SURFACE RECOVERY 2 -- the analyser split the word but the reading cannot be
                    # represented.  Only reached when the root slot is <UNK>, so no currently
                    # decodable reading is touched.
                    closed = self._closed_class_split(clean_w)
                    if closed is not None and f'<P:{closed[1]}>' in self.root2id:
                        return (self.prefix2id.get(closed[0], self.NONE_PREFIX) if closed[0]
                                else self.NONE_PREFIX,
                                self.root2id[f'<P:{closed[1]}>'], self.NONE_WAZN,
                                self.suffix2id.get(closed[2], self.NONE_SUFFIX) if closed[2]
                                else self.NONE_SUFFIX)
                    # SURFACE RECOVERY 3 -- the slots are wrong, not the reading: the closed-class
                    # host is in PREFIX and the clitic in ROOT (له -> ل + ه ; ليست -> ليس + ت ;
                    # منا -> من + ا).  Re-route so both pieces survive.
                    if r and p:
                        _rs = self.suffix2id.get(r)
                        if (p in self._closed_inventory and f'<P:{p}>' in self.root2id
                                and _rs is not None
                                and _rs not in (self.PAD_SUFFIX, self.NONE_SUFFIX)):
                            return (self.NONE_PREFIX, self.root2id[f'<P:{p}>'],
                                    self.NONE_WAZN, _rs)
                        if p in self.prefix2id and f'<P:{p + r}>' in self.root2id:
                            return (self.NONE_PREFIX, self.root2id[f'<P:{p + r}>'],
                                    self.NONE_WAZN, self.NONE_SUFFIX)
                
        # Map Wazn
        if wz is None or wz == '':
            w_id = self.NONE_WAZN
        else:
            w_id = self.wazn2id.get(wz, self.wazn2id['<UNK>'])
            
        # Map Suffix
        if s is None or s == '':
            s_id = self.NONE_SUFFIX
        else:
            s_id = self.suffix2id.get(s, self.suffix2id['<UNK>'])
            
        return (p_id, r_id, w_id, s_id)

    def decode_word(self, p_id: int, r_id: int, w_id: int, s_id: int) -> str:
        """
        Deterministically realizes a (prefix, root, wazn, suffix) tuple back into
        a surface Arabic word string.
        """
        # Check special tokens
        if r_id == self.PAD_ROOT or r_id == self.BOS_ROOT or r_id == self.EOS_ROOT:
            return ''
            
        root_str = self.id2root.get(r_id, '<UNK>')
        wazn_str = self.id2wazn.get(w_id, '<NONE>')
        prefix_str = self.id2prefix.get(p_id, '<NONE>')
        suffix_str = self.id2suffix.get(s_id, '<NONE>')
        
        p_text = '' if prefix_str in ['<NONE>', '<PAD>', '<UNK>'] else prefix_str
        s_text = '' if suffix_str in ['<NONE>', '<PAD>', '<UNK>'] else suffix_str
        
        # Check particle tag
        if root_str.startswith('<P:'):
            particle_core = root_str[3:-1]
            # AL-QALB, GENERATION DIRECTION.  The alif of على/إلى/لدى is MUTTASIL with the
            # pronoun and TURNS INTO ya' (عليه، إليه، لديك); it is not deleted, and the rule is
            # not applied to ما/لا/هذا/كذا/متى.  See AL_QALB_PARTICLES above for the sources.
            if suffix_str in PRONOUN_CLITIC_SET and particle_core in AL_QALB_PARTICLES:
                particle_core = particle_core[:-1] + 'ي'
            return p_text + particle_core + s_text

        if root_str in ['<UNK>', '<PARTICLE>', '<PAD>']:
            # A CONTROL TOKEN IS NEVER SURFACE TEXT.  This branch means the encoder could not
            # represent the word, so the surface is genuinely absent from the tuple; emit only
            # the material the tuple does carry (the affixes) and never the tag.  Previously this
            # returned p_text + '<UNK>' + s_text, so every word reaching it failed the round trip
            # by construction and a literal <UNK> could enter decoded text downstream.
            return p_text + s_text
            
        # Canonical Morphological Realization
        if wazn_str not in ['<NONE>', '<PAD>', '<UNK>']:
            stem = self.base_tok.realize_root_and_wazn(root_str, wazn_str)
        else:
            stem = root_str
            
        return p_text + stem + s_text

    def encode_sentence(self, sentence: str) -> List[Tuple[int, int, int, int]]:
        """Encodes full space-delimited Arabic text into sequence of word-event tuples."""
        words = sentence.strip().split()
        return [self.encode_word(w) for w in words]

    def decode_sentence(self, word_tuples: List[Tuple[int, int, int, int]], apply_syntax: bool = True) -> str:
        """
        Decodes sequence of word-event tuples into space-delimited text.
        When apply_syntax=True, invokes the Basran Syntactic Realizer (محرك الضبط النحوي والإعرابي)
        to enforce jussive elisions, annexation nūn drop, hamza case-harmony, etc.
        """
        if apply_syntax and hasattr(self, 'syntactic_realizer') and self.syntactic_realizer is not None:
            return self.syntactic_realizer.realize_sentence_from_tuples(word_tuples)
        words = [self.decode_word(*t) for t in word_tuples]
        return ' '.join([w for w in words if w])

    def save_vocab(self, output_path: str):
        data = {
            "roots": self.roots_list,
            "awzan": self.awzan_list,
            "prefixes": self.canonical_prefixes,
            "suffixes": self.canonical_suffixes,
            "counts": {
                "roots": self.num_roots,
                "awzan": self.num_awzan,
                "prefixes": self.num_prefixes,
                "suffixes": self.num_suffixes
            }
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Farāhīdian Vocab] Saved vocabulary registry to: {output_path}")

if __name__ == '__main__':
    bp = '/home/absolut7/.gemini/antigravity-ide/scratch/rootformer/v12/data/rootformer_v12_arabic_blueprint.json'
    vocab = FarāhīdianMorphemicVocab(bp)
    print(f"Initialized Vocab: {vocab.num_roots} roots | {vocab.num_awzan} awzan | {vocab.num_prefixes} prefixes | {vocab.num_suffixes} suffixes")
    
    test_sent = 'العالم مكون ومحدث بالعلة الفاعلة في الوجود'
    encoded = vocab.encode_sentence(test_sent)
    print(f"\nSource Sentence: «{test_sent}»")
    print(f"Encoded Events: {len(encoded)} words")
    for w, t in zip(test_sent.split(), encoded):
        print(f"  {w:10} -> p:{t[0]:2} ({vocab.id2prefix[t[0]]}), r:{t[1]:4} ({vocab.id2root[t[1]]}), w:{t[2]:3} ({vocab.id2wazn[t[2]]}), s:{t[3]:2} ({vocab.id2suffix[t[3]]})")
        
    decoded = vocab.decode_sentence(encoded)
    print(f"Decoded Realization: «{decoded}»")
    print(f"Exact Reconstruction Match: {test_sent == decoded}")
    
    out_vocab = '/home/absolut7/.gemini/antigravity-ide/scratch/rootformer/v12/v18_next_root_morph/data/nrmp_vocab.json'
    vocab.save_vocab(out_vocab)
