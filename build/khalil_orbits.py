#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
khalil_orbits.py -- al-Khalīl's التقليبات (taqālīb) as a PARAMETER-TYING mechanism.

WHAT THIS MODULE DOES (ENGINEERING)
-----------------------------------
It builds, over the shipped root inventory, al-Khalīl's permutation families ("orbits"): a
root's orbit is the set of its permutations that are **attested (مستعمل)** -- the مهمل
("unused") permutations al-Khalīl discards are *not* members.  It then offers three
operations over those families:

    orbit_of(root)               -> the tieable family of a root
    tie_embeddings(weight)       -> make orbit-mates share ONE embedding row
    loss(weight)                 -> within-orbit dispersion penalty (keeps them tied in training)

WHAT THIS MODULE EXPLICITLY DOES *NOT* DO (FIDELITY -- READ THIS)
----------------------------------------------------------------
It does NOT implement al-ishtiqāq al-akbar's *semantic* claim -- that the permutations of one
root share one MEANING.  That claim is reported and REJECTED by the tradition, and the
rejection is not softened here.  Ibn ʿUsfūr, *al-Mumtiʿ fī al-Taṣrīf* (corpus:
`corpus/andalusian/13_IbnUsfur_Mumtic_fi_al_Tasrif.txt`, lines 157-161):

    «# أما الاشتقاق الأكبر هو عقد تقاليب الكلمة كلها على المعنى واحد، نحو ما ذهب إليه
     [أبو الفتح] بن جني من عقد تقاليب "القول" الستة على منى الخفة. ولم يقل به أحد من
     النحويين إلا أبا الفتح. وحكى هو عن أبي علي أنه كان يأنس به في بعض الأماكن. والصحيح
     أن هذا النحو من الاشتقاق غير مأخوذ به؛ لعدم اطراده، ولما يلحق فيه من التكلف لمن
     رامه.»

("As for the greater derivation, it is to bind all the permutations of the word to one
meaning ... and no grammarian held it except Abū al-Fatḥ [Ibn Jinnī] ... and the correct
view is that this kind of derivation is NOT ADOPTED, because it is not systematic
(li-ʿadam iṭṭirādihi).")

Consequently: the parameter tying below is justified ONLY by al-Khalīl's own accept/reject
filter in *Kitāb al-ʿAyn* (the used permutations are written down, the unused are cancelled),
NOT by any claim of shared meaning.  Any description of this mechanism as "al-ishtiqāq
al-akbar" without this caveat is wrong.

SOURCES -- VERBATIM (corpus checkouts named in the code; nothing here is a paraphrase)
--------------------------------------------------------------------------------------
[K1] al-Khalīl ibn Aḥmad al-Farāhīdī, *Kitāb al-ʿAyn*, introduction
     (`corpus/basran/Al_Khalil_Al_Ayn.txt`, line 164):

     «قال الليث: قال الخليل: اعلم أن الكلمة الثنائيَّةَ تَتَصَرَّف على وَجْهَيْن نحو: قَدْ، دَقْ،
      شَدْ، دَشُ «1» والكلمةُ الثلاثَّيُة «2» تتصرَّفُ على ستة أوجُه، وتُسمَّى مَسدُوسة «3» ...
      والكلمة الرباعية تتصرَّف على أربعة وعشرين وجها ... فَتصيرَ أربعة وعشرين وَجْهاً،
      يُكَتَب مُسْتَعْمَلها. ويُلغى مُهْمَلها ... والكلمة الخماسية تتصرّف على مائة وعشرين
      وجها ... يُسْتَعْمَل أقَلُّه ويُلغى أكثره»

     ENGINEERING note: the brief handed to this implementation renders the middle clause
     "يُكتب مُستعملها ويُلغى مُهملها"; the corpus edition above reads "يُكَتَب مُسْتَعْمَلها.
     ويُلغى مُهْمَلها" (same words, different vowelling/brace).  Quoted here as the corpus has it.

[K2] al-Khalīl's own worked chapter for the radials ع ل م
     (`corpus/basran/Al_Khalil_Al_Ayn.txt`, line 5817):

     «باب العين واللاّم والميم معهما ع ل م، ع م ل، م ع ل، ل م ع مستعملات»

     Four permutations are declared مستعملات (ع ل م، ع م ل، م ع ل، ل م ع); the remaining two
     of the six (ل ع م، م ل ع) are not in that list.  This is the chapter our orbit for علم is
     measured against.

[K3] al-Khalīl's explicit accept/reject chapter for ع د ب
     (`corpus/basran/Al_Khalil_Al_Ayn.txt`, line 4760):

     «باب العين والدال والباء معهما ع ب د- د ع ب- ب ع د- ب د ع مستعملات ع د ب- د ب ع مهملان»

     Four مستعملات and two explicitly مهملان (ع د ب، د ب ع).  This is the chapter our
     muhmal-exclusion test is measured against.

[K4] Ibn Jinnī, *al-Khaṣāʾiṣ*, «باب في الاشتقاق الأكبر»
     (`corpus/basran/Ibn_Jinni_Al_Khasais.txt`, heading line 7625, text lines 7639-7641):

     «وأما الاشتقاق الأكبر فهو أن تأخذ أصلا من الأصول الثلاثية3، فتعقد عليه وعلى تقاليبه4
      الستة معنى واحدا, تجتمع التراكيب الستة وما يتصرف من كل واحد منها عليه»

     Cited because it is the formulation the brief quotes -- i.e. the formulation this module
     declines to implement (see the Ibn ʿUsfūr rejection above).

[K5] Ibn ʿUsfūr's rejection: see the header block.  This is the tradition's own caveat and it
     must travel with any mention of al-ishtiqāq al-akbar.

THE ATTESTATION RECORD (the data this module actually reads)
-----------------------------------------------------------
`data/khalil_attest_v4.json` is the machine-readable al-ʿAyn verdict record:
`attested` (4,381 ordered permutations) / `unused` (2,750) / `ambiguous`, produced by
`khalil_attest_v4.py`.  Its own docstring states the two rules this module depends on:

  * "The record is keyed by the ORDERED sequence, because al-Khalīl marks permutations
    individually: in باب العين والدال واللام, ع د ل / ع ل د / د ل ع are مستعمل while
    د ع ل / ل ع د / ل د ع are مهمل. Order is the whole point of التقليب."
  * "The complement of a chapter's listed permutations is مuhmal, but only where al-ʿAyn
    asserts it: the window contains فقط, or the chapter already carries an explicit مهمل
    group. Otherwise unlisted permutations are merely *undiscussed* and are not condemned."

ENGINEERING CONSEQUENCE, stated plainly so nobody over-reads the numbers: because that
negative record is deliberately conservative, a permutation that a chapter merely omits (no
فقط, no explicit مهمل group) is NOT condemned, and the default `membership='attested'`
policy therefore demands *positive* attestation for membership.  That is why the orbit of
علم has 4 members and not 6 -- matching [K2] exactly -- while ل ع م and م ل ع, which
al-ʿAyn's chapter simply did not list, are excluded for want of positive attestation rather
than being called مهمل.  The alternative policy `membership='inventory_not_muhmal'` is
available and measured; it ties more (see `stats()`), and it is NOT the default.

Usage
-----
    from khalil_orbits import KhalilPermutationOrbits
    orbits = KhalilPermutationOrbits(vocab)            # vocab: roots_list/root2id/id2root
    orbits.orbit_of('علم')                             # ['علم','عمل','لمع','معل'] (farahidi order)
    embeddings = nn.Embedding(vocab.num_roots, 448)
    orbits.tie_embeddings(embeddings.weight)           # علم and عمل now share one row
    loss = orbits.loss(embeddings.weight)              # keeps them shared during training

CHECKPOINTS -- see `tie_embeddings.__doc__` and `OrbitTiedEmbedding.__doc__`
--------------------------------------------------------------------------
`tie_embeddings` is SHAPE-PRESERVING (it overwrites rows in place), so it can be applied
AFTER `load_state_dict` and does not invalidate any existing checkpoint.  `OrbitTiedEmbedding`,
which realises true storage sharing with fewer rows, DOES change the state_dict and WILL break
existing checkpoint loading; it is provided, but it is opt-in and wired nowhere.
"""
from __future__ import annotations

import itertools
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

try:  # torch is optional: orbit construction/statistics work without it
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

__all__ = [
    "KhalilPermutationOrbits",
    "OrbitTiedEmbedding",
    "normalize_root",
    "is_radical_string",
    "all_permutations",
    "tie_root_embedding_tables",
    "build_from_release",
    "FARAHIDI_ALPHABET_ORDER",
    "FARAHIDI_RANK",
    "DEFAULT_ATTEST_PATH",
    "CITATIONS",
]

MODULE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------
# CITATIONS -- verbatim, so that every fidelity claim in this file is auditable in-repo.
# ---------------------------------------------------------------------------------------
CITATIONS: Dict[str, Dict[str, str]] = {
    "al_ayn_filter": {
        "work": "al-Khalīl ibn Aḥmad al-Farāhīdī, Kitāb al-ʿAyn (introduction)",
        "file": "corpus/basran/Al_Khalil_Al_Ayn.txt",
        "line": "164",
        "text": (
            "والكلمة الرباعية تتصرَّف على أربعة وعشرين وجها ... فَتصيرَ أربعة وعشرين وَجْهاً، "
            "يُكَتَب مُسْتَعْمَلها. ويُلغى مُهْمَلها ... والكلمة الخماسية تتصرّف على مائة "
            "وعشرين وجها ... يُسْتَعْمَل أقَلُّه ويُلغى أكثره"
        ),
        "role": "the accept/reject filter that licenses the orbit SHARING implemented here",
    },
    "al_ayn_chapter_3lm": {
        "work": "al-Khalīl, Kitāb al-ʿAyn, bāb al-ʿayn wa-l-lām wa-l-mīm",
        "file": "corpus/basran/Al_Khalil_Al_Ayn.txt",
        "line": "5817",
        "text": "باب العين واللاّم والميم معهما ع ل م، ع م ل، م ع ل، ل م ع مستعملات",
        "role": "the four-face orbit of علم, the case the orbit_size test measures",
    },
    "al_ayn_chapter_3db": {
        "work": "al-Khalīl, Kitāb al-ʿAyn, bāb al-ʿayn wa-l-dāl wa-l-bāʾ",
        "file": "corpus/basran/Al_Khalil_Al_Ayn.txt",
        "line": "4760",
        "text": "باب العين والدال والباء معهما ع ب د- د ع ب- ب ع د- ب د ع مستعملات ع د ب- د ب ع مهملان",
        "role": "explicit مهمل exclusion: عدب/دبع must NOT be tied to عبد/دعب/بعد/بدع",
    },
    "ibn_jinni_orbit": {
        "work": "Ibn Jinnī, al-Khaṣāʾiṣ, bāb fī al-ishtiqāq al-akbar",
        "file": "corpus/basran/Ibn_Jinni_Al_Khasais.txt",
        "line": "7639-7641",
        "text": (
            "وأما الاشتقاق الأكبر فهو أن تأخذ أصلا من الأصول الثلاثية، فتعقد عليه وعلى "
            "تقاليبه الستة معنى واحدا، تجتمع التراكيب الستة وما يتصرف من كل واحد منها عليه"
        ),
        "role": "the SEMANTIC formulation this module declines to implement",
    },
    "ibn_usfur_rejection": {
        "work": "Ibn ʿUsfūr, al-Mumtiʿ fī al-Taṣrīf",
        "file": "corpus/andalusian/13_IbnUsfur_Mumtic_fi_al_Tasrif.txt",
        "line": "157-161",
        "text": (
            "ولم يقل به أحد من النحويين إلا أبا الفتح. وحكى هو عن أبي علي أنه كان يأنس به في "
            "بعض الأماكن. والصحيح أن هذا النحو من الاشتقاق غير مأخوذ به؛ لعدم اطراده، ولما "
            "يلحق فيه من التكلف لمن رامه."
        ),
        "role": "the tradition's rejection of the semantic claim (mandatory caveat)",
    },
}

# ---------------------------------------------------------------------------------------
# al-Fārāhīdī's phonetic alphabet order.  ENGINEERING: this table mirrors
# `models/khalil_combinatorics.FARAHIDI_ALPHABET_ORDER`; it is duplicated here so that this
# module stays self-contained (it is imported by the audit from two different sys.path
# entries).  `verify_alphabet_mirror()` in the test suite asserts the two agree.
# It is used ONLY to pick a canonical representative for an orbit -- see
# `_representative_id` for what is and is not claimed about it.
# ---------------------------------------------------------------------------------------
FARAHIDI_ALPHABET_ORDER: List[str] = [
    'ع', 'ح', 'ه', 'خ', 'غ',          # Halqiyyah (throat)
    'ق', 'ك',                          # Lahawiyyah
    'ج', 'ش', 'ض',                     # Shajariyyah
    'ص', 'س', 'ز',                     # Asaliyyah
    'ط', 'د', 'ت',                     # Nat'iyyah
    'ظ', 'ذ', 'ث',                     # Lithawiyyah
    'ر', 'ل', 'ن',                     # Dhalaqiyyah
    'ف', 'ب', 'م',                     # Shafawiyyah
    'و', 'ي', 'ا', 'ء',                # Hawi / Jawf / glides
]
FARAHIDI_RANK: Dict[str, int] = {c: i for i, c in enumerate(FARAHIDI_ALPHABET_ORDER)}
_UNRANKED = len(FARAHIDI_ALPHABET_ORDER) + 1

DIACRITICS = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
_ARABIC_LETTER = re.compile(r'^[\u0621-\u064a]+$')
# ENGINEERING: the shipped lexicon writes the hamza radical as bare alif (افل, not أفل), and
# `khalil_attest_v4.py` canonicalises its output the same way ("canonicalisation to bare alif
# on output, because the shipped lexicon writes the hamza radical as ا").  Folding the hamza
# forms is therefore required for the record's keys to meet the inventory's keys at all.
_HAMZA_FOLD = str.maketrans({'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', 'ء': 'ا'})

DEFAULT_ATTEST_PATH = MODULE_DIR / 'data' / 'khalil_attest_v4.json'
_ATTEST_CANDIDATES = (
    MODULE_DIR / 'data' / 'khalil_attest_v4.json',
    MODULE_DIR.parent / 'data' / 'khalil_attest_v4.json',
    Path('/workspace/hf_v19_2_release/data/khalil_attest_v4.json'),
)


def normalize_root(root: Any, fold_hamza: bool = True) -> str:
    """ENGINEERING: canonical form used for every inventory/attest-record comparison.

    Strips the manuscript diacritics and tatweel, drops separators, and (by default) folds
    the hamza spellings onto bare alif, exactly as the al-ʿAyn record does.
    """
    s = DIACRITICS.sub('', str(root)).strip()
    s = s.replace('-', '').replace(' ', '')
    if fold_hamza:
        s = s.translate(_HAMZA_FOLD)
    return s


def is_radical_string(s: str) -> bool:
    """ENGINEERING: a usable radical string is >= 2 Arabic letters and nothing else."""
    return bool(s) and len(s) >= 2 and bool(_ARABIC_LETTER.match(s))


def all_permutations(root: str) -> List[str]:
    """al-Khalīl's التقليبات for one root: distinct n! arrangements (`set` because a root may
    repeat a radical, and al-ʿAyn does not count نتت against itself as two faces)."""
    return sorted({''.join(p) for p in itertools.permutations(root)})


def _orbit_key(root: str) -> Tuple[str, ...]:
    """The orbit key: the root's radicals as a multiset.  Order is discarded HERE and only
    here -- the membership test then re-imposes al-Khalīl's per-permutation verdict."""
    return tuple(sorted(root))


class KhalilPermutationOrbits:
    """al-Khalīl's permutation orbits over an inventory, as a parameter-tying mechanism.

    NOT al-ishtiqāq al-akbar.  The semantic claim that an orbit shares a MEANING is rejected
    by the tradition (Ibn ʿUsfūr: «غير مأخوذ به؛ لعدم اطراده») and is not implemented,
    optimised for, or implied anywhere in this class.  What is implemented is exactly what
    al-Khalīl's own filter licenses: the permutations he records as مستعمل form one family,
    and their parameters are shared.

    Parameters
    ----------
    source : vocab-like | iterable[str] | dict[str, int]
        Anything with `roots_list`/`root2id` (e.g. `nrmp_vocab.FarāhīdianMorphemicVocab`),
        or a plain iterable of root strings.
    attest_path : path, optional
        `khalil_attest_v4.json`.  Defaults to `<release>/data/khalil_attest_v4.json`
        (with fallbacks); if it cannot be found the class still builds orbits but switches to
        `inventory_not_muhmal` membership and says so in `stats()['attest_source']`.
    lengths : tuple[int, ...]
        Radical counts to build orbits for.  Default `(3,)`: the S³ family the audit and
        `nrmt_arch.orbit_consistency_loss` speak of.  al-Khalīl's own sentence [K1] also
        covers 2 (وجهان), 4 (أربعة وعشرون وجها) and 5 (مائة وعشرون وجها); pass e.g.
        `lengths=(2, 3, 4, 5)` to include them (a 4-radical orbit holds up to 24 members).
    membership : {'attested', 'inventory_not_muhmal'}
        'attested' (default) -- a permutation is a member only if al-ʿAyn records it as
            مستعمل.  This is the faithful reading of «يُكتب مستعملها ويُلغى مهملها» and it
            reproduces al-ʿAyn's own chapter counts exactly (علم: 4 of 6, [K2]).
        'inventory_not_muhmal' -- a permutation is a member if it is in the shipped inventory
            and al-ʿAyn does not condemn it as مهمل.  Ties strictly more; measured, not default.
    min_orbit_size : int
        Orbits smaller than this are not tieable (default 2 -- a family of one shares nothing).
    representative : {'farahidi', 'lowest_id'}
        How to choose the row an orbit keeps when `tie_embeddings(method='representative')`.
        'farahidi' (default) picks the member al-Fārāhīdī's alphabet ranks first.
        ENGINEERING / NOT CLAIMED: this reproduces the chapter HEADWORD in the two chapters we
        checked (ع ل م → علم, ع د ب → عبد), but the ordering of the remaining members is NOT
        claimed to reproduce al-ʿAyn's chapter order.
    fold_hamza : bool
        Passed to `normalize_root` (default True; see the module header).
    """

    def __init__(self, source: Any, attest_path: Optional[Union[str, Path]] = None,
                 lengths: Sequence[int] = (3,), membership: str = 'attested',
                 min_orbit_size: int = 2, representative: str = 'farahidi',
                 fold_hamza: bool = True, verbose: bool = False):

        self.fold_hamza = bool(fold_hamza)
        self.lengths = tuple(int(n) for n in lengths)
        self.membership = membership
        self.min_orbit_size = int(min_orbit_size)
        self.representative = representative
        self.verbose = bool(verbose)

        # ---------------------------------------------------------------- inventory
        self.roots_list: List[str] = []
        self.root2id: Dict[str, int] = {}
        if isinstance(source, dict):
            self.root2id = {str(k): int(v) for k, v in source.items()}
            self.roots_list = [None] * (max(self.root2id.values()) + 1)  # type: ignore
            for k, v in self.root2id.items():
                self.roots_list[v] = k
        elif hasattr(source, 'roots_list') and hasattr(source, 'root2id'):
            self.roots_list = list(source.roots_list)
            self.root2id = {str(k): int(v) for k, v in dict(source.root2id).items()}
            if not self.root2id:  # pragma: no cover - defensive
                self.root2id = {r: i for i, r in enumerate(self.roots_list)}
        else:
            self.roots_list = [str(r) for r in source]
            self.root2id = {r: i for i, r in enumerate(self.roots_list)}
        if len(self.roots_list) < len(self.root2id):  # pragma: no cover - defensive
            self.roots_list = [None] * (max(self.root2id.values()) + 1)  # type: ignore
            for k, v in self.root2id.items():
                self.roots_list[v] = k
        self.num_roots = len(self.roots_list)

        # ---------------------------------------------------------------- attestation
        self.attest_path: Optional[Path] = None
        self.attest_source: str = 'none'
        self.attested: Set[str] = set()
        self.muhmal: Set[str] = set()
        self.ambiguous: Set[str] = set()
        self.attest_counts: Dict[str, Any] = {}
        self._load_attest(attest_path)
        self.membership_effective = self.membership if self.attest_source == 'file' \
            else 'inventory_not_muhmal'
        if self.attest_source != 'file' and self.membership == 'attested' and verbose:
            print('[KhalilPermutationOrbits] WARNING: attest record not found; '
                  "membership falls back to 'inventory_not_muhmal'")

        # ---------------------------------------------------------------- orbits
        # ENGINEERING: one normalised key -> the inventory roots under it (id order).
        self._by_key: Dict[Tuple[str, ...], List[str]] = OrderedDict()
        self._root_key: Dict[str, Tuple[str, ...]] = {}
        self._excluded_muhmal: Dict[Tuple[str, ...], List[str]] = {}
        self._excluded_unattested: Dict[Tuple[str, ...], List[str]] = {}
        self._skipped: List[str] = []
        self.n_inventory_considered = 0
        for raw in self.roots_list:
            if raw is None:
                continue
            r = normalize_root(raw, self.fold_hamza)
            if not is_radical_string(r) or len(r) not in self.lengths:
                self._skipped.append(str(raw))
                continue
            key = _orbit_key(r)
            self._by_key.setdefault(key, []).append(str(raw))
            self._root_key[str(raw)] = key
            self._root_key.setdefault(r, key)
            self.n_inventory_considered += 1

        self.orbits: Dict[Tuple[str, ...], List[str]] = OrderedDict()
        self.isolated: List[str] = []
        for key, members in self._by_key.items():
            tieable, muhmal_members, unattested_members = [], [], []
            for m in members:
                nm = normalize_root(m, self.fold_hamza)
                if self.membership_effective == 'attested':
                    if nm in self.attested:
                        tieable.append(m)
                    elif nm in self.muhmal:
                        muhmal_members.append(m)
                    else:
                        unattested_members.append(m)
                else:  # inventory_not_muhmal
                    if nm in self.muhmal:
                        muhmal_members.append(m)
                    else:
                        tieable.append(m)
            if muhmal_members:
                self._excluded_muhmal[key] = muhmal_members
            if unattested_members:
                self._excluded_unattested[key] = unattested_members
            if len(tieable) >= self.min_orbit_size:
                self.orbits[key] = self._order_members(tieable)
            else:
                self.isolated.extend(tieable)

        # number of embedding rows that tying collapses, and the pairs it creates
        self.n_tied_rows = sum(len(v) for v in self.orbits.values())
        self.n_distinct_rows = self.n_tied_rows - len(self.orbits)
        # ENGINEERING: an orbit's FACE count is what al-Khalīl counts (the six وجوه of ع ل م).
        # The shipped inventory writes one face more than once (بدء / بدأ / بدا all fold to
        # بدا), so rows > faces wherever hamza spellings multiply; both are reported.
        self.n_faces = sum(len({normalize_root(m, self.fold_hamza) for m in v})
                           for v in self.orbits.values())
        self.n_duplicate_rows = self.n_tied_rows - self.n_faces

    # ------------------------------------------------------------------ attest record
    def _load_attest(self, attest_path: Optional[Union[str, Path]]) -> None:
        candidates = [Path(attest_path)] if attest_path else list(_ATTEST_CANDIDATES)
        path = next((p for p in candidates if p and Path(p).is_file()), None)
        if path is None:
            return
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        def _seqs(field: str) -> Set[str]:
            out: Set[str] = set()
            for item in data.get(field, []) or []:
                s = ''.join(item) if isinstance(item, (list, tuple)) else str(item)
                out.add(normalize_root(s, self.fold_hamza))
            return out
        self.attested = _seqs('attested')
        self.muhmal = _seqs('unused')
        self.ambiguous = _seqs('ambiguous')
        self.attest_counts = dict(data.get('counts', {}) or {})
        self.attest_path = Path(path)
        self.attest_source = 'file'

    # ------------------------------------------------------------------ ordering
    @staticmethod
    def _rank_tuple(root: str) -> Tuple[int, ...]:
        return tuple(FARAHIDI_RANK.get(c, _UNRANKED) for c in root)

    def _order_members(self, members: List[str]) -> List[str]:
        if self.representative == 'farahidi':
            return sorted(members, key=lambda r: (self._rank_tuple(normalize_root(r, self.fold_hamza)),
                                                  self.root2id.get(r, 0)))
        return sorted(members, key=lambda r: self.root2id.get(r, 0))

    def _representative_id(self, members: Sequence[str], mode: Optional[str] = None) -> int:
        mode = mode or self.representative
        if mode == 'farahidi':
            ordered = sorted(members, key=lambda r: (self._rank_tuple(
                normalize_root(r, self.fold_hamza)), self.root2id.get(r, 0)))
        else:
            ordered = sorted(members, key=lambda r: self.root2id.get(r, 0))
        return self.root2id[ordered[0]]

    # ------------------------------------------------------------------ public orbit API
    def orbit_key(self, root: str) -> Optional[Tuple[str, ...]]:
        """The multiset key of a root, or None if the root is not in the inventory."""
        if str(root) in self._root_key:
            return self._root_key[str(root)]
        return self._root_key.get(normalize_root(root, self.fold_hamza))

    def orbit_of(self, root: str) -> List[str]:
        """The tieable family of `root` (al-Khalīl's مستعمل permutations under its key).

        Returns `[root]` when the root is in the inventory but nothing else under its key is
        attested (a family of one shares nothing), and `[]` when the root is unknown.
        """
        key = self.orbit_key(root)
        if key is None:
            return []
        if key in self.orbits:
            return list(self.orbits[key])
        nm = normalize_root(root, self.fold_hamza)
        for m in self._by_key.get(key, []):
            if normalize_root(m, self.fold_hamza) == nm:
                return [m]
        return []

    def partners(self, root: str) -> List[str]:
        """`orbit_of(root)` minus the root itself."""
        return [r for r in self.orbit_of(root) if r != root]

    def permutations_of(self, root: str) -> List[str]:
        """All n! permutations (التقليبات) of a root, attested or not."""
        return all_permutations(normalize_root(root, self.fold_hamza))

    def excluded(self, root: str) -> Dict[str, List[str]]:
        """Under the root's key: the inventory permutations al-ʿAyn condemns (مهمل) and those
        it never positively records, plus the permutations absent from the inventory."""
        key = self.orbit_key(root)
        if key is None:
            return {'muhmal': [], 'unattested': [], 'not_in_inventory': []}
        inv = {normalize_root(m, self.fold_hamza) for m in self._by_key.get(key, [])}
        return {
            'muhmal': list(self._excluded_muhmal.get(key, [])),
            'unattested': list(self._excluded_unattested.get(key, [])),
            'not_in_inventory': [p for p in all_permutations(''.join(key)) if p not in inv],
        }

    def is_attested(self, root: str) -> bool:
        return normalize_root(root, self.fold_hamza) in self.attested

    def is_muhmal(self, root: str) -> bool:
        return normalize_root(root, self.fold_hamza) in self.muhmal

    def __contains__(self, root: str) -> bool:
        return self.orbit_key(root) is not None

    def __len__(self) -> int:
        return len(self.orbits)

    def __iter__(self) -> Iterator[List[str]]:
        return iter(self.orbits.values())

    # ------------------------------------------------------------------ tying
    def orbit_index(self, num_roots: Optional[int] = None) -> 'torch.Tensor':
        """LongTensor mapping each root id to its orbit's representative id (identity for
        roots in no tieable orbit).  Shape-preserving: use it to fill a tied table."""
        if torch is None:  # pragma: no cover
            raise RuntimeError('orbit_index requires torch')
        n = int(num_roots or self.num_roots)
        idx = torch.arange(n, dtype=torch.long)
        for members in self.orbits.values():
            rep = self._representative_id(members)
            for m in members:
                i = self.root2id.get(m)
                if i is not None:
                    idx[i] = rep
        return idx

    def tie_embeddings(self, embedding_weight: Any, method: str = 'mean',
                       inplace: bool = True, representative: Optional[str] = None,
                       dtype: Any = None) -> Dict[str, Any]:
        """Make every orbit's members share ONE row of `embedding_weight`.

        This is the parameter-tying operation the audit asks for: after the call,
        `weight[root2id['علم']] == weight[root2id['عمل']]` exactly, because they are one family.

        CHECKPOINT IMPACT -- this is shape-preserving.  Rows are overwritten in place; the
        tensor keeps its `num_roots x d` shape and its name, so it can be applied AFTER
        `load_state_dict` and it invalidates no existing checkpoint.  (A *storage*-sharing
        implementation with fewer rows would change the state_dict; see `OrbitTiedEmbedding`,
        which is provided but wired nowhere.)

        Parameters
        ----------
        method : {'mean', 'representative'}
            'mean' (default) -- the orbit's mean row; symmetric, keeps information from every
                member and leaves no arbitrary privileged member when gradients flow.
            'representative' -- copy the `representative` member's row to the whole orbit
                (al-Fārāhīdī's earliest-ranked member by default).
        inplace : bool
            False returns a tied copy and leaves `embedding_weight` untouched.
        dtype : optional
            dtype of the returned tensor when `inplace=False` (defaults to the input dtype).

        Returns
        -------
        dict with 'orbits_tied', 'rows_tied', 'rows_collapsed', 'method', 'representatives'.
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError('tie_embeddings requires torch')
        if not torch.is_tensor(embedding_weight):
            raise TypeError('tie_embeddings expects a torch.Tensor (e.g. nn.Embedding.weight)')
        if method not in ('mean', 'representative'):
            raise ValueError(f"method must be 'mean' or 'representative', got {method!r}")
        W = embedding_weight
        src_dtype = W.dtype
        with torch.no_grad():
            work = W.detach().float().clone()
            reps: Dict[Tuple[str, ...], int] = {}
            rows = 0
            for key, members in self.orbits.items():
                ids = [self.root2id[m] for m in members if m in self.root2id]
                if len(ids) < 2:
                    continue
                rep_id = self._representative_id(members, representative)
                if method == 'mean':
                    row = work[ids].mean(dim=0)
                else:
                    row = work[rep_id]
                for i in ids:
                    work[i] = row
                reps[key] = rep_id
                rows += len(ids)
            out = work.to(dtype or src_dtype)
            if inplace:
                W.copy_(out)
            result = {
                'orbits_tied': len(reps),
                'rows_tied': rows,
                'rows_collapsed': rows - len(reps),
                'method': method,
                'representatives': reps,
            }
        return result

    def tie_gradients(self, embedding_weight: Any, mode: str = 'sum_to_representative') -> int:
        """Re-tie gradients after `backward`, so a tied orbit really gets ONE parameter's
        update rather than k independent ones.  ENGINEERING: this is the textbook remedy for
        weight tying implemented by row duplication; call it between `backward()` and
        `optimizer.step()`, and re-apply `tie_embeddings` after the step."""
        if torch is None:  # pragma: no cover
            raise RuntimeError('tie_gradients requires torch')
        W = embedding_weight
        g = getattr(W, 'grad', None)
        if g is None:
            return 0
        n = 0
        with torch.no_grad():
            for members in self.orbits.values():
                ids = [self.root2id[m] for m in members if m in self.root2id]
                if len(ids) < 2:
                    continue
                if mode == 'sum_to_representative':
                    rep = self._representative_id(members)
                    g[rep] = g[ids].sum(dim=0)
                    for i in ids:
                        if i != rep:
                            g[i] = 0
                elif mode == 'mean_to_all':
                    mean = g[ids].mean(dim=0)
                    for i in ids:
                        g[i] = mean
                else:
                    raise ValueError(f'unknown mode {mode!r}')
                n += 1
        return n

    # ------------------------------------------------------------------ losses
    def pairs(self) -> List[Tuple[int, int]]:
        """Unordered id pairs inside every tieable orbit -- the pair list
        `nrmt_arch.orbit_consistency_loss` consumes."""
        out: List[Tuple[int, int]] = []
        for members in self.orbits.values():
            ids = sorted({self.root2id[m] for m in members if m in self.root2id})
            out.extend(itertools.combinations(ids, 2))
        return out

    def loss(self, embedding_weight: Any, mode: str = 'mse', normalize: bool = True,
             max_pairs: Optional[int] = 4096, generator: Any = None) -> Any:
        """Within-orbit dispersion: 0 when every orbit's members share one row.

        mode='mse'    -- mean over orbits of the members' mean squared deviation from the
                         orbit mean; divided by the mean squared row norm when `normalize`.
                         Scale-free, so it can be added to a token loss with a small weight.
                         Differentiable: gradients flow into `embedding_weight`.
        mode='cosine' -- mean(1 - cos(row_a, row_b)) over orbit pairs (random `max_pairs`
                         sample when there are more), matching the shape of the legacy
                         `nrmt_arch.RootformerNRMT.orbit_consistency_loss`.
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError('loss requires torch')
        if not torch.is_tensor(embedding_weight):
            raise TypeError('loss expects a torch.Tensor (e.g. nn.Embedding.weight)')
        W = embedding_weight
        if not self.orbits:
            return W.sum() * 0.0
        if mode == 'mse':
            terms = []
            for members in self.orbits.values():
                ids = torch.tensor([self.root2id[m] for m in members if m in self.root2id],
                                   device=W.device)
                if len(ids) < 2:
                    continue
                rows = W[ids].float()
                var = (rows - rows.mean(dim=0, keepdim=True)).pow(2).mean()
                if normalize:
                    var = var / (rows.pow(2).mean() + 1e-8)
                terms.append(var)
            if not terms:
                return W.sum() * 0.0
            return torch.stack(terms).mean()
        if mode == 'cosine':
            pairs = self.pairs()
            if not pairs:
                return W.sum() * 0.0
            idx = torch.arange(len(pairs))
            if max_pairs and len(pairs) > max_pairs:
                idx = torch.randint(0, len(pairs), (int(max_pairs),), generator=generator)
            a = torch.tensor([pairs[i][0] for i in idx.tolist()], device=W.device)
            b = torch.tensor([pairs[i][1] for i in idx.tolist()], device=W.device)
            ea = torch.nn.functional.normalize(W[a].float(), dim=-1)
            eb = torch.nn.functional.normalize(W[b].float(), dim=-1)
            return (1.0 - (ea * eb).sum(-1)).mean()
        raise ValueError(f"mode must be 'mse' or 'cosine', got {mode!r}")

    # ------------------------------------------------------------------ reporting
    def stats(self) -> Dict[str, Any]:
        sizes = Counter(len(v) for v in self.orbits.values())
        faces = Counter(len({normalize_root(m, self.fold_hamza) for m in v})
                        for v in self.orbits.values())
        return {
            'membership': self.membership,
            'membership_effective': self.membership_effective,
            'representative': self.representative,
            'lengths': list(self.lengths),
            'attest_source': self.attest_source,
            'attest_path': str(self.attest_path) if self.attest_path else None,
            'attest_record_counts': self.attest_counts,
            'attested_keys': len(self.attested),
            'muhmal_keys': len(self.muhmal),
            'inventory_roots_considered': self.n_inventory_considered,
            'inventory_roots_skipped': len(self._skipped),
            'keys': len(self._by_key),
            'orbits': len(self.orbits),
            'orbit_size_histogram': dict(sorted(sizes.items())),
            'orbit_face_size_histogram': dict(sorted(faces.items())),
            'isolated_roots': len(self.isolated),
            'tied_rows': self.n_tied_rows,
            'tied_faces': self.n_faces,
            'duplicate_hamza_rows': self.n_duplicate_rows,
            'rows_collapsed': self.n_distinct_rows,
            'muhmal_excluded_roots': sum(len(v) for v in self._excluded_muhmal.values()),
            'unattested_excluded_roots': sum(len(v) for v in self._excluded_unattested.values()),
            'pairs': len(self.pairs()),
        }

    def report(self) -> str:
        s = self.stats()
        lines = [
            'KhalilPermutationOrbits -- التقليبات as parameter tying (NOT al-ishtiqāq al-akbar;',
            '  the semantic claim is rejected by the tradition: «غير مأخوذ به؛ لعدم اطراده»)',
            f"  membership        : {s['membership']} (effective: {s['membership_effective']})",
            f"  attest record     : {s['attest_source']} {s['attest_path'] or ''}",
            f"  lengths           : {s['lengths']}",
            f"  inventory roots   : {s['inventory_roots_considered']} considered, "
            f"{s['inventory_roots_skipped']} skipped (not radical strings)",
            f"  permutation keys  : {s['keys']}",
            f"  tieable orbits    : {s['orbits']}  faces={s['orbit_face_size_histogram']}",
            f"  isolated roots    : {s['isolated_roots']}",
            f"  tied rows         : {s['tied_rows']} rows / {s['tied_faces']} faces -> "
            f"{s['orbits']} orbits ({s['duplicate_hamza_rows']} duplicate hamza-spelling rows)",
            f"  al-ʿAyn مهمل kept out : {s['muhmal_excluded_roots']} inventory roots",
            f"  unattested kept out   : {s['unattested_excluded_roots']} inventory roots",
            f"  internal id pairs     : {s['pairs']}",
        ]
        return '\n'.join(lines)


class OrbitTiedEmbedding:
    """True storage sharing: ONE trainable row per al-Khalīl orbit (`weight` is
    `n_orbits x d`), gathered by an index map at `forward`.

    *** CHECKPOINT WARNING -- READ BEFORE USE ***
    This changes the state_dict: `weight` has one row per ORBIT, not one per root, and an
    extra `orbit_index` buffer appears.  Any existing checkpoint that stores
    `morphemic_embed.root_embed.weight` with `num_roots x d_root` rows (e.g.
    `checkpoints/rootformer_v19_2_synthesis_*.safetensors`) will FAIL to load into it.
    That is why this class is opt-in and wired into nothing: the shape-preserving
    `KhalilPermutationOrbits.tie_embeddings` is the checkpoint-safe mechanism.  Converting a
    released checkpoint requires re-indexing its embedding rows into orbit representatives and
    is deliberately NOT done here.
    """

    def __init__(self, orbits: KhalilPermutationOrbits, num_roots: int, d_model: int,
                 dtype: Any = None, device: Any = None):
        if torch is None:  # pragma: no cover
            raise RuntimeError('OrbitTiedEmbedding requires torch')
        self.orbits = orbits
        self.num_roots = int(num_roots)
        self.d_model = int(d_model)
        reps = sorted({orbits._representative_id(m) for m in orbits.orbits.values()})
        self.rep_ids: List[int] = reps
        pos = {r: i for i, r in enumerate(reps)}
        idx = list(range(self.num_roots))
        for members in orbits.orbits.values():
            p = pos[orbits._representative_id(members)]
            for m in members:
                i = orbits.root2id.get(m)
                if i is not None:
                    idx[i] = p
        self.weight = torch.nn.Parameter(
            torch.zeros(len(reps), self.d_model, dtype=dtype or torch.float32, device=device))
        self.register_index = torch.tensor(idx, dtype=torch.long)
        self.state_dict_shapes = {'weight': (len(reps), self.d_model),
                                  'orbit_index': (self.num_roots,)}

    def forward(self, root_ids: Any) -> Any:
        return self.weight[self.register_index.to(root_ids.device)][root_ids]

    def state_dict_keys(self) -> List[str]:
        """The keys a checkpoint WOULD have to carry (none of which match the released ones)."""
        return ['weight', 'orbit_index']


def tie_root_embedding_tables(model_or_state: Any, orbits: KhalilPermutationOrbits,
                              method: str = 'mean') -> Dict[str, Any]:
    """Tie every root-embedding table found in an `nn.Module` or a state_dict mapping.

    ENGINEERING: the shipped checkpoints carry the root tables as
    `backbone.layers.<N>.self_attn.root_embed.weight` (24 of them in
    `checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors`), *not* under the
    `morphemic_embed.root_embed.weight` name the NRMT head uses -- so tying has to be applied
    where the tables actually are.  This walks the object, matches names containing
    `root_embed` and ending in `weight`, and calls `orbits.tie_embeddings` on each.

    CHECKPOINT CONFLICT -- REPORTED, NOT FORCED.  A table whose row count is not
    `orbits.num_roots` is SKIPPED and listed under 'skipped' instead of being resized: row
    counts are part of the checkpoint's shape contract.  This matters here -- the shipped
    ar_backbone checkpoint's tables are [9015, 64] while the current vocabulary reports 9,114
    roots, a pre-existing blueprint/checkpoint mismatch that this function does not touch
    (remapped ids must be settled before any id-indexed tying is meaningful).
    """
    items: Iterable[Tuple[str, Any]]
    if hasattr(model_or_state, 'named_parameters'):
        items = list(model_or_state.named_parameters())
    elif hasattr(model_or_state, 'items'):
        items = list(model_or_state.items())
    else:
        raise TypeError('expected an nn.Module or a mapping of name -> tensor')
    tied: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}
    for name, tensor in items:
        if 'root_embed' not in name or not name.endswith('weight'):
            continue
        if torch is None or not torch.is_tensor(tensor):  # pragma: no cover
            skipped[name] = 'not a tensor'
            continue
        if tensor.shape[0] != orbits.num_roots:
            skipped[name] = (f'rows={int(tensor.shape[0])} != vocab num_roots={orbits.num_roots}'
                             ' -- left untouched (shape is part of the checkpoint contract)')
            continue
        tied[name] = orbits.tie_embeddings(tensor, method=method)
    return {'tied': tied, 'skipped': skipped,
            'n_tied_tables': len(tied), 'n_skipped_tables': len(skipped)}


def build_from_release(release_dir: Union[str, Path] = MODULE_DIR, **kwargs: Any
                       ) -> KhalilPermutationOrbits:
    """ENGINEERING convenience: build over the shipped vocabulary and al-ʿAyn record.

    Reads `data/rootformer_v12_arabic_blueprint.json` through `nrmp_vocab.MorphemicVocab`
    when that is importable, else the blueprint's root partition directly.
    """
    release = Path(release_dir)
    blueprint = release / 'data' / 'rootformer_v12_arabic_blueprint.json'
    attest = release / 'data' / 'khalil_attest_v4.json'
    roots: List[str] = []
    try:  # the vocabulary is the authority on root ids
        import sys
        for p in (str(release), str(release / 'models')):
            if p not in sys.path:
                sys.path.insert(0, p)
        import nrmp_vocab as nv  # type: ignore
        cls = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
        vocab = cls(str(blueprint))
        return KhalilPermutationOrbits(vocab, attest_path=attest, **kwargs)
    except Exception:
        data = json.loads(blueprint.read_text(encoding='utf-8'))
        parts = data.get('partitions', {}) or {}
        part = parts.get('classical_roots') or parts.get('root') or parts.get('roots')
        id2tok = data.get('vocab_id_to_token', {}) or {}
        if isinstance(part, (list, tuple)) and len(part) == 2 and all(
                isinstance(x, int) for x in part):
            roots = [id2tok[str(i)] for i in range(int(part[0]), int(part[1]))
                     if str(i) in id2tok]
        elif isinstance(part, dict):
            roots = list(part.keys())
        elif isinstance(part, (list, tuple)):
            roots = [p.get('token', p) if isinstance(p, dict) else p for p in part]
        else:
            roots = [t for i, t in sorted(id2tok.items(), key=lambda kv: int(kv[0]))]
        return KhalilPermutationOrbits(roots, attest_path=attest, **kwargs)


if __name__ == '__main__':
    import sys
    rel = sys.argv[1] if len(sys.argv) > 1 else str(MODULE_DIR)
    for membership in ('attested', 'inventory_not_muhmal'):
        o = build_from_release(rel, membership=membership, verbose=True)
        print(o.report())
        print('  orbit_of(علم)   :', o.orbit_of('علم'))
        print('  orbit_of(عبد)   :', o.orbit_of('عبد'))
        print('  excluded(عبد)   :', {k: v for k, v in o.excluded('عبد').items() if v})
        print()
