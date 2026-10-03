"""
validated_segmentation.py -- al-Khalil-validated segmentation, as a drop-in for the shipped
tokenizer's decompose_arabic_word().

WHY THIS EXISTS
---------------
The classical algorithms were implemented and verified, but the pipeline never called them: the
tokenizer read roots off the surface and accepted any residue that happened to be in the 9,013-root
lexicon.  That lexicon contains نتت، كمم، عنن، so membership could not discriminate, and the
measured result on the shipped tree was that a 26x larger al-'Ayn attestation record changed
NOTHING (see KHALIL_ATTEST_V4_FINDINGS.md).

This module puts the validated analysis on the path the model actually runs.  It is imported by
models/morphemic_tokenizer_v12_arabic.py, which routes decompose_arabic_word() through it.

WHAT IT DOES
------------
Given the tokenizer's greedy reading, it re-segments only on POSITIVE evidence, using:

  1. al-'Ayn's record, RANKED above the blueprint lexicon (ATTESTED +8, LEXICON +3, MUHMAL -12);
  2. al-Khalil's root CALCULATION, which reaches roots whose radical was elided (كنت -> كون);
  3. the muda''af pair gate (نتت is the doubling of the pair نت, which al-'Ayn marks muhmal);
  4. elision constrained by the ajwaf / naqis division (only a NON-final weak letter may vanish).

MODES
-----
  ROOTFORMER_VALIDATED_SEG=0   off -- the greedy reading, byte-identical to the previous tokenizer
  ROOTFORMER_VALIDATED_SEG=1   DEFAULT.  The closed class only: a word consumed by clitics plus a
                               particle has no root (عليه -> على + ه، فكم -> ف + كم، منها -> من + ها).
                               Measured: 1,103 root changes in 20,000 corpus words (5.51%), 40/40 of
                               a hand-checked sample correct, 48,381 words/s (1.5x the greedy cost).
  ROOTFORMER_VALIDATED_SEG=2   also re-assign ROOTS via al-Khalil's calculation.  Measured: 15.03%
                               of roots change, ~24 of a 40-word sample correct -- good enough to
                               keep working on, NOT good enough to enable.  Needs the gold set.

  ROOTFORMER_ATTEST_RECORD=<path>   use a different al-'Ayn record

NOTE FOR TRAINING
-----------------
This changes the token stream.  /workspace/nrmp_cache/*.pt was built with the greedy analyzer, so
any model trained on it must be rebuilt before its metrics mean anything.
"""

import re
import sys

import os
from pathlib import Path

# the release root holds tasrif_engine.py; models/ holds this file
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE.parent, _HERE):
    if str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))


# --------------------------------------------------------------------------------------------
# VISIBILITY (pillar 3).  Every handler below that used to swallow an exception now reports it
# on stderr.  The RETURN VALUE of each handler is unchanged -- the validator is on the shipped
# decode path and must keep degrading to the greedy reading -- but a degraded validation can no
# longer look like a successful one.  `_warn_once` rate-limits the per-candidate sites
# (_generates/_align/_ops/_kathra are called 10^3-10^5 times per run) so the fix cannot turn a
# latent bug into a log flood; the first failure is always reported.
# --------------------------------------------------------------------------------------------
_WARNED_SITES = set()


def _warn_once(site, message):
    if site not in _WARNED_SITES:
        _WARNED_SITES.add(site)
        print(f'[validated_segmentation] {message}', file=sys.stderr)


DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')

# al-huruf al-zawa'id, the mnemonic sa-altumuniha
# Clitics only.  The mudari'a and derivational letters (أ ي ن ت م س، مست، يست) belong to the
# WAZN, not to the clitic layer: stripping them tore words apart -- ألف -> ليف, أدخ -> دوخ,
# أصل -> صول, تكل -> كول.  Anything the patterns already own is not a prefix.
PREFIXES = ['وال', 'فال', 'بال', 'كال', 'لل', 'ال', 'و', 'ف', 'ب', 'ك', 'ل']
SUFFIXES = ['هما', 'كما', 'هم', 'هن', 'كم', 'كن', 'نا', 'ني', 'ها', 'وا', 'ون', 'ين', 'ات',
            'ان', 'ية', 'تم', 'تن', 'ت', 'ه', 'ك', 'ي', 'ا', 'ن']
# THE DIVINE NAME IS LEFT ALONE -- on instruction, not on a grammatical claim.
#
# I had justified this exemption by saying a proper name has no root.  That justification is wrong:
# Sibawayh derives it («وكأن الاسم والله أعلم إله، فلما أدخل فيه الألف واللام حذفوا الألف وصارت
# الألف واللام خلفا منها») and Ibn Faris gives the root ل-ا-ه.  The exemption is therefore kept
# as a DELIBERATE product decision -- the word is not re-segmented -- and NOT as fidelity to the
# sources, which would analyse it.  Nothing downstream should read this as a claim about اللّه.
INDECLINABLE_NAMES = {'الله', 'اللهم', 'إله', 'اله', 'لله'}
# a preposition or conjunction may precede the name: بالله، والله، فالله، تالله
NAME_WORD = re.compile(r'^(?:و|ف|ب|ك|ل|ت|ال|بال|وال|فال|كال|لل|تال)?(?:ا?لله|اللهم|ا?له)$')

# --------------------------------------------------------------------------------------------
# THE CLOSED CLASS, checked BEFORE any root is contemplated.
#
# al-Khalil's first division is harf / ism / fi'l, and a harf has no root.  Skipping this check
# made the validator invent roots for function words on real text -- it read عليه as root علو,
# منها as root نها, فلا as root فلو, فلما as root فلم.  That is the same class of error the
# whole exercise exists to remove, and it showed up on 18.9% of corpus words.
#
# A word is closed-class when it is consumed entirely by: an optional leading conjunction or
# preposition, a particle, and an optional pronominal clitic -- on على + ه -> عليه, where the
# particle's final alif is dropped before the pronoun.
# --------------------------------------------------------------------------------------------
# wazn names whose leading letter is literal (يَفْعُلُ, تَفَعَّلَ, أَفْعَلَ, نَفْعَلُ),
# not one of the ف/ع/ل placeholders
MUdARIA_LITERAL = set('يتأن')

PRON_SUFFIXES = ['هما', 'كما', 'هن', 'هم', 'كم', 'كن', 'نا', 'ني', 'ها', 'ه', 'ك', 'ي']
PRE_CLOSED = ['وال', 'فال', 'بال', 'كال', 'لل', 'ال', 'و', 'ف', 'ب', 'ك', 'ل']
# PREPOSITIONS that cannot stand before a bare particle.  و and ف are CONJUNCTIONS and
# legitimately precede one (فلم، فلا، ومن، وفي) -- including them here rejected exactly the
# function words this rule exists to catch.
SINGLE_PREPS = {'ب', 'ك', 'ل', 'س'}
# SUPPLEMENT ONLY.  The authoritative closed inventory is the blueprint's own SIBawayhian
# partition -- 46 category tokens plus 65 bare particles at ids 117..228, which the project built
# from the sources.  The set below is a hand-written ADDITION of items that partition does not
# carry (the demonstrative duals, كذا/هكذا, the relative duals, some adverbs).
#
# WHY THESE ITEMS ARE CLOSED AT ALL -- and the limit of the claim.  The CATALOGUING is sourced:
# Ibn Malik ranks the ma'arif «وجملة المعارف سبعة: المضمر، والعلم، واسم الإشارة، والموصول،
# والمعرف بالأداة...»; Sibawayh lists «الأسماء المبهمة فنحو هذا وهذه، وهذان وهاتان، وهؤلاء، وذلك
# وتلك ... وأولئك»; Ibn Jinni says the demonstratives and relatives «جارية مجرى الأسماء المضمرة».
# But "they are not derived from roots" is NOT stated for them -- the explicit «غير مشتق» is only
# for PRONOUNS («جميع الأسماء المضمرة مبني غير مشتق»), and Ibn Sayyidih even gives الذي radicals:
# «الأصول من الذي ثلاثة أحرف لام وذال وياء».  So this list asserts closed-class membership, which
# is sourced, and nothing about roots, which would not be.
SUPPLEMENT_CLOSED = {
    'من', 'في', 'على', 'إلى', 'الى', 'عن', 'حتى', 'مع', 'إن', 'ان', 'أن', 'كأن', 'لكن', 'ليت',
    'لعل', 'لا', 'ما', 'لم', 'لن', 'هل', 'قد', 'ثم', 'أو', 'او', 'أم', 'ام', 'بل', 'كم', 'كيف',
    'أين', 'اين', 'متى', 'إذا', 'اذا', 'إذ', 'اذ', 'لو', 'لولا', 'هذا', 'هذه', 'ذلك', 'تلك',
    'الذي', 'التي', 'هو', 'هي', 'هما', 'هم', 'هن', 'أنا', 'انا', 'نحن', 'أنت', 'انت', 'لما',
    'كل', 'بعض', 'غير', 'سوف', 'كي', 'لكي', 'إلا', 'الا', 'حيث', 'منذ', 'عند', 'لدى', 'بين',
    'فوق', 'تحت', 'خلف', 'أمام', 'امام', 'ضد', 'نحو', 'هنا', 'هناك', 'الآن', 'الان', 'نعم',
    'بلى', 'أي', 'اي', 'أيا', 'إما', 'اما', 'أما', 'كلما', 'بينما', 'حين', 'حينما', 'عندما',
    'كذا', 'هكذا', 'كذلك', 'هؤلاء', 'اولئك', 'أولئك', 'هذان', 'هاتان', 'هذين', 'هاتين',
    'الذين', 'اللذان', 'اللتان', 'اللاتي', 'اللواتي', 'هؤلاء', 'هنالك',
}


PARTICLES = {'من', 'في', 'إلى', 'على', 'عن', 'حتى', 'مع', 'إن', 'أن', 'كأن', 'لكن', 'ليت',
             'لعل', 'لا', 'ما', 'لم', 'لن', 'هل', 'قد', 'ثم', 'أو', 'أم', 'بل', 'كم', 'كيف',
             'أين', 'متى', 'إذا', 'إذ', 'لو', 'لولا', 'هذا', 'هذه', 'ذلك', 'تلك', 'الذي',
             'التي', 'هو', 'هي', 'أنا', 'نحن', 'أنت', 'هم', 'هن'}


def strip_diac(s):
    return DIAC.sub('', s)


class ValidatedSegmentation:
    """Validate the tokenizer's segmentation against al-Khalil."""

    def __init__(self, tok, verbose=False):
        self.tok = tok
        self.verbose = verbose
        try:
            from tasrif_engine import TasrifEngine
            self.engine = TasrifEngine(tok)
        except Exception as exc:
            # VISIBILITY: with no engine, decompose() returns the greedy reading untouched, so a
            # broken tasrif import silently turns the whole validator into a no-op.  Default
            # (engine = None) is unchanged.
            _warn_once('engine', f'tasrif_engine unavailable -- validation is INERT: {exc!r}')
            self.engine = None
        # THE MISSING CONSTRAINT: the engine generates a form for ANY triliteral root, so
        # "round-trips" alone is satisfied by nonsense roots (kuntu -> ka + n-t-t). The residue
        # must also be a REAL root. al-Khalil's attestation table supplies that.
        # THE EVIDENCE BASE is the blueprint's OWN root lexicon (9,013 roots). al-Khalil's
        # attestation table is only 2,483 entries -- too partial to confirm a root, though it can
        # still CONDEMN one. So: lexicon for positive evidence, al-'Ayn for negative.
        self.lexicon = {strip_diac(r) for r in getattr(tok, 'roots_set', set())}
        self.attested, self.unused, self.unused_pairs = set(), set(), set()
        try:
            import json
            d = json.load(open(_record_path()))
            # all lengths: the 2-letter entries are the PAIR record, which governs the muda''af
            self.attested = {tuple(t) for t in d.get('attested', [])}
            self.unused = {tuple(t) for t in d.get('unused', [])}
            self.unused_pairs = {t for t in self.unused if len(t) == 2}
        except Exception as exc:
            # VISIBILITY: this is the al-'Ayn evidence base.  Swallowing the failure left
            # attested/unused EMPTY, which silently strips the MUHMAL penalty and the pair gate
            # out of _real_root() -- the validator then still reports success on every word.
            # Defaults (three empty sets) are unchanged.
            _warn_once('attest_record',
                       f'al-Ayn attestation record {_record_path()!r} unreadable: {exc!r} -- '
                       f'attested/unused evidence is EMPTY, validation is degraded')
        self.stats = {'calls': 0, 'changed': 0, 'closed_class': 0, 'root_reseg': 0}
        # 1 = closed class only (the measured-safe half); 2 = also root re-segmentation
        self.mode = MODES.get(os.environ.get('ROOTFORMER_VALIDATED_SEG', '1').lower(), 0)
        # al-kathra table, if built.  Absent -> _kathra() returns 0 and the ranking degrades to
        # the pre-kathra order rather than breaking.
        self._kathra_tbl = None
        if self.mode >= 2:
            try:
                import kathra as _K
                # BUCKET=10 by default: raw counts are noisy at small n (66 vs 64 decided يحج) and
                # the subagent measured that bucketing keeps 29 of 30 genuine fixes while dropping
                # 11 of 21 errors.  Set ROOTFORMER_KATHRA_BUCKET=1 for the raw, strictly
                # source-grounded criterion.
                _b = int(os.environ.get('ROOTFORMER_KATHRA_BUCKET', '10'))
                self._kathra_tbl = _K.default(bucket=_b)
                if self._kathra_tbl is not None:
                    self._kathra_tbl.bucket = _b      # default() caches; bucketing is per-run
            except Exception:
                self._kathra_tbl = None
        # the tokenizer's own closed-class inventory, if it exposes one.
        # DELIBERATE FALLBACK, kept: models/morphemic_tokenizer_v12_arabic imports THIS module,
        # so this back-import is circular and must never break module load.  If it is
        # unavailable, the blueprint's SIBawayhian partition below is still the authoritative
        # inventory, so the closed class degrades predictably instead of failing.
        self._closed = set(SUPPLEMENT_CLOSED)
        try:
            from morphemic_tokenizer_v12_arabic import CLOSED_PARTICLES as _CP
        except Exception:
            try:
                from models.morphemic_tokenizer_v12_arabic import CLOSED_PARTICLES as _CP
            except Exception:
                _CP = ()
        try:  # purely defensive: _CP is a container; do not lose the supplement on a bad union
            self._closed |= set(_CP)
        except Exception:
            pass
        # 1. THE SOURCED INVENTORY: the blueprint's SIBawayhian particle partition.
        self.stats['closed_supplement'] = len(self._closed)
        try:
            import json as _json
            _bp = _json.load(open(getattr(self.tok, 'blueprint_path', ''), encoding='utf-8'))
            _i2t = _bp['vocab_id_to_token']
            _lo, _hi = _bp['partitions']['sibawayh_particles']
            _bare = [_i2t[str(i)] for i in range(_lo, _hi)]
            self._closed |= {p for p in _bare if not p.startswith('<')}
            self.stats['closed_from_blueprint'] = len([p for p in _bare if not p.startswith('<')])
        except Exception as exc:
            # VISIBILITY: a failed partition read shrank the closed-class inventory silently,
            # which changes which words are re-segmented.  Default (0 read) is unchanged.
            _warn_once('blueprint_particles',
                       f'blueprint particle partition unreadable ({exc!r}) -- the closed-class '
                       f'inventory is only the hand-written supplement')

    # -- delegate everything else ----------------------------------------------------------
    def __getattr__(self, k):
        return getattr(self.tok, k)

    def _generates(self, stem, root, wazn):
        """Does (root, wazn) generate this stem -- allowing the ELIDED reading?

        كنت is كان + -tu.  The surface stem is كن, because the alif of كان is deleted before the
        suffix (al-hadhf).  generate(كون, فَعَلَ) returns كان: the stem with one weak letter
        elided.  Exact equality therefore REJECTS the correct root, which is why every earlier
        version of this analyzer could not reach كون and settled for the invented نتت instead.
        The elision is the rule tasrif_engine already models, so it is admitted here: the stem
        must be the generated form with zero or more weak letters deleted, order preserved.
        """
        if not (self.engine and root and wazn):
            return False
        try:
            g = self.engine.generate(root, wazn)
        except Exception as exc:
            # VISIBILITY: tasrif_engine.generate returns None for an impossible (root, wazn); it
            # RAISING means an engine/tokenizer bug.  Swallowing it scored every affected
            # candidate as "does not generate", i.e. silently wrong analyses.  Still False.
            _warn_once('generate', f'tasrif generate({root!r}, {wazn!r}) raised: {exc!r}')
            return False
        if not g:
            return False
        gen, tgt = strip_diac(g[0]), strip_diac(stem)
        if gen == tgt:
            return True
        # WHICH weak letter may vanish is not a free choice -- it is the ajwaf / naqis division,
        # and the sources state it directly:
        #   AL-AJwaf  the MEDIAL radical is weak and IS DELETED before a pronoun/suffix.
        #     Ibn Jinni, al-Khasa'is: «لما سكنت عين فعلت ولامه حذفوا العين البتة فقالوا: قلت
        #     وبعت وخفت، ولم يقولوا: قولت ولا بيعت ولا خيفت»
        #     Ibn 'Usfur, al-Mumti': «وتحذف العين لالتقاء الساكنين ... فتقول: خفت وكدت وطلت»
        #   AL-NAQIS  the FINAL radical is weak and is RESTORED in the same position.
        #     Ibn 'Usfur, al-Mumti': «وإن أسند إلى ضمير متكلم أو مخاطب ... رددت الألف إلى أصلها
        #     من الياء أو الواو، نحو: رميت وغزوت»
        # Skipping any weak letter let كني win the tie for كنت, because deleting the final ى from
        # كنى also yields كن.  Only a NON-FINAL weak letter may be elided here.
        #
        # Two honest limits.  (a) The sources state the contrast for THIS position only: the naqis
        # does delete before the ta' of femininity -- «وإن كان لامه ألفا حذفت لالتقاء الساكنين،
        # نحو: رمت هند» (al-Mumti').  (b) كنت itself is never used as an ajwaf exemplar; the
        # analysis is Ibn Jinni's, in Sirr Sina'at al-I'rab: «وكذا كان القياس أن تقول في كنت:
        # كوني، تحذف التاء ... فترد الواو التي هي عين الفعل من كنت» -- which is why the waw is
        # restored rather than the surface taken at face value.
        i = 0
        for pos, ch in enumerate(gen):
            if i < len(tgt) and tgt[i] == ch:
                i += 1
            elif ch in 'اويىء' and pos != len(gen) - 1:
                continue                      # medial elision / transposition (al-hadhf, al-qalb)
            else:
                return False
        return i == len(tgt)

    def _candidate_roots(self, stem, wazn):
        """al-Khalil's CALCULATION, not a lookup.

        The tokenizer only ever proposes roots it can read off the surface, so a root whose
        radical was DELETED is unreachable: no amount of lexicon filtering finds كون in كنت.
        candidate_pairs() walks the wazn template against the surface and is allowed to spend a
        slot on a radical that left no letter behind (HADHF), then validates by generating --
        which is the generator run backwards, exactly as al-Khalil's taqalib + attestation is.
        """
        out = []
        if self.engine is None:
            return out
        try:
            from tasrif_engine import Pattern
        except Exception as exc:
            # VISIBILITY: without Pattern the candidate inventory is empty, so root
            # re-segmentation silently cannot happen at all.  Default ([]) is unchanged.
            _warn_once('pattern_import', f'tasrif Pattern unavailable: {exc!r}')
            return out
        pats = [wazn] if wazn else []
        for p in ('فَعَلَ', 'فَعِلَ', 'فَعُلَ'):
            if p not in pats:
                pats.append(p)
        for p in pats:
            if not isinstance(p, str):
                continue
            try:
                roots = self.engine._align(strip_diac(stem), Pattern(p))
            except Exception as exc:
                # VISIBILITY: an align failure silently removed ONE candidate pattern from the
                # search, which changes the chosen reading with no trace.  Still skipped.
                _warn_once('align', f'tasrif _align({stem!r}, {p!r}) raised: {exc!r}')
                continue
            # VALIDATE HERE, LOOSELY.  candidate_pairs() would have rejected كون for كنت: it
            # requires generate(root,wazn) to equal the stem exactly, and generate(كون,فَعَلَ) is
            # كان -- the stem with the alif elided.  The elision is the rule being modelled, so
            # the validator must admit it, and _generates() does.
            for r in roots:
                if (r, p) not in out and self._generates(stem, r, p):
                    out.append((r, p))
        return out

    # evidence strengths: al-'Ayn's own entry beats the blueprint lexicon, which is what finally
    # separates كون (attested) from كنن (lexicon only) and كم (attested particle) from كمم.
    MUHMAL, UNKNOWN, LEXICON, ATTESTED = -1, 0, 1, 2

    def _real_root(self, root):
        if not root:
            return self.UNKNOWN
        r = strip_diac(root)
        if len(r) != 3:
            return self.UNKNOWN
        rt = tuple(r)
        if rt in self.unused:
            return self.MUHMAL               # al-Khalil marks this permutation unused
        if rt in self.attested:
            return self.ATTESTED            # al-'Ayn enters it -- outranks the lexicon
        # THE MUDA''AF GATE.  A doubled root C1 C2 C2 IS the doubling of the ordered pair
        # (C1,C2), and al-'Ayn's two-letter chapters enumerate exactly which pairs are used:
        #   باب التاء والنون  ت ن  يستعمل فقط تن:      -> the pair ن ت is M U H M A L
        # so نتت -- the tokenizer's doubling of the stem نت -- is not a root of the language,
        # even though the blueprint's 9,013-root lexicon lists it.
        # The gate is applied ONLY to C2==C3.  It must not touch C1==C3 roots: al-'Ayn enters
        #    نتن inside باب التاء والنون itself («نتن ينتن نتنا ... وهذه المادة من الثلاثي»),
        # so the pair record plainly does not govern them, and reducing them to their skeleton
        # wrongly condemned نتن، ثلث، and the quadriliterals قنقل، قرقل، لغلغ.
        if r[1] == r[2] and (r[0], r[1]) in self.unused_pairs:
            return self.MUHMAL
        if r in self.lexicon:
            return self.LEXICON           # in the blueprint lexicon, but nothing else vouches
        return self.UNKNOWN

    # ============================================================================================
    # THE DECISION PROCEDURE -- lexicographic, over cited principles.  NOT a weighted sum.
    #
    # I had been ranking candidate analyses with invented integers (+10, +8, -6, -12).  The
    # tradition does not work that way; it decides by named principles, and Ibn 'Usfur al-Ishbili
    # (al-Mumti' fi al-Tasrif) even gives the list of criteria for the hardest case -- is a letter
    # ZA'ID or ASLI:
    #
    #   «أما الأدلة التي يعرف بها الزائد من الأصلي فهي: الاشتقاق، والتصريف، والكثرة، واللزوم،
    #    ولزوم حرف الزيادة البناء وكون الزيادة لمعنى، والنظير، والخروج عن النظير، والدخول في
    #    أوسع البابين عند لزوم الخروج عن النظير.»
    #
    # The order below follows the tradition's own weighting, most decisive first:
    #
    #   1. A harf is not a root at all.          Sibawayh: «الكلم: اسم، وفعل، وحرف»
    #                                            Ibn Jinni: «الحروف يشتق منها ولا تشتق هي أبدا»
    #   2. MUSTA'MAL beats MUHMAL.               al-'Ayn, introduction: «يُكتب مُستعملها ويُلغى
    #                                            مُهملها» -- a hard accept/reject filter
    #   3. AL-HAML 'ALA AL-AKTHAR.               al-Shatibi: «الثابت في الأصول أن الكثرة دليل
    #                                            الأصالة»; «والحمل على الأكثر واجب»
    #                                            Ibn Jinni: «والحكم على الأكثر لا على الأقل»
    #   4. AL-ASL 'ADAM AL-ZIYADA.               al-Shatibi: «والأصل عدم الزيادة، فمن ادعاها
    #                                            فعليه الدليل» -- fewer affixes wins
    #   5. AL-HAML 'ALA AL-ASL.                  Ibn Jinni: «ومتى أمكن تناول الكلمة على ظاهرها
    #                                            لم يجز العدول عن ذلك بها» -- fewer phonological
    #                                            operations wins; Abu Hayyan: «إذا دار الأمر إلى
    #                                            حذف، أو إلى الرد إلى أصلين كان الرد أولى»
    #   6. WIDER PATTERN.                        Ibn 'Usfur's tie-breaker: «الدخول في أوسع
    #                                            البابين عند لزوم الخروج عن النظير»
    #
    # CAVEAT the sources themselves insist on: al-asl is a tie-breaker, NOT absolute.  Ibn Jinni,
    # Sirr: «هذا أصل وإن قامت الدلالة عليه فإنه مرفوض، كما أن أصل قام: قوم، ولكنه لا ينطق به على
    # أصله» -- evidence (دليل) overrides it, which is why attestation ranks first.
    # ============================================================================================
    PARTICLE_BONUS = 8        # kept only for the legacy _score(); unused by _rank()

    def _ops(self, stem, root, wazn):
        """How many weak letters the reading must ELIDE (0 = the surface read at face value).

        This is the quantity al-haml 'ala al-asl minimises.  Returns a large sentinel when the
        reading cannot generate the stem at all, so it can never win.
        """
        if not (self.engine and root and wazn):
            return 99
        try:
            g = self.engine.generate(root, wazn)
        except Exception as exc:
            # VISIBILITY: the elision count is the al-haml 'ala al-asl tie-break; a swallowed
            # engine failure made the reading look like it "cannot generate" (99) and lose every
            # tie silently.  Sentinel 99 is unchanged.
            _warn_once('ops', f'tasrif generate({root!r}, {wazn!r}) raised in _ops: {exc!r}')
            return 99
        if not g:
            return 99
        gen, tgt = strip_diac(g[0]), strip_diac(stem)
        if gen == tgt:
            return 0
        i = skipped = 0
        for pos, ch in enumerate(gen):
            if i < len(tgt) and tgt[i] == ch:
                i += 1
            elif ch in 'اويىء' and pos != len(gen) - 1:
                skipped += 1
            else:
                return 99
        return skipped if i == len(tgt) else 99

    def _rank(self, prefix, stem, suffix, root, wazn, is_particle):
        """The classical decision procedure as an ordered tuple.  Higher wins; compare
        lexicographically.  Every position cites a principle -- see the block above."""
        strength = self._real_root(root)
        affixes = len(prefix or '') + len(suffix or '')
        ops = self._ops(stem, root, wazn)
        # THE ORDER IS IBN 'USFUR'S OWN, and that matters.  His list begins «الاشتقاق، والتصريف،
        # والكثرة...» -- DERIVATION AND INFLECTIONAL BEHAVIOUR come before frequency.  I had put
        # attestation first, which made يشذ (root شذذ, which generates the surface exactly) lose
        # to شوذ (an attested root that does not generate it).  Being attested says a root exists;
        # it does not say it is the root OF THIS WORD.  الفرق بين الوجود والاشتقاق.
        return (
            1 if is_particle else 0,                          # 1. harf is not a root (Sibawayh)
            0 if strength == self.MUHMAL else 1,              # 2. muhmal rejected (al-'Ayn)
            1 if ops != 99 else 0,                            # 3. al-tasrif GATE: the reading must
                                                               #    generate the surface at all
            self._kathra(prefix, stem, suffix, root, wazn),    # 4. AL-HAML 'ALA AL-AKTHAR
                                                               #    al-Shatibi «والحمل على الأكثر
                                                               #    واجب»; «الكثرة دليل الأصالة»
            -ops,                                              # 5. fewest elisions (Ibn Jinni,
                                                               #    al-haml 'ala al-asl) -- a tie-break
            1 if strength == self.ATTESTED else 0,            # 6. attested: existence, not use --
                                                               #    the weakest positive voice
            -affixes,                                          # 7. al-asl 'adam al-ziyada (al-Shatibi)
        )

    def _kathra(self, prefix, stem, suffix, root, wazn):
        """al-kathra (الكثرة) -- how often this reading's surface actually occurs.

        The sources make frequency BINDING, not advisory: al-Shatibi «والحمل على الأكثر واجب» and
        «والحمل على الأكثر متعين»; Ibn Jinni «والحكم على الأكثر لا على الأقل».  A binary
        musta'mal/muhmal record cannot express it, and could not separate كنت (kuntu, كان+ت) from
        كَنْتُ (kann-tu, كنّ+ت) -- both generate the surface with no elision.  Counted, they are
        64,151 to 726.

        Returns 0 when the module or its table is absent, so this degrades to the previous ranking
        rather than failing.  The BUCKET is engineering, not fidelity -- see kathra.py E7.
        """
        if self._kathra_tbl is None:
            return 0
        try:
            return self._kathra_tbl.reading_count(root, wazn, self.engine,
                                                  prefix=prefix or '', stem=stem,
                                                  suffix=suffix or '')
        except Exception as exc:
            # VISIBILITY: al-kathra is BINDING on the ranking (al-Shatibi); a silent 0 here
            # silently re-ordered the candidate readings.  Default 0 is unchanged.
            _warn_once('kathra', f'kathra.reading_count({root!r}, {wazn!r}) raised: {exc!r}')
            return 0

    def _score(self, prefix, stem, suffix, root, wazn):
        is_particle = strip_diac(stem) in PARTICLES
        if is_particle:
            # A recognised particle IS the analysis.  Its letters are not radicals, so no root
            # may be read off it -- awarding the particle bonus AND a root bonus on the same stem
            # is what let the doubled كمم (score 23) outbid the particle كم (score 7).
            return self.PARTICLE_BONUS - len(prefix) - len(suffix)
        sc = 0
        if self._generates(stem, root, wazn):
            sc += 10                      # the residue round-trips
        elif root:
            sc -= 6                       # a root that cannot generate its own stem
        strength = self._real_root(root)
        if strength == self.ATTESTED:
            sc += 8                       # al-'Ayn enters this root
        elif strength == self.LEXICON:
            sc += 3                       # in the lexicon only -- weaker
        elif strength == self.MUHMAL:
            sc -= 12                      # al-Khalil says this permutation is unused
        sc -= len(prefix) + len(suffix)   # penalise every clitic removed
        return sc

    def _closed_split(self, clean):
        """(prefix, particle, suffix) if the whole word is function words, else None."""
        s = strip_diac(clean)
        for pre in [''] + PRE_CLOSED:
            if pre and not s.startswith(pre):
                continue
            rest = s[len(pre):]
            if not rest:
                continue
            if rest in self._closed:
                if pre in SINGLE_PREPS:
                    # A BARE PREPOSITION IS NOT SELF-SUFFICIENT.  Ibn Malik (Tashil al-Fawa'id)
                    # gives the reason exactly: «وخص الجر بالاسم لأن عامله لا يستقل» -- the
                    # jarr-operator does not stand on its own.  So ب+هل، ك+انت، س+ان are not words.
                    continue
                # و and ف MAY enter upon a harf: Sibawayh «والواو تدخل على هل»; al-Mubarrad
                # «إنما الواو تدخل عليهن ... وهل ... وكيف ... ومتى ... وأين»; al-Zajjaji «فإن دخل
                # عليها الواو أو الفاء».  PARTIAL: stated for named particles, not as a general
                # rule -- and Ibn Jinni's own examples give «فلم تجبني» and «فقد أجبتك».
                return (pre or None, rest, None)
            # THE PRONOMINAL SET IS SIBAWAYH'S OWN ENUMERATION (Al-Kitab, باب علامة المضمرين
            # المنصوبين): «الكاف التي في رأيتك، وكما التي في رأيتكما، وكم ... وكن ... والهاء
            # التي في رأيته ... ورأيتها ... ورأيتهما ... ورأيتهم ... ورأيتهن ... ونى ... ونا».
            for suf in PRON_SUFFIXES:
                if not rest.endswith(suf) or len(rest) <= len(suf):
                    continue
                core = rest[:-len(suf)]
                if core in self._closed:
                    return (pre or None, core, suf)
                # AL-QALB, NOT AL-HADHF -- the sources say the alif is TURNED INTO ya', not
                # deleted.  Ibn 'Usfur (Sharh Jumal al-Zajjaji): «إن العرب قد تقلب الألف ياء مع
                # المضمر في نحو: عليه وإليه ولديه».  Al-Zajjaji (Huruf al-Ma'ani, لدى): «ومع
                # المضمر تنقلب ياء تقول لدى زيد ولديك».  Abu Hayyan: «وقلبت ألفه ياء لإضافته إلى
                # المضمر، كما قلبوا في عليك ولديك».  I had written "the alif is dropped" and
                # offered ما + ه -> مه, which no source supports.
                if core[-1] in 'يىا' and core[:-1] + 'ى' in self._closed:
                    return (pre or None, core[:-1] + 'ى', suf)
                # a bare preposition with a pronoun stays one closed token: بها، له، فيه
                if core in SINGLE_PREPS:
                    return (pre or None, rest, None)
        return None

    def decompose(self, word, base=None):
        base = _greedy(self.tok, word) if base is None else base
        if self.engine is None:
            return base
        self.stats['calls'] += 1
        clean = self.tok.clean_arabic(word)
        if not clean:
            return base
        if strip_diac(clean) in INDECLINABLE_NAMES or NAME_WORD.match(strip_diac(clean)):
            return base                   # a name, not a root-and-pattern word
        # only re-segment when the greedy result claims a ROOT; particles are already fine
        bp, br, bw, bs = base
        if not br or (br and bw is None and bp is None and bs is None):
            return base
        # THE CLOSED CLASS.  A function word has no root, so no root-and-pattern reading may be
        # manufactured for it.  This runs even when the greedy read the whole word as one root:
        # مني is من + ي, not a root.  The case it must NOT break -- كان read as «ك + ان» -- is
        # handled inside _closed_split, which refuses to strip a bare preposition without a
        # pronoun, so dropping the earlier whole-word gate is safe.
        closed = self._closed_split(clean)
        if closed is not None:
            pre, particle, suf = closed
            self.stats['closed_class'] += 1
            return (pre, particle, None, suf)
        # AL-'AYN ALREADY VOUCHES FOR THE GREEDY ROOT -> do not touch it.  Without this the
        # calculator happily replaced an attested root with another attested one: أصل -> صول,
        # بالعين عين -> بلع, لتكون كون -> وتك.  A re-segmentation must be a RESCUE of a reading
        # the sources do not support, never a swap between two they do.
        if self._real_root(br) == self.ATTESTED:
            return base
        if self.mode < 2:
            return base                    # closed-class-only mode: no root is re-assigned
        self.stats['root_reseg'] += 1

        # THE BASELINE GETS NO ROUND-TRIP BONUS. The analyser read the root FROM the word, so
        # regenerating it is vacuous -- every 3-letter word "round-trips" its own root. The
        # round trip is only informative for a SPLIT candidate, where the residue must generate
        # the stem. Awarding it to the baseline is what let kamm win over fa- + kam.
        # THE BASELINE IS RANKED BY THE SAME PROCEDURE, not given a free 0.  Its stem is the
        # surface minus its own clitics, so its al-haml 'ala al-asl count is meaningful.
        _bp, _bs = bp or '', bs or ''
        _bstem = clean[len(_bp):len(clean) - len(_bs)] if len(clean) > len(_bp) + len(_bs) else clean
        best = base
        best_key = self._rank(_bp, _bstem, _bs, br, bw, strip_diac(_bstem) in PARTICLES)
        for pre in [''] + PREFIXES:
            if pre and not clean.startswith(pre):
                continue
            rest = clean[len(pre):]
            if len(rest) < 2:
                continue
            for suf in [''] + SUFFIXES:
                if suf and not rest.endswith(suf):
                    continue
                stem = rest[:len(rest) - len(suf)] if suf else rest
                if len(strip_diac(stem)) < 2:
                    continue
                try:
                    p, r, w, s = _greedy(self.tok, stem)
                except Exception as exc:
                    # VISIBILITY: a _greedy failure dropped one candidate stem per iteration; a
                    # systematically broken tokenizer would have been invisible here.  Skipped.
                    _warn_once('greedy', f'tokenizer _decompose_greedy({stem!r}) raised: {exc!r}')
                    continue
                if not r:
                    continue
                # the tokenizer's own reading, plus every root al-Khalil's calculation reaches
                # for this stem -- including the ones whose radical was elided
                # A MUdARI'A PREFIX BELONGS TO THE WAZN.  يَفْعُلُ already contains its ي, so
                # reading يشذ as ي + شذ and then a different wazn double-counts the same letter:
                # it turned root شذذ into شوذ.  Reject a prefix that the stem's own wazn owns.
                if pre and bw and bw[0] in MUdARIA_LITERAL and pre[-1] == bw[0]:
                    continue
                is_part = strip_diac(stem) in PARTICLES
                if is_part:
                    readings = [(r, w)]      # a particle has no radicals to calculate from
                else:
                    readings = [(r, w)] + [c for c in self._candidate_roots(stem, w)
                                           if c != (r, w)]
                # CONSERVATIVE RULE: re-segment only on POSITIVE evidence -- the new residue's
                # root must be attested in al-'Ayn. Requiring merely "not condemned" lets a
                # nonsense root through whenever it is simply uncovered by the (partial) table,
                # which is how kuntu became ka + n-t-t.
                # a PARTICLE residue is its own evidence -- it is not a triliteral root, so the
                # root-attestation gate must not apply to it (that is why fa- + kam was skipped)
                for r2, w2 in readings:
                    if not r2:
                        continue
                    if not is_part:
                        # POSITIVE EVIDENCE, and al-'Ayn's own entry is what counts.  Admitting
                        # LEXICON-level evidence let the lexicon's junk (نتت، كمم، عنن) through
                        # on 22% of corpus words; requiring an actual al-'Ayn entry makes the
                        # validator change a word only when the sources vouch for the new root.
                        if self._real_root(r2) != self.ATTESTED:
                            continue
                        # THE ROUND TRIP IS THE VALIDATION, so a reading that cannot produce
                        # the stem it was read from must not replace the baseline at all.
                        # Raising al-'Ayn's weight to +8 was otherwise enough for مَفْعَل on
                        # مدرسة -- which generates مدرس, not مدرسة -- to displace the correct
                        # split, silently dropping the ta' marbuta from the output stream.
                        if not self._generates(stem, r2, w2):
                            continue
                    key = self._rank(pre, stem, suf, r2, w2, is_part)
                    if key > best_key:
                        best_key, best = key, (pre or None, r2, w2, suf or None)
        if best != base:
            self.stats['changed'] += 1
            if self.verbose:
                print(f'  {word}: {base} -> {best}')
        return best


# --------------------------------------------------------------------------------------------
# module-level API: what the tokenizer calls
# --------------------------------------------------------------------------------------------
_CACHE = {}


MODES = {'0': 0, '': 0, 'off': 0, 'false': 0, 'no': 0,
         '1': 1, 'closed': 1, 'true': 1, 'yes': 1, 'on': 1,
         '2': 2, 'full': 2, 'root': 2}


def mode():
    return MODES.get(os.environ.get('ROOTFORMER_VALIDATED_SEG', '1').lower(), 0)


def enabled():
    # DEFAULT OFF.  On a 20,000-word corpus sample the enabled validator changed ~15% of roots,
    # and hand-classifying 40 of those changes put the error rate well above zero (يشذ -> شوذ,
    # كذا -> كوذ, باعي -> بعو).  A switch whose error rate has not been measured against a gold
    # set must not be the default.  Set ROOTFORMER_VALIDATED_SEG=1 to enable.
    return mode() > 0


def _record_path():
    p = os.environ.get('ROOTFORMER_ATTEST_RECORD')
    if p:
        return p
    for cand in (_HERE.parent / 'data' / 'khalil_attest_v4.json',
                 _HERE / 'data' / 'khalil_attest_v4.json',
                 Path('/workspace/khalil_attest_v4.json')):
        if cand.exists():
            return str(cand)
    return None


def _greedy(tok, word):
    """The tokenizer's own analyse, bypassing the patched entry point so this cannot recurse."""
    f = getattr(tok, '_decompose_greedy', None)
    return f(word) if f is not None else tok.decompose_arabic_word(word)


def validator(tok):
    """One validator per tokenizer, built lazily -- the record is 360 KB and the engine is cheap,
    but a tokenizer may be constructed many times in one process."""
    if not enabled():
        return None
    v = _CACHE.get(id(tok))
    if v is None:
        try:
            v = ValidatedSegmentation(tok)
        except Exception as exc:
            # VISIBILITY: the validator is built lazily and the failure is cached, so without
            # this line the ENTIRE validation layer disappeared for the whole run in silence.
            # Default (False -> no validator) is unchanged.
            print(f'[validated_segmentation] validator construction failed, the greedy tokenizer '
                  f'is used unchanged: {exc!r}', file=sys.stderr)
            v = False                      # remember the failure; do not retry per word
        _CACHE[id(tok)] = v
    return v or None


def validated_decompose(tok, word, base):
    """Entry point used by the tokenizer.  Returns `base` unchanged whenever validation is off,
    unavailable, or inapplicable -- so the failure mode is always the previous behaviour."""
    v = validator(tok)
    if v is None:
        return base
    try:
        return v.decompose(word, base)
    except Exception as exc:
        # VISIBILITY: this is the outermost guard on the shipped path; swallowing a decompose bug
        # per word made the fallback to the greedy reading look like a normal result.  Still base.
        _warn_once('decompose', f'decompose({word!r}) raised, greedy reading returned: {exc!r}')
        return base
