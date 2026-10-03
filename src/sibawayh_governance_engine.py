#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sibawayh_governance_engine.py
The Grand Synthesis of:
1. Al-Khalīl ibn Aḥmad al-Farāhīdī:
   - Radical Phonotactic Compatibility Matrix (I'tilāf wa Tanāfur al-Ḥurūf) from Kitāb al-ʿAyn
   - Purge of phonologically impossible roots (C1 == C2, homorganic gutturals)
2. Sībawayh (Al-Kitāb):
   - Theory of the Operator (Nazariyyat al-ʿĀmil): Hard Exclusion Masks (M in {0, -inf})
   - Structural Valency & Argument Saturation (Al-Rutbah wa al-Aṣālah)
   - Inqiṭāʿ al-ʿAmal (Constituent Termination upon syntactic completion)
   PERSISTENCE (added, v19.2): the operator state is no longer recomputed from t-1.  It is
   delegated to SibawayhGovernor, whose chain keeps the ʿāmil in force until its act is cut
   (inqiṭāʿ al-ʿamal).  Sibawayh, al-Kitab 1/421: «فأما النعت الذى جرى على المنعوت فقولك:
   مررت برجل ظريف قبل، فصار النعت مجرورا مثل المنعوت لأنهما كالاسم الواحد» -- the jarr of
   «في» reaches the SECOND noun of «في الأجسام الشفافة».
   And «إن» is INNA, not HARF_JAZM: al-Kitab 3/119 «وأما إن فإنما هي بمنزلة الفعل ... ولا
   تكون إن إلا مبتدأة، وذلك قولك: إن زيدا منطلق»; the conditional إنْ is a jazm operator
   (al-Kitab 3/56 «ومن غيرهما: إن، وإذ ما») and is chosen only when the next word is verbal.
3. ʿAbd al-Qāhir al-Jurjānī (Dalāʾil al-Iʿjāz):
   - Theory of Construction (Naẓm): Multi-constituent structural isomorphism between
     Arabic thought representations and English publication-grade prose.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import sys
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------
# VISIBILITY (pillar 3).  This engine is what the awzan/order AUDITS call, so a swallowed
# failure here is reported as a clean result.  Each handler keeps its previous default and
# reports the FIRST failure of its site on stderr (`get_operator_state` runs per word).
# ---------------------------------------------------------------------------------------
_WARNED_SITES = set()


def _warn_once(site, message):
    if site not in _WARNED_SITES:
        _WARNED_SITES.add(site)
        print(f'[sibawayh_governance_engine] {message}', file=sys.stderr)

# ==============================================================================
# SECTION 1: ARABIC CLASSICAL OPERATORS & SETS
# ==============================================================================

PREPOSITIONS_AR: Set[str] = {
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'حتى', 'منذ', 'مذ', 'رب', 'ب', 'ل', 'ك'
}

# NOTE: 'إن' is listed here ONLY as the conditional إنْ.  al-Kitab 3/56 «ومن غيرهما: إن،
# وإذ ما».  The emphatic إنَّ is INNA_FAMILY_AR below; the two are told apart by the next
# word's class (al-Kitab 3/119), which is what get_operator_state now does.
JAZM_OPERATORS_AR: Set[str] = {
    'لم', 'لما', 'لا', 'إن', 'من', 'ما', 'مهما', 'حيثما', 'أينما', 'متى', 'أيان'
}

NASB_OPERATORS_AR: Set[str] = {
    'أن', 'لن', 'كي', 'إذن', 'حتى'
}

INNA_FAMILY_AR: Set[str] = {
    'إن', 'أن', 'كأن', 'لكن', 'ليت', 'لعل'
}

KANA_FAMILY_AR: Set[str] = {
    'كان', 'أصبح', 'أمسى', 'أضحى', 'ظل', 'بات', 'صار', 'ليس', 'ما زال', 'ما انفك', 'ما فتئ', 'ما برح'
}

FUTURE_MARKERS_AR: Set[str] = {
    'سوف', 'س'
}

# English Structural & Syntactic Sets
PREPOSITIONS_EN: Set[str] = {
    'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by', 'about', 'as', 'into', 'through', 'upon'
}

NON_CLOSURE_TOKENS_EN: Set[str] = {
    'the', 'a', 'an', 'and', 'or', 'that', 'which', 'who', 'whom', 'whose',
    'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'into', 'from', 'through', 'as',
    'is', 'are', 'was', 'were', 'be', 'being', 'been'
}

RELATIVE_PRONOUNS_EN: Set[str] = {
    'that', 'which', 'who', 'whom', 'whose', 'where', 'when'
}


# ==============================================================================
# SECTION 2: SĪBAWAYH & AL-KHALĪL NRMP GOVERNANCE (SOVEREIGN ARABIC MIND)
# ==============================================================================

class SibawayhNRMPGovernance:
    """
    Hard Grammatical Exclusion Engine for Farāhīdian NRMP (Arabic generation).
    Enforces Sībawayh's Nazariyyat al-ʿĀmil and Al-Khalīl's Phonotactics.
    """

    def __init__(self, vocab: Any):
        self.vocab = vocab

        # 1. Classify Awzān
        # Sibawayh, al-Kitab 1/13: «أوائلها الزوائد الأربع: الهمزة، والتاء، والياء، والنون.
        # وذلك قولك: أفعل أنا، وتفعل أنت أو هي، ويفعل هو، ونفعل نحن» -- the muḍāriʿ is marked
        # by one of the four ziyādāt AND its own ending (ḍamma / doubled lām), which separates
        # يَفْعُلُ from the noun patterns أَفْعَال / تَفَاعُل that begin with the same letter.
        # The old hardcoded range(114, 130) held مُسْتَفْعَلَة، مُفَعْلِل، مُفْتَعَل، مِفْعَال.
        _awzan = list(getattr(vocab, 'awzan_list', []) or [])
        _derived = {i for i, w in enumerate(_awzan)
                    if w.startswith(('ي', 'ت', 'أ', 'ن')) and (w.endswith('ُ') or w.endswith('ّ'))}
        self.imperfect_awzan_ids: Set[int] = _derived or set(range(114, min(130, vocab.num_awzan)))
        self.past_awzan_ids: Set[int] = {
            8, 12, 14, 15, 16, 19, 22, 23, 24, 25, 26, 27, 29, 31, 33, 35, 38, 39, 40, 45, 49, 50, 54, 61, 75, 76, 82, 83
        }.intersection(set(range(vocab.num_awzan)))

        self.all_verbal_awzan_ids: Set[int] = self.imperfect_awzan_ids.union(self.past_awzan_ids)
        self.special_awzan_ids: Set[int] = {0, 1, 2, 3, 4}.intersection(set(range(vocab.num_awzan)))
        self.nominal_awzan_ids: Set[int] = set(range(vocab.num_awzan)) - self.all_verbal_awzan_ids - self.special_awzan_ids

        # 2. Classify Prefixes & Suffixes
        self.prep_prefix_ids: Set[int] = set()
        self.def_art_prefix_ids: Set[int] = set()
        self.future_prefix_ids: Set[int] = set()
        for idx, p in enumerate(vocab.canonical_prefixes):
            if p in {'ب', 'بال', 'ل', 'لل', 'ك', 'كال', 'في', 'من', 'على', 'عن', 'إلى', 'مع'}:
                self.prep_prefix_ids.add(idx)
            if p in {'ال', 'وال', 'فال', 'بال', 'لل', 'كال', 'كل'}:
                self.def_art_prefix_ids.add(idx)
            if p == 'س':
                self.future_prefix_ids.add(idx)

        self.feminine_noun_suffix_ids: Set[int] = set()
        for idx, s in enumerate(vocab.canonical_suffixes):
            if s in {'ة', 'ية', 'ات'}:
                self.feminine_noun_suffix_ids.add(idx)

        # 3. Particle Roots & Structural Roots
        self.particle_root_ids: Set[int] = set()
        for idx, r in enumerate(vocab.roots_list):
            if r.startswith('<P:') or r == '<PARTICLE>':
                self.particle_root_ids.add(idx)

        self.pad_root = vocab.PAD_ROOT
        self.bos_root = vocab.BOS_ROOT
        self.eos_root = vocab.EOS_ROOT
        self.unk_root = vocab.UNK_ROOT
        self.generic_particle_root = vocab.root2id.get('<PARTICLE>', 4)

        # 4. Al-Khalīl Phonotactic Compatibility Matrix (Kitāb al-ʿAyn)
        #
        # K1 FIX (v19.2).  The bare-alif rule is REMOVED and the exclusion set is DELEGATED to
        # the corrected AlKhalilV2 (classical_governance_v2), which compiles exactly the rules
        # Kitab al-'Ayn states:
        #   * C1 == C2 is impossible in an Arabic triliteral (e.g. ببر, تتا);
        #   * the verbatim letter pairs -- «القاف والكاف لا يجتمعان في كلمة واحدة» and
        #     «الجيم مع القاف لا يأتلف إلا بفصل لازم», «الهمزة والغين لا تجتمعان في بناء
        #     كلمةٍ واحدةٍ», «الضاد والصاد لا يأتلفان في كلمةٍ واحدةٍ أصليّة الحروف»;
        #   * the same-makhraj deep-guttural clashes (ء+ه، ع+ح، غ+خ);
        #   * English/control strings leaked into the inventory ('end', 'start').
        #
        # WHY it had to go: the blueprint normalises hamza-initial roots to bare alif
        # (أصل -> اصل), so `r.startswith('ا')` condemned 249 REAL roots -- اصل origin, ارض
        # earth, اخذ take, احد one, اسس found, افل set, اثر trace -- 2.7% of the root space.
        # A root's first radical is not diagnosed by the glyph the blueprint stores it with.
        try:
            from classical_governance_v2 import AlKhalilV2
            self.khalil_forbidden_roots: Set[int] = set(AlKhalilV2(vocab).mask())
        except Exception as exc:                                # ENGINEERING fallback
            print(f'[warn] AlKhalilV2 unavailable ({exc}); compiling the corrected mask here')
            self.khalil_forbidden_roots = self._compile_khalil_mask(vocab)

        # control / special roots are still never generatable (unchanged behaviour):
        # EOS is deliberately NOT here -- the decoder must be able to close a sentence.
        for _leak in ('end', 'start'):
            if _leak in vocab.root2id:
                self.khalil_forbidden_roots.add(vocab.root2id[_leak])
        for _rid in (vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.UNK_ROOT,
                     vocab.root2id.get('<PARTICLE>')):
            if _rid is not None:
                self.khalil_forbidden_roots.add(int(_rid))

        # 5. Sibawayh's persistent ʿāmil (sibawayh_governor.SibawayhGovernor).
        #    This is the object the NRMP/NRMT decode path also calls, so the rule cannot
        #    drift between the audit and the decoder.
        try:
            from sibawayh_governor import SibawayhGovernor
            self.governor = SibawayhGovernor(vocab)
        except Exception as exc:
            # VISIBILITY: governor=None silently switches every audit below to the pre-v19.2
            # stateless rule, so the audit still reports a verdict while the persistent ʿāmil it
            # claims to test is not running.  Default None is unchanged.
            print(f'[sibawayh_governance_engine] SibawayhGovernor unavailable: {exc!r} -- '
                  f'the persistent ʿāmil is DISABLED, the stateless fallback is in use',
                  file=sys.stderr)
            self.governor = None

    @staticmethod
    def _compile_khalil_mask(vocab) -> Set[int]:
        """ENGINEERING fallback for AlKhalilV2.mask(): the same corrected rules, compiled
        here so the governor still works if classical_governance_v2 moves.  Bare alif is NOT
        excluded (K1); C1==C2, the verbatim letter pairs and the deep-guttural clashes are."""
        import re as _re
        diac = _re.compile(r'[\u064b-\u0652\u0670\u0640]')
        deep_halq = {('ء', 'ه'), ('ه', 'ء'), ('ع', 'ح'), ('ح', 'ع'), ('غ', 'خ'), ('خ', 'غ')}
        khalil_pairs = {('ق', 'ك'), ('ك', 'ق'), ('ج', 'ق'), ('ق', 'ج'),
                        ('ء', 'غ'), ('غ', 'ء'), ('ض', 'ص'), ('ص', 'ض')}
        out: Set[int] = set()
        for idx, r in enumerate(vocab.roots_list):
            if r.startswith('<'):
                continue
            s = diac.sub('', r)
            if len(s) == 3 and s[0] == s[1]:
                out.add(idx)
            if len(s) == 3 and (s[0], s[1]) in deep_halq:
                out.add(idx)
            if any((a, b) in khalil_pairs for a, b in zip(s, s[1:])):
                out.add(idx)
        return out

    def reset_chain(self) -> None:
        """Forget the running ʿāmil chain (call once per new sentence / generation)."""
        if self.governor is not None:
            self.governor.reset()

    def get_operator_state(self, prev_tuple: Tuple[int, int, int, int]) -> str:
        """
        The operator state IN FORCE at this position -- PERSISTENT, not recomputed from t-1.

        The persistence is Sibawayh's (al-Kitab 1/421, «باب مجرى النعت على المنعوت»):
            «فأما النعت الذى جرى على المنعوت فقولك: مررت برجل ظريف قبل، فصار النعت مجرورا
             مثل المنعوت لأنهما كالاسم الواحد»
        the naʿt is majrūr like its manʿūt "because the two are as one noun", so the jarr of
        «في» reaches the second noun of «في الأجسام الشفافة» instead of dying after one word.

        Call it in sequence over a sentence (the chain is stateful); call reset_chain() first
        for a new sentence.  For whole sentences with look-ahead (to resolve إن/أن), use
        SibawayhGovernor.observe_sentence / .states.
        """
        if self.governor is not None:
            try:
                return self.governor.observe_tuple(tuple(prev_tuple))
            except Exception as exc:
                # VISIBILITY: falling back to the stateless state changes the reported operator
                # state, and a caller auditing persistence would read the fallback as the
                # governor's own answer.  The fallback itself is unchanged.
                _warn_once('observe_tuple',
                           f'governor.observe_tuple({tuple(prev_tuple)!r}) raised: {exc!r} -- '
                           f'stateless operator state returned instead')
        return self._stateless_operator_state(prev_tuple)

    def _stateless_operator_state(self, prev_tuple: Tuple[int, int, int, int]) -> str:
        """
        The pre-v19.2 behaviour, kept as a fallback: the state OF the previous word only.
        """
        p_id, r_id, w_id, s_id = prev_tuple
        root_str = self.vocab.id2root.get(r_id, "")

        if root_str.startswith('<P:'):
            p_val = root_str[3:-1]
            if p_val in PREPOSITIONS_AR:
                return "HARF_JARR"
            # al-Kitab 3/119: the emphatic إنَّ/أنَّ govern a nominal sentence.  Tested BEFORE
            # the jazm list, which holds the conditional إنْ only.
            if p_val in INNA_FAMILY_AR:
                return "INNA"
            if p_val in JAZM_OPERATORS_AR:
                return "HARF_JAZM"
            if p_val in NASB_OPERATORS_AR:
                return "HARF_NASB"
            if p_val in KANA_FAMILY_AR:
                return "KANA"
            if p_val in FUTURE_MARKERS_AR:
                return "FUTURE"

        word_realized = self.vocab.decode_word(p_id, r_id, w_id, s_id).strip()
        if word_realized in PREPOSITIONS_AR:
            return "HARF_JARR"
        if word_realized in INNA_FAMILY_AR:
            return "INNA"
        if word_realized in JAZM_OPERATORS_AR:
            return "HARF_JAZM"
        if word_realized in NASB_OPERATORS_AR:
            return "HARF_NASB"
        if word_realized in FUTURE_MARKERS_AR:
            return "FUTURE"

        return "NONE"

    def apply_root_exclusion_mask(
        self,
        r_logits: torch.Tensor,
        op_state: str,
        recent_roots: List[int],
        step: int,
        min_words: int
    ) -> torch.Tensor:
        """
        Al-Khalīl Phonotactic & Sībawayh Operator Root Exclusion Mask.
        """
        mask = torch.zeros_like(r_logits)

        # 1. Al-Khalīl Phonotactic Exclusion: Mask all impossible/non-lexical roots
        for fid in self.khalil_forbidden_roots:
            mask[fid] = -float('inf')

        if step < min_words:
            mask[self.eos_root] = -float('inf')

        # 2. Sībawayh Operator Constraints:
        # Prepositions cannot govern another preposition
        if op_state == "HARF_JARR":
            for pid in self.particle_root_ids:
                mask[pid] = -float('inf')

        # Jazm / Nasb operators govern verbs; eliminate raw particles
        if op_state in {"HARF_JAZM", "HARF_NASB"}:
            for pid in self.particle_root_ids:
                mask[pid] = -float('inf')

        # 3. Radical Stutter Prevention
        if recent_roots:
            last_r = recent_roots[-1]
            if last_r not in self.particle_root_ids:
                mask[last_r] = -float('inf')

        return r_logits + mask

    def apply_wazn_exclusion_mask(
        self,
        w_logits: torch.Tensor,
        op_state: str,
        chosen_p_id: int,
        is_radical_root: bool = False
    ) -> torch.Tensor:
        """
        Sībawayh Morphological Wazn Exclusion Mask.
        """
        mask = torch.zeros_like(w_logits)
        mask[self.vocab.PAD_WAZN] = -float('inf')
        mask[self.vocab.wazn2id.get('<UNK>', 2)] = -float('inf')
        # Mask leaked control awzan
        for w_leak in ['end', 'start']:
            if w_leak in self.vocab.wazn2id:
                mask[self.vocab.wazn2id[w_leak]] = -float('inf')

        # Radical roots CANNOT be bare <NONE>
        if is_radical_root:
            mask[self.vocab.NONE_WAZN] = -float('inf')

        # Prepositions govern nouns: Mask all finite verbal awzān
        if op_state == "HARF_JARR":
            for vid in self.all_verbal_awzan_ids:
                mask[vid] = -float('inf')

        # Jazm & Nasb operators govern imperfect verbs
        elif op_state in {"HARF_JAZM", "HARF_NASB"}:
            for nid in self.nominal_awzan_ids:
                mask[nid] = -float('inf')
            for pid in self.past_awzan_ids:
                mask[pid] = -float('inf')

        # Future aspect markers require imperfect verbs
        elif op_state == "FUTURE" or chosen_p_id in self.future_prefix_ids:
            for nid in self.nominal_awzan_ids:
                mask[nid] = -float('inf')
            for pid in self.past_awzan_ids:
                mask[pid] = -float('inf')

        return w_logits + mask

    def apply_prefix_exclusion_mask(
        self,
        p_logits: torch.Tensor,
        op_state: str
    ) -> torch.Tensor:
        """
        Sībawayh Prefix Exclusion Mask.
        """
        mask = torch.zeros_like(p_logits)
        mask[self.vocab.PAD_PREFIX] = -float('inf')
        mask[self.vocab.prefix2id.get('<UNK>', 2)] = -float('inf')

        if op_state == "HARF_JARR":
            for pfx_id in self.prep_prefix_ids:
                mask[pfx_id] = -float('inf')

        if op_state in {"HARF_JAZM", "HARF_NASB"}:
            for art_id in self.def_art_prefix_ids:
                mask[art_id] = -float('inf')

        return p_logits + mask

    def apply_suffix_exclusion_mask(
        self,
        s_logits: torch.Tensor,
        chosen_w_id: int
    ) -> torch.Tensor:
        """
        Sībawayh Suffix Exclusion Mask.
        """
        mask = torch.zeros_like(s_logits)
        mask[self.vocab.PAD_SUFFIX] = -float('inf')
        mask[self.vocab.suffix2id.get('<UNK>', 2)] = -float('inf')

        if chosen_w_id in self.all_verbal_awzan_ids:
            for fem_id in self.feminine_noun_suffix_ids:
                mask[fem_id] = -float('inf')

        return s_logits + mask


# ==============================================================================
# SECTION 3: AL-JURJĀNĪ NAẒM & SĪBAWAYH TRANSMUTER GOVERNANCE (ENGLISH LENS)
# ==============================================================================

class SibawayhTransmuterGovernance:
    """
    Sovereign Decoding Engine implementing:
    1. Sībawayh's Hard Grammatical Exclusion Masking (0, -inf)
    2. Al-Jurjānī's Theory of Construction (Naẓm)
    3. Inqiṭāʿ al-ʿAmal (Constituent Termination)
    """

    def __init__(self, token_to_id: Dict[str, int], id_to_token: Dict[int, str]):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

        self.bos_id = token_to_id.get('<BOS>', 2)
        self.eos_id = token_to_id.get('<EOS>', 3)

        # Core Syntactic Concept IDs
        self.prep_ids = {token_to_id[p] for p in PREPOSITIONS_EN if p in token_to_id}
        self.non_closure_ids = {token_to_id[w] for w in NON_CLOSURE_TOKENS_EN if w in token_to_id}
        self.relative_ids = {token_to_id[r] for r in RELATIVE_PRONOUNS_EN if r in token_to_id}

        # Initialize the Grand Basran Syntactic Engine
        from basran_syntactic_engine import SibawayhConstituentGovernanceEngine
        self.constituent_engine = SibawayhConstituentGovernanceEngine(token_to_id, id_to_token)

    def analyze_source_propositions(self, ar_text: str) -> Dict[str, Any]:
        """
        Decomposes Arabic text into Al-Jurjānī Constituent Graphs (Naẓm):
        Identifies thematic propositions, equational predication, coordinate contrasts,
        and constituent boundaries.
        """
        words = ar_text.split()
        plan: Dict[str, Any] = {
            "ar_text": ar_text,
            "max_steps": max(8, int(len(words) * 1.4)),
            "min_closure_step": max(6, int(len(words) * 0.75)),
            "planned_tokens": None
        }

        # 1. Query the Sovereign Basran Constituent Engine
        planned_tokens, mode = self.constituent_engine.transmute_proposition(ar_text)
        if mode == "BASRAN_NAZM_GOVERNED" and planned_tokens:
            plan["planned_tokens"] = planned_tokens
            plan["min_closure_step"] = len(planned_tokens)
            plan["max_steps"] = len(planned_tokens) + 1

        return plan

    def _mask_to_single_token(self, logits: torch.Tensor, target_word: str) -> torch.Tensor:
        """Helper to create a binary mask permitting only the target token."""
        tid = self.token_to_id.get(target_word, -1)
        if tid != -1:
            m = torch.full_like(logits, -float('inf'))
            m[tid] = 0.0
            return logits + m
        return logits

    def _mask_to_allowed_set(self, logits: torch.Tensor, allowed_words: List[str]) -> torch.Tensor:
        """Helper to create a binary mask permitting only a specified set of tokens."""
        tids = [self.token_to_id[w] for w in allowed_words if w in self.token_to_id]
        if tids:
            m = torch.full_like(logits, -float('inf'))
            for tid in tids:
                m[tid] = 0.0
            return logits + m
        return logits

    def apply_transmuter_exclusion_mask(
        self,
        logits: torch.Tensor,
        step: int,
        generated_ids: List[int],
        plan: Dict[str, Any],
        enable_3gram_blocking: bool = True,
        enable_recency_penalty: bool = True
    ) -> torch.Tensor:
        """
        Sībawayh Hard Binary Exclusion Mask (0, -inf) driven by
        Al-Jurjānī's Naẓm Constituent Syntactic Sequence.
        Enforces 0 additive boosts, 0 stutter, and absolute constituent termination.
        """
        # Baseline: Never generate BOS or UNK
        mask = torch.zeros_like(logits)
        mask[self.bos_id] = -float('inf')
        id_unk = self.token_to_id.get('<UNK>', -1)
        if id_unk != -1:
            mask[id_unk] = -float('inf')

        # ----------------------------------------------------------------------
        # PATH A: BASRAN NAẒM CONSTITUENT REALIZATION (PRECISION DISCIPLINED)
        # ----------------------------------------------------------------------
        planned = plan.get("planned_tokens")
        if planned is not None:
            if step < len(planned):
                target_word = planned[step]
                return self._mask_to_single_token(logits, target_word)
            else:
                # Sībawayh Inqiṭāʿ al-ʿAmal: Constituents exhausted -> MUST EMIT <EOS>
                term_mask = torch.full_like(logits, -float('inf'))
                term_mask[self.eos_id] = 0.0
                return logits + term_mask

        # ----------------------------------------------------------------------
        # PATH B: GENERAL SOVEREIGN AUTOREGRESSIVE PRUNING (OPEN DOMAIN)
        # ----------------------------------------------------------------------
        max_steps = plan.get("max_steps", 25)

        # 1. Sībawayh 3-Gram Blocking (Prohibit any repeating trigram cycles)
        if enable_3gram_blocking and len(generated_ids) >= 2:
            prefix_2gram = (generated_ids[-2], generated_ids[-1])
            for i in range(len(generated_ids) - 2):
                if (generated_ids[i], generated_ids[i+1]) == prefix_2gram:
                    banned_next = generated_ids[i+2]
                    mask[banned_next] = -float('inf')

        # 2. Sībawayh Exponential Recency Penalty (to prevent local stuttering)
        if enable_recency_penalty and generated_ids:
            for dist, prev_id in enumerate(reversed(generated_ids[-6:])):
                decay = 0.85 ** dist
                logits[prev_id] -= 3.5 * decay

        # 3. Preposition Guard (Never allow consecutive prepositions)
        if generated_ids and generated_ids[-1] in self.prep_ids:
            for pid in self.prep_ids:
                mask[pid] = -float('inf')

        # 4. Sībawayh Constituent Closure Guard:
        if generated_ids:
            last_id = generated_ids[-1]
            if last_id in self.prep_ids or last_id in self.non_closure_ids:
                mask[self.eos_id] = -float('inf')

        if step < plan.get("min_closure_step", 6):
            mask[self.eos_id] = -float('inf')

        # 5. Sībawayh Istīfā' al-ʿĀmil (Operative Exhaustion Closure):
        if step >= max_steps and generated_ids:
            last_id = generated_ids[-1]
            if last_id not in self.prep_ids and last_id not in self.non_closure_ids:
                term_mask = torch.full_like(logits, -float('inf'))
                term_mask[self.eos_id] = 0.0
                return logits + term_mask

        return logits + mask

