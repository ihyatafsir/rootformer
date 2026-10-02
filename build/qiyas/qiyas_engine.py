#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qiyas_engine.py -- al-qiyas (analogical extension) as a computed rule.

THE PROCEDURE (usul al-fiqh / usul al-nahw anatomy)
---------------------------------------------------
    asl    the attested case        -> an attested (root, wazn) -> surface realisation
    far'   the new case             -> a (root, wazn) whose realisation is not attested
    'illa  the effective cause      -> a DETERMINATE PROPERTY of the root shared by asl and far'
    hukm   the ruling               -> the realisation (the root->surface ALIGNMENT)

The ruling established for the asl extends to the far' BY VIRTUE OF THE SHARED 'ILLA.
This module implements

    qiyas al-'illa   (extension by a shared effective cause)   <-- the strong form

and deliberately does NOT implement

    qiyas al-shabah  (extension by resemblance)     -- weaker; would let an unmotivated
                                                       similarity stand in for a cause
    qiyas al-dalala  (extension by textual indication) -- needs a text, not a rule

Reason for choosing qiyas al-'illa: the project's central claim is that a SHARED ROOT
REPRESENTATION generalises to unseen derivational forms.  That claim is exactly a claim
about an effective cause: the property of the root that carries the ruling across.  Only
qiyas al-'illa makes that cause explicit and testable.  qiyas al-shabah would be
unfalsifiable here (any root resembles any other in some respect), and qiyas al-dalala
would just be re-reading the corpus.

WHAT IS COMPUTED, NOT TRAINED
-----------------------------
Nothing here is fitted by gradient descent.  The hukm for a cell ('illa-label, wazn) is
the MODAL root->surface alignment over the attested asl in that cell.  Induction is
counting.  The only free choices are (i) which 'illa function to use and (ii) how to back
off when a cell is empty -- and both are subjected to the three validity conditions below,
which can REJECT them.

THE THREE USULI VALIDITY CONDITIONS, EACH WITH A MEASURABLE TEST
---------------------------------------------------------------
munda-bit   precise / well-defined
            TEST  determinacy: the 'illa is a total, single-valued function of the root
                  (must be 1.000 -- a vague cause is not a cause); and the hukm it
                  determines is well-defined -- cell purity = support(modal alignment)/n.
muttarid    consistent / regular, holds without exception across the attested set
            TEST  ittirad rate: fraction of attested (root, wazn) observations that the
                  'illa's own induced hukm reproduces EXACTLY.  1 - this = exception rate.
mun'akis    co-extensive: present wherever the ruling applies, absent where it does not
            TEST  discrimination lift: accuracy of the cell's hukm applied INSIDE the
                  class minus its accuracy applied OUTSIDE the class (same wazn).
                  A cause with lift ~ 0 does no work: it is a universal rule masquerading
                  as a cause (this is what rejects 'illa = wazn-alone).

An 'illa that fails its own test is REJECTED, not shipped (cf. Ibn 'Usfur's rejection of
Ibn Jinni's al-ishtiqaq al-akbar: «ghayr ma'khudh bihi; li-'adam ittiradihi»).
"""

from __future__ import annotations

import collections
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------------------
# 1. THE 'ILLA FUNCTIONS -- each a total, deterministic root -> label map
# ---------------------------------------------------------------------------------------

WEAK = frozenset('ويىا')      # waw, ya, alif-maqsura, alif (as a radical these are weak)
HAMZA = frozenset('ءأإآؤئ')


def _weak_positions(r: str) -> Tuple[int, ...]:
    return tuple(i for i, c in enumerate(r) if c in WEAK)


def _has_hamza(r: str) -> bool:
    return any(c in HAMZA for c in r)


def _geminate_positions(r: str) -> Tuple[Tuple[int, int], ...]:
    return tuple((i, i + 1) for i in range(len(r) - 1) if r[i] == r[i + 1])


def illa_identity(root: str) -> str:
    """'illa = THE ROOT ITSELF.  The task's literal scheme: asl and far' share the root,
    the far' being a different derivational form of the SAME root.  This is the 'illa that
    Tier-1 (unseen (root, wazn) pair, seen root) is entitled to use."""
    return root


def illa_wazn_only(root: str) -> str:
    """'illa = NOTHING about the root (a constant).  The context-free template.  Included
    as a NEGATIVE CONTROL: it is the strongest possible 'illa by coverage and must be
    rejected by mun'akis, because it is present where the ruling does NOT apply."""
    return '*'


def illa_coarse_weak(root: str) -> str:
    """'illa = {sound, has-a-weak-radical, geminate, hamza}.  The over-general 'illa that
    lumps ajwaf + naqis + lafif + mithal into one cause.  Classical target of the ittirad
    objection: the cause is real but not precise enough to carry one ruling."""
    if _has_hamza(root):
        return 'MAHMUZ'
    if _geminate_positions(root):
        return 'MUDAAF'
    if _weak_positions(root):
        return 'WEAK'
    return 'SAHIH'


def illa_strict7(root: str) -> str:
    """'illa = the classical seven-fold phonological division of the triliteral/quadriliteral
    root: SAHIH, MITHAL (r1 weak), AJWAF (r2 weak), NAQIS (r3 weak), LAFIF (two weak
    radicals), MUDAAF (geminate), MAHMUZ (hamza radical).  This is the 'illa the classical
    morphophonology actually uses (i'lal / idgham / hamza rules are stated per division)."""
    if _has_hamza(root):
        return 'MAHMUZ'
    w = _weak_positions(root)
    g = _geminate_positions(root)
    if g:
        return 'MUDAAF'
    if len(w) >= 2:
        return 'LAFIF'
    if len(w) == 1:
        i = w[0]
        if i == 0:
            return 'MITHAL'
        if i == len(root) - 1:
            return 'NAQIS'
        return 'AJWAF'
    return 'SAHIH'


def illa_positional(root: str) -> str:
    """'illa = strict7 refined by WHICH weak radical and its identity (waw vs ya), plus
    the geminate position.  The finest division: a stronger candidate for ittirad, at the
    cost of sparser cells."""
    if _has_hamza(root):
        return 'MAHMUZ'
    w = _weak_positions(root)
    g = _geminate_positions(root)
    if g:
        return 'MUDAAF_%d%d' % g[0]
    if len(w) >= 2:
        return 'LAFIF_%s' % ''.join(str(i) for i in w)
    if len(w) == 1:
        i = w[0]
        letter = root[i]
        if i == 0:
            return 'MITHAL_%s' % letter
        if i == len(root) - 1:
            return 'NAQIS_%s' % letter
        return 'AJWAF_%s' % letter
    return 'SAHIH'


ILLAS: Dict[str, Callable[[str], str]] = {
    'identity': illa_identity,
    'wazn_only': illa_wazn_only,
    'coarse_weak': illa_coarse_weak,
    'strict7': illa_strict7,
    'positional': illa_positional,
}

#: back-off lattice: a fine 'illa falls back to a coarser one when a cell is empty
BACKOFF: Dict[str, Optional[str]] = {
    'positional': 'strict7',
    'strict7': 'coarse_weak',
    'coarse_weak': 'wazn_only',
    'wazn_only': None,
    'identity': 'strict7',
}


# ---------------------------------------------------------------------------------------
# 2. ALIGNMENT -- the hukm.  A root->surface realisation IS an alignment.
# ---------------------------------------------------------------------------------------
#
# An alignment is a string over tokens.  Token forms:
#     '^i'   the i-th radical slot of the root (0-based, i < len(root))
#     'c'    the literal character c (an afformative the wazn supplies)
# The radical slots in an alignment must INCREASE, but they need not be contiguous and need
# not exhaust the root: dropping a radical is exactly what i'lal does.  Example: the hollow
# verb qa-la (قول -> قال) realises radicals 0 and 2 and ELIDES the weak radical 1, so its
# alignment is '^0ا^2'.  Instantiation stays well defined because the order is preserved.
#
# Examples:
#     root 'كتب'  surface 'كاتب'  ->  '^0ا^1^2'      instantiate -> 'كاتب'
#     root 'قول'  surface 'قال'   ->  '^0ا^2'        instantiate -> 'قال'   (i'lal: 1 elided)
#     root 'كرر'  surface 'كر'    ->  '^0^1'         instantiate -> 'كر'    (idgham)
#     root 'شفي'  surface 'شفاء'  ->  '^0^1اء'       instantiate -> 'شفاء'
#
# The hukm for a cell is the MODAL alignment over the attested asl.  Nothing is hand-coded:
# every one of the tokenizer's ~40 hard-coded wazn branches is RE-DERIVED here by counting.

#: an admissible hukm must realise at least this many radicals (a realisation that ignores
#: the root is not a realisation of that root).  Set to 2: the classical patterns never
#: realise fewer than two radicals of a triliteral/quadriliteral root.
MIN_ARITY = 2

ALIGN_RE = None  # kept for API symmetry; parsing is done by hand


def parse_alignment(aln: str) -> List[str]:
    """Split an alignment string into tokens ('^i' or a single literal char)."""
    out: List[str] = []
    i = 0
    while i < len(aln):
        if aln[i] == '^':
            out.append(aln[i:i + 2])
            i += 2
        else:
            out.append(aln[i])
            i += 1
    return out


def alignment_slots(aln: str) -> List[int]:
    return [int(t[1]) for t in parse_alignment(aln) if t.startswith('^')]


def align(surface: str, root: str, min_arity: int = MIN_ARITY) -> Optional[str]:
    """Deterministic, order-preserving alignment of `surface` to `root`.

    Computed by longest-common-subsequence dynamic programming over the root's radicals:
    the alignment uses a MAXIMUM-CARDINALITY increasing matching of radicals into the
    surface, so a surface character that coincides with a later radical cannot steal it
    from an earlier proper match (the naive left-to-right greedy aligner mis-anchors
    imperfectives such as يشفى on the root شفي, matching the prefix ي to the final radical).

    Ties are broken so that the EARLIEST radicals are matched and the excess is emitted as
    literal afformative characters.  Returns None if fewer than `min_arity` radicals are
    matched (a surface that ignores the root is not a realisation of that root) or if the
    surface contains '^'.  By construction instantiate(align(s, r), r) == s.
    """
    if not surface or not root or '^' in surface:
        return None
    n, m = len(surface), len(root)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, nxt = dp[i], dp[i + 1]
        for j in range(m - 1, -1, -1):
            if surface[i] == root[j]:
                row[j] = 1 + nxt[j + 1]
            else:
                a, b = nxt[j], row[j + 1]
                row[j] = a if a >= b else b
    toks: List[str] = []
    i = j = 0
    used = 0
    while i < n:
        if j < m and surface[i] == root[j] and dp[i][j] == 1 + dp[i + 1][j + 1]:
            toks.append('^%d' % j)
            i += 1
            j += 1
            used += 1
        elif dp[i][j] == dp[i + 1][j]:
            toks.append(surface[i])
            i += 1
        else:
            j += 1                      # skip a root radical (i'lal elision)
    if used < min_arity:
        return None
    return ''.join(toks)


def instantiate(aln: str, root: str) -> Optional[str]:
    """Apply an alignment to a (possibly unseen) root.  None if the alignment references a
    radical slot the root does not have."""
    out: List[str] = []
    for t in parse_alignment(aln):
        if t.startswith('^'):
            i = int(t[1:])
            if i >= len(root):
                return None
            out.append(root[i])
        else:
            out.append(t)
    return ''.join(out)


def alignment_arity(aln: str) -> int:
    s = alignment_slots(aln)
    return (max(s) + 1) if s else 0


# ---------------------------------------------------------------------------------------
# 3. THE QIYAS ENGINE
# ---------------------------------------------------------------------------------------

class Qiyas:
    """Computed analogical extension over an attested set of (root, wazn) -> surface.

    Usage
    -----
        q = Qiyas('strict7')
        for root, wazn, stem in attested: q.observe(root, wazn, stem)
        q.induce()
        q.realize('ثور', 'فَعَلَ')      # -> ('ثار', provenance)

    The engine is deliberately blind to any neural component: the hukm is a count.
    """

    def __init__(self, illa: str, backoff: bool = True):
        if illa not in ILLAS:
            raise KeyError('unknown ʿilla %r (have %s)' % (illa, sorted(ILLAS)))
        self.illa_name = illa
        self.illa = ILLAS[illa]
        self.backoff = backoff
        # The back-off LATTICE, materialised.  Each level induces its OWN table; a level's
        # labels are meaningless in another level's table (a root string is not a class
        # name), so looking up a coarser label in a finer table would silently miss and
        # drop every far' to the context-free global rule.  That was a real bug, caught by
        # the ʿilla comparison: strict7/positional/coarse_weak returned IDENTICAL far'
        # numbers because all three collapsed to the global rule.
        chain: List[str] = []
        n: Optional[str] = illa
        while n is not None:
            if n not in chain:
                chain.append(n)
            n = BACKOFF.get(n) if backoff else None
        if 'wazn_only' not in chain:
            chain.append('wazn_only')
        self.chain = chain
        self.obs: List[Tuple[str, str, str, str]] = []       # (label, wazn, root, surface)
        self.tables: Dict[str, Dict[Tuple[str, str], 'collections.Counter[str]']] = {}
        self.cells: Dict[Tuple[str, str], 'collections.Counter[str]'] = {}
        self.wazn_cells: Dict[str, 'collections.Counter[str]'] = {}
        self.roots: set = set()
        self.skipped_unalignable = 0
        self._induced = False

    # -- induction ---------------------------------------------------------------------

    @staticmethod
    def _label(name: str, root: str) -> str:
        return '*' if name == 'wazn_only' else ILLAS[name](root)

    def observe(self, root: str, wazn: str, surface: str) -> bool:
        """Add one attested asl.  Returns False if it is not alignable (and is therefore not
        an admissible asl for this procedure)."""
        a = align(surface, root)
        if a is None:
            self.skipped_unalignable += 1
            return False
        self.obs.append((self._label(self.illa_name, root), wazn, root, surface))
        self.roots.add(root)
        return True

    def induce(self) -> 'Qiyas':
        self.tables = {name: collections.defaultdict(collections.Counter)
                       for name in self.chain}
        self.wazn_cells = collections.defaultdict(collections.Counter)
        for _label, wazn, root, surface in self.obs:
            a = align(surface, root)
            if a is None:
                continue
            for name in self.chain:
                self.tables[name][(self._label(name, root), wazn)][a] += 1
            self.wazn_cells[wazn][a] += 1
        self.cells = self.tables[self.illa_name]
        self._induced = True
        return self

    # -- lookup ------------------------------------------------------------------------

    def _modal(self, table, key):
        """Modal alignment of a cell, with a TOTAL deterministic tie-break: greatest count,
        then lexicographically smallest alignment.  Counts may have been temporarily
        decremented by ittirad(), so non-positive entries are treated as absent."""
        c = table.get(key)
        if not c:
            return None
        live = [(a, n) for a, n in c.items() if n > 0]
        if not live:
            return None
        a, n = min(live, key=lambda t: (-t[1], t[0]))
        return a, n, sum(n for _, n in live), len(live)

    def rule(self, root: str, wazn: str) -> Optional[Tuple[str, str, int, int, int]]:
        """Return (alignment, label_that_fired, support, cell_n, cell_distinct) or None.

        Back-off walks the lattice, consulting EACH LEVEL'S OWN table, so the label that
        actually fired is reported and downstream can tell whether the primary 'illa carried
        the ruling or a coarser one did (``GLOBAL`` = the context-free wazn rule, the last
        resort, which is exactly the ʿilla the mun'akis test rejects).
        """
        if not self._induced:
            raise RuntimeError('call induce() first')
        for name in self.chain:
            label = self._label(name, root)
            m = self._modal(self.tables[name], (label, wazn))
            if m:
                a, sup, tot, dist = m
                tag = 'GLOBAL' if name == 'wazn_only' else '%s:%s' % (name, label)
                return a, tag, sup, tot, dist
        m = self._modal(self.wazn_cells, wazn)
        if m:
            a, sup, tot, dist = m
            return a, 'GLOBAL', sup, tot, dist
        return None

    def realize(self, root: str, wazn: str) -> Optional[Tuple[str, str, int, int, int]]:
        """Realise (root, wazn) by qiyas.  Returns (surface, provenance, support, cell_n,
        cell_distinct) or None when no analogue exists anywhere in the lattice."""
        r = self.rule(root, wazn)
        if r is None:
            return None
        a, prov, sup, tot, dist = r
        s = instantiate(a, root)
        if s is None:
            return None
        return s, prov, sup, tot, dist

    # -- the three validity conditions --------------------------------------------------

    def ittirad(self, holdout_n: int = 0, seed: int = 0):
        """MUTTARID -- consistency.  Leave-one-out over the attested set: for each admitted
        asl, re-induce WITHOUT it and check the 'illa reproduces it exactly.  This is the
        non-circular form of the test (a rule cannot be credited for memorising the example
        it was counted from).

        Exact O(n) implementation: decrement the observation's own cell count (and the
        wazn cell, and every coarser cell in the back-off chain) for the duration of the
        test, then restore.  Equivalent to re-inducing per observation.
        """
        if not self._induced:
            raise RuntimeError('call induce() first')
        chain_names = list(self.chain)

        ok = 0
        fail = 0
        fails = []
        for _label, wazn, root, surface in self.obs:
            a = align(surface, root)
            if a is None:
                continue
            # decrement this observation's own counts at EVERY level of the lattice
            touched = []
            for nm in chain_names:
                key = (self._label(nm, root), wazn)
                self.tables[nm][key][a] -= 1
                touched.append((nm, key, a))
            # pick the same way rule() would
            pred = None
            for nm in chain_names:
                m = self._modal(self.tables[nm], (self._label(nm, root), wazn))
                if m:
                    pred = m[0]
                    break
            if pred is None:
                m = self._modal(self.wazn_cells, wazn)
                pred = m[0] if m else None
            # restore
            for nm, key, aa in touched:
                self.tables[nm][key][aa] += 1
            if pred == a:
                ok += 1
            else:
                fail += 1
                if len(fails) < 40:
                    fails.append((root, wazn, surface, pred))
        return {
            'n': ok + fail,
            'ok': ok,
            'fail': fail,
            'ittirad_rate': (ok / (ok + fail)) if (ok + fail) else float('nan'),
            'examples': fails,
        }

    def mundabit(self):
        """MUNDABIT -- precision / well-definedness.

        (a) determinacy of the 'illa itself: fraction of the attested roots on which it
            returns exactly one label (must be 1.000; single-valued by construction).
        (b) well-definedness of the hukm it determines: cell purity = support of the modal
            alignment / observations, weighted by observations, plus the share of cells
            that are impure (purity < 1).
        """
        single = 0
        multi = 0
        for r in self.roots:
            try:
                lab = self.illa(r)
            except Exception:
                multi += 1
                continue
            if isinstance(lab, str):
                single += 1
            else:
                multi += 1
        tot_obs = 0
        weighted = 0.0
        impure = 0
        cells = 0
        dist_hist = collections.Counter()
        for (lab, wazn), c in self.cells.items():
            n = sum(c.values())
            if n == 0:
                continue
            cells += 1
            sup = c.most_common(1)[0][1]
            tot_obs += n
            weighted += sup
            if sup != n:
                impure += 1
            dist_hist[len(c)] += 1
        return {
            'n_roots': len(self.roots),
            'determinate_roots': single,
            'indeterminate_roots': multi,
            'determinacy': single / max(single + multi, 1),
            'n_cells': cells,
            'n_observations': tot_obs,
            'impure_cells': impure,
            'purity': (weighted / tot_obs) if tot_obs else float('nan'),
            'distinct_alignment_histogram': dict(sorted(dist_hist.items())),
        }

    def munakis(self, min_cell: int = 30):
        """MUN'AKIS -- co-extensiveness.

        For every cell (label L, wazn W) with at least `min_cell` observations, apply the
        cell's modal alignment to
            IN  : roots observed with label L and wazn W
            OUT : roots NOT of label L that were observed with wazn W
        and compare exact-surface accuracy.  The cause is co-extensive iff it is present
        wherever the ruling applies and absent where it does not -- i.e. iff
        accuracy_in is high and accuracy_in - accuracy_out (the LIFT) is materially > 0.

        This is the test that rejects 'illa = wazn-alone: there L is constant, so IN and OUT
        are the same set and the lift is exactly 0 by construction.
        """
        # index observations by (root, wazn) -> surface  (for the OUT check we need the
        # surface of a root that is NOT of label L)
        by_wazn = collections.defaultdict(dict)
        for label, wazn, root, surface in self.obs:
            by_wazn[wazn][root] = surface

        rows = []
        tot_in = tot_out = ok_in = ok_out = 0
        for (label, wazn), c in self.cells.items():
            n = sum(c.values())
            if n < min_cell:
                continue
            aln, sup = c.most_common(1)[0]
            if label == '*':
                continue                       # a constant cause has no OUTSIDE; reported separately
            i_ok = i_n = o_ok = o_n = 0
            for root, surface in by_wazn.get(wazn, {}).items():
                lab_r = self.illa(root)
                pred = instantiate(aln, root)
                if lab_r == label:
                    i_n += 1
                    if pred == surface:
                        i_ok += 1
                else:
                    o_n += 1
                    if pred == surface:
                        o_ok += 1
            if i_n == 0:
                continue
            rows.append({
                'cell': '%s|%s' % (label, wazn), 'n': n,
                'in_n': i_n, 'in_acc': i_ok / i_n,
                'out_n': o_n, 'out_acc': (o_ok / o_n) if o_n else None,
                'lift': (i_ok / i_n) - ((o_ok / o_n) if o_n else 0.0),
            })
            tot_in += i_n
            ok_in += i_ok
            tot_out += o_n
            ok_out += o_ok
        return {
            'n_cells': len(rows),
            'in_n': tot_in, 'in_acc': (ok_in / tot_in) if tot_in else float('nan'),
            'out_n': tot_out, 'out_acc': (ok_out / tot_out) if tot_out else float('nan'),
            'lift': ((ok_in / tot_in) if tot_in else 0.0) - ((ok_out / tot_out) if tot_out else 0.0),
            'worst_cells': sorted(rows, key=lambda r: r['lift'])[:15],
            'rows': rows,
        }

    # -- reporting ---------------------------------------------------------------------

    def coverage(self, targets: Iterable[Tuple[str, str]]):
        """How much of a target (root, wazn) list the lattice can reach, by firing level."""
        lv = collections.Counter()
        miss = 0
        for root, wazn in targets:
            r = self.rule(root, wazn)
            if r is None:
                miss += 1
                lv['MISS'] += 1
            else:
                lv[r[1].split(':')[0]] += 1
        return dict(lv), miss


# ---------------------------------------------------------------------------------------
# 4. BRUTE FORCE -- for validating the modal-alignment induction
# ---------------------------------------------------------------------------------------

def brute_force_alignments(root: str, alphabet: Sequence[str], max_extra: int = 2):
    """Enumerate the FULL bounded alignment space for `root` as the aligner defines it:
    every order-preserving, possibly non-contiguous subset of the radicals (at least
    MIN_ARITY of them) interleaved with 0..max_extra literal characters from `alphabet`.

    Yields alignment strings.  Used to prove that Qiyas.realize returns the argmax over the
    space rather than something an accident of the greedy aligner produced.
    """
    for k in range(min(MIN_ARITY, len(root)), len(root) + 1):   # how many radicals realised
        for slots in itertools.combinations(range(len(root)), k):
            body = ['^%d' % i for i in slots]
            for e in range(0, max_extra + 1):
                for pos in itertools.combinations_with_replacement(range(len(body) + 1), e):
                    for letters in itertools.product(alphabet, repeat=e):
                        out = list(body)
                        for off, (p, ch) in enumerate(zip(sorted(pos), letters)):
                            out.insert(p, ch)
                        yield ''.join(out)


def brute_force_cell(cell_obs, alphabet, max_extra: int = 2):
    """The hukm a cell SHOULD carry, computed by exhaustive search over the alignment space.

    `cell_obs` is the cell's observations [(root, surface), ...] (roots may differ).  The
    search enumerates, for the LONGEST root in the cell, every alignment of the bounded
    space and scores it by the number of observations it reproduces exactly; the argmax is
    the brute-force hukm.  Returns (alignment, score, n_candidates).
    """
    if not cell_obs:
        return None, 0, 0
    longest = max((r for r, _ in cell_obs), key=len)
    best, best_n, n_cand = None, -1, 0
    for aln in brute_force_alignments(longest, alphabet, max_extra):
        n_cand += 1
        n = 0
        for r, s in cell_obs:
            if instantiate(aln, r) == s:
                n += 1
        # the SAME total tie-break as Qiyas._modal: greatest count, then lexicographically
        # smallest alignment
        if n > best_n or (n == best_n and best is not None and aln < best):
            best_n, best = n, aln
    return best, best_n, n_cand
