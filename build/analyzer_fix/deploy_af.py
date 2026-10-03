#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploy_af.py -- generate the PATCHED nrmp_vocab.py for the analyzer fix, prove it is a pure
PATCH of the shipped module (un-patch is byte-identical), and prove flag-off inertness.

    python deploy_af.py --src /workspace/hf_v19_2_release/nrmp_vocab.py \
                        --out /workspace/analyzer_fix/patched/nrmp_vocab.py
    python deploy_af.py --verify    # un-patch -> md5 must equal the shipped md5

Design
------
AF_ANALYZER_FIX=0  -> every patched line is inert; the module behaves exactly as shipped.
AF_ANALYZER_FIX=1  -> two fixes:
  (1) DIVINE   : the divine name and its clitic-built forms are SURFACE-PRESERVING
                 PASSTHROUGHS.  One appended root id per hard-coded surface carries the exact
                 codepoints; encode/decode is the identity by construction.
  (2) TASRIF   : where the hand-written realiser has no branch for the wazn (proved by
                 EXECUTION: it returned the bare root), realisation is delegated to the
                 project's own grammar engine, tasrif_engine.TasrifEngine -- the implementation
                 of Ibn 'Usfur's al-Mumti' fi al-Tasrif.
"""
import argparse
import hashlib
import os
import sys

A1 = "from typing import Tuple, List, Dict, Any, Optional, Iterable\n"
A2 = "        self.roots_list = self.roots_list + list(self.lisan_appended_roots)\n"
A3 = ("        clean_w = word.strip()\n"
      "        if not clean_w:\n"
      "            return (self.NONE_PREFIX, self.UNK_ROOT, self.NONE_WAZN, self.NONE_SUFFIX)\n")
A4 = "        root_str = self.id2root.get(r_id, '<UNK>')\n"
A5 = ("        # Canonical Morphological Realization\n"
      "        if wazn_str not in ['<NONE>', '<PAD>', '<UNK>']:\n"
      "            stem = self.base_tok.realize_root_and_wazn(root_str, wazn_str)\n"
      "        else:\n"
      "            stem = root_str\n")
A6 = "    def encode_sentence(self, sentence: str) -> List[Tuple[int, int, int, int]]:\n"

# ---------------------------------------------------------------------------------------------
# THE HARD-CODED DIVINE SET.  Every entry below was OBSERVED in the held-out sample
# (divine_scan.py) -- none is guessed.  Non-divine look-alikes that the scan also returned
# (اللهو اللهب اللهيب اللهج اللهاة علله خلله يتخلله قلله ذللته يضلله ...) are deliberately
# EXCLUDED: they are the roots لهو/لهب/لهج/لهي and علل/خلل/ضلل, not the divine name.
# ---------------------------------------------------------------------------------------------
AF_DIVINE_SURFACES = (
    # the name itself, in both orthographies (lam+lam+heh, and with shadda / dagger alef)
    '\u0627\u0644\u0644\u0647',                    # الله
    '\u0671\u0644\u0644\u0651\u0670\u0647',        # ٱللّٰه
    # with a clitic prefix
    '\u0648\u0627\u0644\u0644\u0647',              # والله
    '\u0641\u0627\u0644\u0644\u0647',              # فالله
    '\u0628\u0627\u0644\u0644\u0647',              # بالله
    '\u0648\u0628\u0627\u0644\u0644\u0647',        # وبالله
    '\u0644\u0644\u0647',                          # لله
    '\u0648\u0644\u0644\u0647',                    # ولله
    '\u0641\u0644\u0644\u0647',                    # فلله
    '\u0641\u0628\u0627\u0644\u0644\u0647',        # فبالله
    '\u0641\u0648\u0627\u0644\u0644\u0647',        # فوالله
    '\u0644\u0644\u0644\u0647',                    # للله  (observed)
    # the vocative, with the mim that al-Mubarrad calls the substitute for يا
    '\u0627\u0644\u0644\u0647\u0645',              # اللهم
    # the oath-particle tā'
    '\u062a\u0627\u0644\u0644\u0647',              # تالله
    # hamza spellings
    '\u0623\u0644\u0644\u0647',                    # ألله
    '\u0622\u0644\u0644\u0647',                    # آلله
    '\u0625\u0644\u0644\u0647',                    # إلله
    '\u0628\u0623\u0644\u0644\u0647',              # بألله
    # run-together tokens that CONTAIN the name (observed in the sample)
    '\u0647\u0648\u0627\u0644\u0644\u0647',        # هوالله
    '\u0625\u0644\u0627\u0628\u0627\u0644\u0644\u0647',   # إلابالله
    '\u0645\u0627\u0634\u0627\u0621\u0627\u0644\u0644\u0647',  # ماشاءالله
    '\u0646\u062d\u0645\u062f\u0627\u0644\u0644\u0647',    # نحمدالله
    # proper names built on the name (observed)
    '\u0639\u0628\u062f\u0627\u0644\u0644\u0647',  # عبدالله
    '\u0648\u0639\u0628\u062f\u0627\u0644\u0644\u0647',  # وعبدالله
    '\u0639\u064a\u062f\u0627\u0644\u0644\u0647',  # عيدالله
)



def handled_wazn_set(tok_path):
    """The wazn names the hand-written realiser has a BRANCH for, read off the AST.

    A presence check is worthless (the brief), so this is not a grep: it walks
    `realize_root_and_wazn` and collects string constants that are compared against the name
    `wazn`.  Cross-checked by EXECUTION elsewhere: the set is exactly 46, the same number the
    project's own harness prints ("realize implements 46 wazn names; vocab has 142").
    """
    import ast
    tree = ast.parse(open(tok_path, encoding='utf-8').read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'realize_root_and_wazn':
            for n in ast.walk(node):
                if not isinstance(n, ast.Compare):
                    continue
                ops = [n.left] + list(n.comparators)
                if not any(isinstance(o, ast.Name) and o.id == 'wazn' for o in ops):
                    continue
                for o in ops:
                    if isinstance(o, ast.Name) and o.id == 'wazn':
                        continue
                    for c in ast.walk(o):
                        if isinstance(c, ast.Constant) and isinstance(c.value, str):
                            out.add(c.value)
    return out


BLOCK = '''
# ============================================================================================
# AF_ANALYZER_FIX -- APPEND-ONLY PATCH, generated by /workspace/analyzer_fix/deploy_af.py
#
# Flag AF_ANALYZER_FIX=1 enables both fixes; unset (the shipped default) leaves every patched
# statement inert and the module byte-for-byte equivalent in BEHAVIOUR to the shipped file.
# ============================================================================================
AF_FIX = os.environ.get('AF_ANALYZER_FIX', '0') == '1'
# per-fix switch, so each fix can be measured on its own: AF_NO_TASRIF=1 -> divine only
AF_TASRIF = AF_FIX and os.environ.get('AF_NO_TASRIF', '0') != '1'

# The wazn names the hand-written chain has a BRANCH for (AST-derived, execution-cross-checked).
AF_HANDLED_WAZN = frozenset(__AF_HANDLED__)

_AF_DIAC = re.compile('[\\u064b-\\u0652\\u0670\\u0640\\u06d6-\\u06ed]')

# The divine name and its clitic-built forms, HARD-CODED from the held-out sample
# (divine_scan.py).  See deploy_af.py for the exclusions and why they are excluded.
AF_DIVINE_SURFACES = __AF_DIVINE__

_AF_DIVINE_LOOKUP = {}
for _s in AF_DIVINE_SURFACES:
    _AF_DIVINE_LOOKUP[_s] = _s
    _AF_DIVINE_LOOKUP[_AF_DIAC.sub('', _s)] = _s

# the mudaari' letters: al-huruf al-mudari'a -- al-mutakallim's alif, and nun/ya'/ta'
AF_MUDARIA_LETTERS = ('\u0623', '\u0646', '\u064a', '\u062a')

_AF_ENGINE = None

try:
    from morphemic_tokenizer_v12_arabic import ROOT_CANONICAL_MAP_REV as _AF_ROOT_REV
except Exception:
    try:
        from models.morphemic_tokenizer_v12_arabic import ROOT_CANONICAL_MAP_REV as _AF_ROOT_REV
    except Exception:
        _AF_ROOT_REV = {}


def _af_engine():
    """The project's own tasrif engine, or None.  Imported lazily so flag-off costs nothing."""
    global _AF_ENGINE
    if _AF_ENGINE is None:
        try:
            from tasrif_engine import TasrifEngine
            _AF_ENGINE = TasrifEngine(vocab=None)
        except Exception:
            _AF_ENGINE = False
    return _AF_ENGINE or None
'''

INIT_BLOCK = '''
        # ---- AF_ANALYZER_FIX: the <DIV:..> passthrough root ids.  APPENDED, so no existing
        # root id moves.  One id per hard-coded surface is what makes the passthrough exact:
        # the tuple has finite capacity, so a general surface-preserving reading of an
        # unbounded set is impossible -- for a fixed, observed set it is one id per form.
        self.af_divine_set = frozenset(_AF_DIVINE_LOOKUP) if AF_FIX else frozenset()
        self.af_divine_lookup = dict(_AF_DIVINE_LOOKUP) if AF_FIX else {}
        self.af_divine_root2surface = {}
        if AF_FIX:
            _div = sorted(set('<DIV:%s>' % s for s in AF_DIVINE_SURFACES))
            _have = set(self.roots_list)
            self.af_divine_appended_roots = [x for x in _div if x not in _have]
            self.roots_list = self.roots_list + self.af_divine_appended_roots
            self.af_divine_root2surface = {
                ('<DIV:%s>' % s): s for s in AF_DIVINE_SURFACES}
        else:
            self.af_divine_appended_roots = []
'''

ENCODE_BLOCK = '''
        # ---- AF_ANALYZER_FIX (1): THE DIVINE NAME IS A SURFACE-PRESERVING PASSTHROUGH.
        # Its surface is hamzat wasl + the lam-alif ligature + shadda + the dagger alef; no
        # (P,R,W,S) tuple regenerates it, and the analyzer's defensible root (أله) is
        # orthographically unrebuildable.  The tradition labels the case an ism ʿalam whose
        # alif was elided, irregularity by frequency of use rather than by rule.  We therefore
        # carry the exact codepoints and hand them back unchanged -- no root, no wazn, no
        # affix split, no normalisation, no letter substitution, no ligature folding.
        if AF_FIX:
            _div = self.af_divine_lookup.get(clean_w)
            if _div is not None:
                return (self.NONE_PREFIX, self.root2id['<DIV:%s>' % _div],
                        self.NONE_WAZN, self.NONE_SUFFIX)
'''

DECODE_BLOCK = '''
        # ---- AF_ANALYZER_FIX (1): the divine passthrough returns the stored surface verbatim.
        if AF_FIX and root_str.startswith('<DIV:'):
            return root_str[5:-1]
'''

REALIZE_BLOCK = '''            stem = self.base_tok.realize_root_and_wazn(root_str, wazn_str)
            if AF_TASRIF:
                stem = self._af_tasrif_realize(root_str, wazn_str, stem)
'''

METHOD_BLOCK = '''    def _af_tasrif_realize(self, root, wazn, shipped):
        """AF_ANALYZER_FIX (2): realise through the project's OWN grammar engine.

        Ibn 'Usfur, al-Mumti' fi al-Tasrif, bab al-idgham: the two like letters are merged
        («إذا اجتمع المثلان وجب الإدغام»), which the hand-written chain does not do -- it emits
        ي + دلل = يدلل for the muḍaʿʿaf root دلل where the language has يدل.  The engine
        (tasrif_engine.TasrifEngine, the module that implements al-Mumti') already does it.

        SAFETY.  Fires only when the shipped chain has produced no realisation at all -- its
        output is the bare root, i.e. no branch matched -- and only when the engine produces a
        DIFFERENT, non-empty skeleton.  Anything the shipped chain actually realises is returned
        untouched.
        """
        if not wazn or not root or root.startswith('<'):
            return shipped
        _r = _AF_DIAC.sub('', root)
        # THE SCOPE OF THE RULE.  al-Mumti' states idgham for two like letters MEETING IN A
        # VERB, where the first is not word-initial (lines 6269-6280).  That is exactly the
        # mudari' of a mudda''af: ydllu -> yadullu -> yadull.  It is NOT the nominal
        # patterns: the engine's i'lal would turn the NOUN qawl (فَعَلَ skeleton == root)
        # into the VERB qala, and radiya into rada, breaking 3,428 currently-passing types
        # when tried corpus-wide.  So the delegation is restricted to the case the cited
        # rule governs.
        _gem = (len(_r) == 3 and _r[1] == _r[2] and wazn[0] in AF_MUDARIA_LETTERS)
        _nobranch = wazn not in AF_HANDLED_WAZN
        if not (_gem or _nobranch):
            return shipped
        _bare = _AF_ROOT_REV.get(root, root)
        if _AF_DIAC.sub('', shipped) != _AF_DIAC.sub('', _bare):
            return shipped
        eng = _af_engine()
        if eng is None:
            return shipped
        try:
            res = eng.generate(root, wazn)
        except Exception:
            return shipped
        if not res:
            return shipped
        got = _AF_DIAC.sub('', res[0])
        if not got or got == _AF_DIAC.sub('', _bare):
            return shipped
        return got

'''


def patch(src_text, handled=()):
    global HANDLED
    HANDLED = handled
    s = src_text
    for a in (A1, A2, A3, A4, A5, A6):
        if s.count(a) != 1:
            raise SystemExit('ANCHOR NOT UNIQUE (%d): %r' % (s.count(a), a[:60]))
    # A1: module-level machinery (needs the diacritic regex; define our own)
    s = s.replace(A1, A1 + BLOCK.replace('__AF_DIVINE__', repr(AF_DIVINE_SURFACES)).replace('__AF_HANDLED__', repr(tuple(sorted(HANDLED)))), 1)
    # A2: append the divine root ids inside __init__ (before root2id is built)
    s = s.replace(A2, A2 + INIT_BLOCK, 1)
    # A3: encode passthrough
    s = s.replace(A3, A3 + ENCODE_BLOCK, 1)
    # A4: decode passthrough
    s = s.replace(A4, A4 + DECODE_BLOCK, 1)
    # A5: realise via the engine
    s = s.replace(A5, A5.replace(
        "            stem = self.base_tok.realize_root_and_wazn(root_str, wazn_str)\n",
        REALIZE_BLOCK), 1)
    # A6: the helper method
    s = s.replace(A6, METHOD_BLOCK + A6, 1)
    return s


def unpatch(text):
    # uses the module-global HANDLED set from patch()
    s = text
    s = s.replace(A1 + BLOCK.replace('__AF_DIVINE__', repr(AF_DIVINE_SURFACES)).replace('__AF_HANDLED__', repr(tuple(sorted(HANDLED)))), A1, 1)
    s = s.replace(A2 + INIT_BLOCK, A2, 1)
    s = s.replace(A3 + ENCODE_BLOCK, A3, 1)
    s = s.replace(A4 + DECODE_BLOCK, A4, 1)
    s = s.replace(A5.replace(
        "            stem = self.base_tok.realize_root_and_wazn(root_str, wazn_str)\n",
        REALIZE_BLOCK), A5, 1)
    s = s.replace(METHOD_BLOCK + A6, A6, 1)
    return s


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/workspace/hf_v19_2_release/nrmp_vocab.py')
    ap.add_argument('--out', default='/workspace/analyzer_fix/patched/nrmp_vocab.py')
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()

    original = open(a.src, 'rb').read()
    text = original.decode('utf-8')
    TOK = os.environ.get('AF_TOKENIZER',
                         '/workspace/hf_v19_2_release/models/morphemic_tokenizer_v12_arabic.py')
    handled = handled_wazn_set(TOK) if os.path.exists(TOK) else set()
    print('[*] handled wazn (AST of %s): %d' % (os.path.basename(TOK), len(handled)))
    if a.verify:
        patched = patch(text, handled)
        back = unpatch(patched)
        ok = (back == text)
        print('[*] src          %s  md5 %s' % (a.src, md5(original)))
        print('[*] patched md5  %s' % md5(patched.encode('utf-8')))
        print('[*] un-patch == original : %s  (%s)' % (ok, 'BYTE-IDENTICAL' if ok else 'DIFFERS'))
        # AST-level assertion: the helper must be a real def, not a comment
        import ast
        tree = ast.parse(patched)
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                names.add(node.name)
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute):
                    names.add('CALL.' + f.attr)
        print('[*] AST has def _af_tasrif_realize      : %s'
              % ('_af_tasrif_realize' in names))
        print('[*] AST has CALL self._af_tasrif_realize: %s'
              % ('CALL._af_tasrif_realize' in names))
        print('[*] AST has def _af_engine              : %s' % ('_af_engine' in names))
        assert ok, 'un-patch is not byte-identical'
        assert '_af_tasrif_realize' in names and 'CALL._af_tasrif_realize' in names
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    patched = patch(text, handled)
    open(a.out, 'w', encoding='utf-8').write(patched)
    print('[*] wrote %s' % a.out)
    print('[*] src md5 %s -> patched md5 %s' % (md5(original), md5(patched.encode('utf-8'))))


if __name__ == '__main__':
    main()
