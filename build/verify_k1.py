#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Before/after proof of the Al-Khalīl K1 fix in SibawayhNRMPGovernance.

Loads the pre-change class from sibawayh_governance_engine.py.bak and the current one from
sibawayh_governance_engine.py, on the same vocabulary, and compares khalil_forbidden_roots
against the Farāhīdian lexicon the audit uses.
"""
import importlib.machinery
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models'))
LEXICON = Path('/workspace/rootformer_v12/v18_next_root_morph/data/farahidian_3pillar_lexicon_clean.json')
SPOT = ['اصل', 'ارض', 'اخذ', 'احد', 'اسس', 'افل', 'اثر']
RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append(ok)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))


def load(path, name):
    # a .bak suffix is not a source suffix, so load it through an explicit SourceFileLoader
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def main():
    import torch                                    # noqa: F401  (the engine needs it)
    import nrmp_vocab as nv
    cls = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = cls(str(ROOT / 'data/rootformer_v12_arabic_blueprint.json'))
    lex = json.load(open(LEXICON, encoding='utf-8')) if LEXICON.exists() else {}
    print(f'[*] lexicon entries: {len(lex)}  | vocab roots: {vocab.num_roots}')

    old_mod = load(ROOT / 'sibawayh_governance_engine.py.bak', 'old_engine')
    new_mod = load(ROOT / 'sibawayh_governance_engine.py', 'new_engine')
    old = old_mod.SibawayhNRMPGovernance(vocab)
    new = new_mod.SibawayhNRMPGovernance(vocab)
    fo, fn = old.khalil_forbidden_roots, new.khalil_forbidden_roots
    R = vocab.id2root

    def bare3(forb):
        return [R[i] for i in sorted(forb) if len(R[i]) == 3 and R[i].startswith('ا')]

    lo, ln = [r for r in bare3(fo) if r in lex], [r for r in bare3(fn) if r in lex]
    print(f'\n--- forbidden-root mask ---')
    print(f'  BEFORE: {len(fo)} roots forbidden, of which {len(lo)} are REAL bare-alif roots')
    print(f'  AFTER : {len(fn)} roots forbidden, of which {len(ln)} are REAL bare-alif roots')
    check('no legitimate hamza-initial root is excluded', len(ln) == 0, f'{len(ln)} left (was {len(lo)})')
    killed_o = [r for r in SPOT if r in vocab.root2id and vocab.root2id[r] in fo]
    killed_n = [r for r in SPOT if r in vocab.root2id and vocab.root2id[r] in fn]
    print(f'  spot-check {SPOT}: killed BEFORE={killed_o} AFTER={killed_n}')
    check('spot-check اصل/ارض/اخذ/احد/اسس/افل/اثر allowed', not killed_n, f'still killed: {killed_n}')

    def c12(forb):
        return [R[i] for i in sorted(forb) if len(R[i]) == 3 and R[i][0] == R[i][1]]
    c_o, c_n = c12(fo), c12(fn)
    print(f'  C1==C2 excluded: BEFORE={len(c_o)} AFTER={len(c_n)}  e.g. {c_n[:6]}')
    check('C1==C2 roots are STILL excluded (the genuine rule holds)', len(c_n) > 0,
          f'{len(c_n)} excluded (was {len(c_o)})')
    check('the C1==C2 set is preserved exactly (no genuine exclusion lost)',
          set(c_o) == set(c_n), f'{len(set(c_o) ^ set(c_n))} symmetric difference')

    # the other genuine exclusions
    leaks = [r for r in ('end', 'start') if r in vocab.root2id and vocab.root2id[r] in fn]
    parts = [i for i in (vocab.root2id.get('<PARTICLE>'), vocab.PAD_ROOT, vocab.BOS_ROOT,
                         vocab.UNK_ROOT) if i is not None and i in fn]
    check("leaked control strings ('end','start') still excluded", len(leaks) == 2, f'{leaks}')
    check('PAD/BOS/UNK/<PARTICLE> still never generatable', len(parts) == 4, f'{len(parts)}/4')
    check('EOS is NOT masked (the decoder must be able to stop)',
          vocab.EOS_ROOT not in fn)
    verbatim = [r for r in vocab.roots_list if r in ('جقق',)]
    check('Kitāb al-ʿAyn verbatim pairs still have teeth (جقق stays excluded)',
          all(vocab.root2id[r] in fn for r in verbatim if r in vocab.root2id), f'{verbatim}')

    # how many REAL lexicon roots remain forbidden at all (informational; C1==C2 junk may be)
    all_lex_forbidden = sorted(r for r in {R[i] for i in fn} if r in lex)
    print(f'\n  lexicon roots still forbidden (any reason, informational): '
          f'{len(all_lex_forbidden)} {all_lex_forbidden[:10]}')

    print()
    n_fail = RESULTS.count(False)
    print(f'RESULT: {len(RESULTS)-n_fail}/{len(RESULTS)} checks pass')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
