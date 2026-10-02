#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_khalil_orbits.py -- the AL-TAQALIB deliverable, proved.

Proves, against the real release inventory and the real al-ʿAyn record:
  T1  `KhalilPermutationOrbits` is importable from where test_grammar_impl.py imports it
      (`models.khalil_combinatorics`) and from the release root (`khalil_orbits`).
  T2  علم and عمل are in ONE orbit and END UP SHARING embedding parameters.
  T3  A permutation al-ʿAyn marks مهمل (عدب, bāb al-ʿayn wa-l-dāl wa-l-bāʾ) is NOT tied to
      its orbit-mates (negative control).
  T4  The orbit is built over ATTESTED permutations only, not all 6: the ع ل م orbit has the
      four faces al-ʿAyn lists as مستعملات (and not لعم / ملع).
  T5  Every orbit member is positively attested in data/khalil_attest_v4.json (4,381 / 2,750).
  T6  `loss` behaves (0 when tied, >0 untied) and is differentiable.
  T7  Tying is SHAPE-PRESERVING, i.e. checkpoint-safe; and `OrbitTiedEmbedding` -- the true
      storage-sharing variant -- has a different shape, which is why it is wired nowhere.
  T8  The Farahidi rank table used to pick representatives mirrors the one in
      models/khalil_combinatorics.py, and the representative is the chapter headword.
  T9  Radical counts beyond 3: what the al-ʿAyn record does and does not support.
  T10 Tying the tables the SHIPPED checkpoints actually carry (24 per-layer tables), where the
      9015-row vs 9,114-root mismatch is reported, never forced.

FIDELITY CAVEAT (mandatory): this is al-Khalīl's own مستعمل/مهمل filter in *Kitāb al-ʿAyn*
(«يُكَتَب مُسْتَعْمَلها. ويُلغى مُهْمَلها»).  It is NOT al-ishtiqāq al-akbar: the claim that the
permutations of a root share one MEANING is reported and REJECTED by the tradition -- Ibn
ʿUsfūr, *al-Mumtiʿ fī al-Taṣrīf*: «ولم يقل به أحد من النحويين إلا أبا الفتح ... والصحيح أن هذا
النحو من الاشتقاق غير مأخوذ به؛ لعدم اطراده» -- and is not implemented here.

Usage: /workspace/venvs/rootformer/bin/python test_khalil_orbits.py
"""
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models'))

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))


def main():
    import nrmp_vocab as nv
    cls = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = cls(str(ROOT / 'data/rootformer_v12_arabic_blueprint.json'))
    rec = json.loads((ROOT / 'data/khalil_attest_v4.json').read_text(encoding='utf-8'))

    print('=== T1: importability from both entry points ===')
    try:
        from models.khalil_combinatorics import KhalilPermutationOrbits as K1  # audit's path
        ok1 = True
        d1 = 'models.khalil_combinatorics'
    except Exception as e:
        K1, ok1, d1 = None, False, f'{type(e).__name__}: {e}'
    try:
        from khalil_orbits import KhalilPermutationOrbits as K2  # release-root path
        ok2 = True
        d2 = 'khalil_orbits'
    except Exception as e:
        K2, ok2, d2 = None, False, f'{type(e).__name__}: {e}'
    check('KhalilPermutationOrbits importable from models.khalil_combinatorics', ok1, d1)
    check('KhalilPermutationOrbits importable from khalil_orbits', ok2, d2)
    if not (ok1 or ok2):
        return finish()
    KPO = K1 or K2

    orbits = KPO(vocab)
    st = orbits.stats()
    print('  ' + orbits.report().replace('\n', '\n  '))

    print('\n=== T2: علم and عمل are one orbit AND share parameters ===')
    fam_lm = orbits.orbit_of('علم')
    fam_am = orbits.orbit_of('عمل')
    check('علم and عمل are in one orbit', fam_lm == fam_am and 'عمل' in fam_lm,
          f'orbit(علم)={fam_lm}')
    w = torch.randn(vocab.num_roots, 16, generator=torch.Generator().manual_seed(0))
    i_lm, i_am = vocab.root2id['علم'], vocab.root2id['عمل']
    distinct_before = not torch.equal(w[i_lm], w[i_am])
    check('before tying the two rows are independent (control)', distinct_before)
    stats = orbits.tie_embeddings(w)
    check('after tie_embeddings the two rows are IDENTICAL (parameter sharing)',
          torch.equal(w[i_lm], w[i_am]),
          f'tied {stats["orbits_tied"]} orbits / {stats["rows_tied"]} rows, '
          f'{stats["rows_collapsed"]} collapsed, method={stats["method"]}')
    check('all four ع ل م orbit members share the same row',
          all(torch.equal(w[vocab.root2id[m]], w[i_lm]) for m in fam_lm), f'{fam_lm}')

    print('\n=== T3: a مهمل permutation is NOT tied (negative control) ===')
    fam_abd = orbits.orbit_of('عبد')
    muhmal = orbits.excluded('عبد')['muhmal']
    check('al-ʿAyn bāb ع د ب marks عدب (and دبع) مهمل',
          'عدب' in muhmal, f'muhmal under the ع د ب key: {muhmal}; '
          f'«باب العين والدال والباء معهما ع ب د- د ع ب- ب ع د- ب د ع مستعملات ع د ب- د ب ع مهملان»')
    check('عدب is NOT in orbit(عبد)', 'عدب' not in fam_abd, f'orbit(عبد)={fam_abd}')
    if 'عدب' in vocab.root2id:
        check('after tying, عدب keeps its own row (not shared with عبد)',
              not torch.equal(w[vocab.root2id['عدب']], w[vocab.root2id['عبد']]),
              'مهمل stays untied -- «ويُلغى مُهْمَلها»')
    else:
        check('after tying, عدب keeps its own row (not shared with عبد)', False,
              'عدب absent from the inventory')

    print('\n=== T4: the orbit is over ATTESTED faces only, not all 6 ===')
    check('orbit(علم) has exactly the four faces al-ʿAyn lists as مستعملات',
          set(fam_lm) == {'علم', 'عمل', 'معل', 'لمع'} and len(fam_lm) == 4,
          f'{fam_lm}  («باب العين واللاّم والميم معهما ع ل م، ع م ل، م ع ل، ل م ع مستعملات»)')
    check('the unlisted faces لعم and ملع are not tied in',
          not ({'لعم', 'ملع'} & set(fam_lm)), f'excluded={orbits.excluded("علم")}')
    check('the six raw permutations exist but only four are family',
          len(orbits.permutations_of('علم')) == 6 and len(fam_lm) == 4,
          f'all 6 = {orbits.permutations_of("علم")}')

    print('\n=== T5: membership is grounded in the al-ʿAyn record ===')
    check('record counts are the documented 4,381 attested / 2,750 مهمل',
          st['attested_keys'] == 4381 and st['muhmal_keys'] == 2750,
          f"file counts={st['attest_record_counts']} parsed={st['attested_keys']}/{st['muhmal_keys']}")
    members = [m for fam in orbits.orbits.values() for m in fam]
    not_attested = [m for m in members if not orbits.is_attested(m)]
    check('every tied member is positively attested', not not_attested,
          f'{len(not_attested)} violations e.g. {not_attested[:5]}')
    muhmal_tied = [m for m in members if orbits.is_muhmal(m)]
    check('no مهمل root is ever a tied member', not muhmal_tied,
          f'{len(muhmal_tied)} violations e.g. {muhmal_tied[:5]}')
    check('al-ʿAyn مهمل roots present in the inventory are kept out',
          st['muhmal_excluded_roots'] == 425,
          f"{st['muhmal_excluded_roots']} inventory roots marked مهمل in the record were excluded")

    print('\n=== T6: loss ===')
    w2 = torch.randn(vocab.num_roots, 16, generator=torch.Generator().manual_seed(1),
                     requires_grad=True)
    l_before = float(orbits.loss(w2))
    with torch.no_grad():
        orbits.tie_embeddings(w2)
    l_after = float(orbits.loss(w2))
    check('loss > 0 untied and ~0 tied', l_before > 0 and l_after < 1e-8,
          f'before={l_before:.6g} after={l_after:.3g}')
    w3 = torch.randn(vocab.num_roots, 16, generator=torch.Generator().manual_seed(2),
                     requires_grad=True)
    orbits.loss(w3).backward()
    check('loss is differentiable (grad reaches the embedding)',
          w3.grad is not None and float(w3.grad.abs().sum()) > 0,
          f'grad_norm={float(w3.grad.norm()):.4g}' if w3.grad is not None else 'grad is None')
    check('cosine-mode loss agrees in sign with the legacy orbit loss',
          float(orbits.loss(w3, mode='cosine')) > 0,
          f'cosine={float(orbits.loss(w3, mode="cosine")):.4g}')

    print('\n=== T7: checkpoint safety ===')
    emb = torch.nn.Embedding(vocab.num_roots, 8)
    keys_before = sorted(emb.state_dict().keys())
    shape_before = tuple(emb.weight.shape)
    orbits.tie_embeddings(emb.weight)
    keys_after = sorted(emb.state_dict().keys())
    check('tying leaves state_dict keys and shape unchanged (shape-preserving)',
          keys_before == keys_after == ['weight'] and tuple(emb.weight.shape) == shape_before,
          f'keys={keys_after} shape={shape_before}')
    fresh = torch.nn.Embedding(vocab.num_roots, 8)
    fresh.load_state_dict(emb.state_dict())  # exactly what a checkpoint path does
    check('a tied state_dict still loads into a fresh embedding',
          torch.equal(fresh.weight[vocab.root2id['علم']], fresh.weight[vocab.root2id['عمل']]))
    from khalil_orbits import OrbitTiedEmbedding
    ote = OrbitTiedEmbedding(orbits, vocab.num_roots, 8)
    check('OrbitTiedEmbedding (storage sharing) has a DIFFERENT shape -> checkpoint-breaking',
          tuple(ote.weight.shape)[0] != vocab.num_roots,
          f'weight={tuple(ote.weight.shape)} keys={ote.state_dict_keys()} '
          f'vs released morphemic_embed.root_embed.weight=({vocab.num_roots}, d_root)')

    print('\n=== T8: the Farahidi rank table used for representatives is the mirrored one ===')
    try:
        from models import khalil_combinatorics as kc
        import khalil_orbits as ko
        check('khalil_orbits.FARAHIDI_ALPHABET_ORDER == khalil_combinatorics.FARAHIDI_ALPHABET_ORDER',
              list(kc.FARAHIDI_ALPHABET_ORDER) == list(ko.FARAHIDI_ALPHABET_ORDER))
    except Exception as e:
        check('Farahidi table mirror check', False, f'{type(e).__name__}: {e}')
    check('orbit representative is the Farahidi-first face (chapter headword)',
          fam_lm[0] == 'علم' and fam_abd[0] == 'عبد',
          f'representatives: ع ل م -> {fam_lm[0]}, ع د ب -> {fam_abd[0]}')

    print('\n=== T9: radical counts beyond 3 -- what the al-ʿAyn record actually supports ===')
    # al-Khalīl's sentence covers 2/3/4/5 (وجهان / ستة أوجه / أربعة وعشرين وجها /
    # مائة وعشرين وجها).  ENGINEERING MEASUREMENT over the shipped data: the attest record
    # holds only 29 quadriliteral attestations and no pentliteral ones, so under the faithful
    # 'attested' policy NO length-4 or length-5 orbit reaches two members and the mechanism is
    # exercised at S^3 only.  Under the looser policy the 4/5-face families do appear, which is
    # what shows that n! > 3 is handled.
    counts = {n: KPO(vocab, lengths=(n,)).stats() for n in (2, 3, 4, 5)}
    check('attested-only: only S^3 reaches a tieable orbit in this data',
          counts[3]['orbits'] == 967 and counts[2]['orbits'] == 0
          and counts[4]['orbits'] == 0 and counts[5]['orbits'] == 0,
          'orbits by radical count=' + str({n: counts[n]['orbits'] for n in (2, 3, 4, 5)})
          + "; the record holds 29 quadriliteral and 0 pentliteral attestations, so al-Khalīl's "
            '«أربعة وعشرين وجها» family has no attested pair to tie in this inventory')
    loose = KPO(vocab, lengths=(2, 3, 4, 5), membership='inventory_not_muhmal')
    by_len = Counter(len(k) for k in loose.orbits)
    check('4- and 5-radical families build (n! > 3 handled) under the looser policy',
          by_len.get(4, 0) > 0 and by_len.get(5, 0) > 0
          and max(loose.stats()['orbit_face_size_histogram']) <= 120,
          f'orbits by radical count={dict(sorted(by_len.items()))}, '
          f"faces={loose.stats()['orbit_face_size_histogram']}")
    o_inv = KPO(vocab, membership='inventory_not_muhmal')
    check("alternative policy 'inventory_not_muhmal' ties strictly more",
          o_inv.stats()['pairs'] > st['pairs'],
          f"attested={st['pairs']} pairs / {st['orbits']} orbits vs "
          f"inventory_not_muhmal={o_inv.stats()['pairs']} pairs / {o_inv.stats()['orbits']} orbits")
    check('even the looser policy never ties a مهمل root',
          all(not o_inv.is_muhmal(m) for fam in o_inv.orbits.values() for m in fam),
          f"{o_inv.stats()['muhmal_excluded_roots']} مهمل inventory roots excluded")

    print('\n=== T10: the tables the SHIPPED checkpoints actually carry ===')
    from khalil_orbits import tie_root_embedding_tables

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Module() for _ in range(2)])
            for i, mod in enumerate(self.layers):
                mod.self_attn = torch.nn.Module()
                mod.self_attn.root_embed = torch.nn.Embedding(vocab.num_roots, 4)

    fb = FakeBackbone()
    res = tie_root_embedding_tables(fb, orbits)
    got = [dict(fb.named_parameters())[n] for n in res['tied']]
    check('module root_embed tables are found and tied',
          res['n_tied_tables'] == 2 and all(
              torch.equal(t[vocab.root2id['علم']], t[vocab.root2id['عمل']]) for t in got),
          f"tied={res['n_tied_tables']} skipped={res['n_skipped_tables']}")

    ck = ROOT / 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors'
    if ck.is_file():
        from safetensors import safe_open
        with safe_open(str(ck), framework='pt') as fh:
            tables = {k: fh.get_tensor(k) for k in fh.keys()
                      if k.endswith('self_attn.root_embed.weight')}
        res2 = tie_root_embedding_tables(tables, orbits)
        n_t, n_s = res2['n_tied_tables'], res2['n_skipped_tables']
        check('shipped checkpoint root tables are tied or REPORTED, never resized',
              n_t + n_s == len(tables) and all('rows=' in v for v in res2['skipped'].values()),
              f'tied={n_t} skipped={n_s} of {len(tables)} tables; '
              + (list(res2['skipped'].values())[0] if res2['skipped'] else 'no mismatch'))
    else:
        check('shipped checkpoint root tables are tied or REPORTED, never resized', False,
              f'checkpoint not found: {ck}')
    return finish()


def finish():
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print('\n' + '=' * 74)
    print(f'RESULT: {len(RESULTS) - n_fail}/{len(RESULTS)} checks pass, {n_fail} FAIL')
    print('=' * 74)
    for name, ok, detail in RESULTS:
        if not ok:
            print(f'  FAIL  {name}')
            if detail:
                print(f'        {detail}')
    return 1 if n_fail else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
