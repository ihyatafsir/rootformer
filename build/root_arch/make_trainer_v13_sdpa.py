#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_trainer_v13_sdpa.py -- generate `nrmt_train_v13_sdpa.py` from the PROVEN `nrmt_train_v13.py`.

`nrmt_train_v13.py` is READ-ONLY here and is NOT edited: it is the verified artefact another agent
built, and the root pathway inside it must not be rebuilt.  This script writes a sibling plus a
unified diff, exactly the way `make_trainer_v13.py` generated v13 from the A2 trainer.

WHAT THE SIBLING ADDS (seven patches, nothing else)
-------------------------------------------------
  P1  sys.path += /workspace/root_arch
  P2  --sdpa and --pillar-freeze flags
  P3  install the SDPA attention classes BEFORE the model is constructed, so
      `unified_rootformer_v13` imports `IshtiqaqAttentionV12` already swapped.  (Phase 0a.)
  P4  --pillar-freeze: hold the AynEngine Pillar-I/II gate at exactly 0 and take it OUT of the
      optimizer.  Rationale, in full:
        * the Pillar-I/II terms are a function of the POST-SOFTMAX attention weights, which
          `F.scaled_dot_product_attention` does not return, so they cannot be reproduced by a
          fused kernel;
        * they are NOT one of the two declared mechanisms (residual injector, native score bias),
          so leaving them live would confound the A-vs-C comparison with an undeclared third term;
        * `pillar_gate` is initialised to 0.0 and multiplied by `coverage_weight=2.0` /
          `governance_strength=1.5` (both LIVE from the checkpoint), so it is a no-op at step 0 --
          but it IS in `root_path_parameters()`, so the stock trainer hands it to the optimizer and
          it would leave 0 and then trip the SDPA guard.  Freezing it makes the skip STRUCTURAL
          rather than a `0.0 * finite` accident.
      This does not regress the proven init-equivalence (gate 0 == no pathway, bit-equal): it
      makes that identity hold for EVERY step instead of only at step 0.
  P6  hand the root pathways a CLONE, so nothing can mutate the tensor `morphemic_embed`
      already saved for backward.
  P7  install `rca_compat`, which wraps `RootHistoryCrossAttention.forward` so each layer clamps
      its OWN root-id tensor -- the workaround for the in-place `clamp_` at
      `root_cross_attn.py:136`, WHICH IS NOT EDITED.
  P5  a PREFLIGHT assertion on the root count before anything expensive happens.  A missing
      release module (`validated_segmentation.py`) makes `nrmp_vocab` report 9445 roots instead
      of 9490; nothing downstream raises, so every number silently shifts.  This refuses to run.
"""
import difflib
import hashlib
import pathlib
import sys

SRC = pathlib.Path('/workspace/transmute_v2/nrmt_train_v13.py')
DST = pathlib.Path('/workspace/root_arch/nrmt_train_v13_sdpa.py')
DIFF = pathlib.Path('/workspace/root_arch/nrmt_train_v13_sdpa.diff')


def md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def sub_once(text, old, new, tag):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f'[make_trainer_v13_sdpa] anchor {tag!r} matched {n} times, expected 1')
    return text.replace(old, new, 1)


def patch(text):
    # ---------------------------------------------------------------- P1: sys.path
    text = sub_once(
        text,
        "if '/workspace/transmute_v2' not in sys.path:\n"
        "    sys.path.insert(0, '/workspace/transmute_v2')\n",
        "if '/workspace/transmute_v2' not in sys.path:\n"
        "    sys.path.insert(0, '/workspace/transmute_v2')\n"
        "# root_arch: Phase 0a SDPA attention + the 0d flash-hook decision\n"
        "if '/workspace/root_arch' not in sys.path:\n"
        "    sys.path.insert(0, '/workspace/root_arch')\n",
        'P1 sys.path')

    # ---------------------------------------------------------------- P2: flags
    text = sub_once(
        text,
        "    ap.add_argument('--flash-hooks', choices=('on', 'off'), default='on',\n"
        "                    help='deepseek_v4_1_flash_model registers forward hooks on trunk '\n"
        "                         'layers 1/11/14 whose 151.95M params are in NO checkpoint and NO '\n"
        "                         'optimizer (zero-init INERT).  `off` detaches them.')\n",
        "    ap.add_argument('--flash-hooks', choices=('on', 'off'), default='on',\n"
        "                    help='deepseek_v4_1_flash_model registers forward hooks on trunk '\n"
        "                         'layers 1/11/14 whose 151.95M params are in NO checkpoint and NO '\n"
        "                         'optimizer (zero-init INERT).  `off` detaches them.')\n"
        "    # ---- PHASE 0a: fused attention ---------------------------------------------------\n"
        "    ap.add_argument('--sdpa', action='store_true',\n"
        "                    help='replace the hand-materialised [B,14,T,T] attention with '\n"
        "                         'F.scaled_dot_product_attention (root_arch/sdpa_attention.py). '\n"
        "                         'Installed BEFORE the model is built, so the backbone is '\n"
        "                         'constructed from the SDPA classes outright.')\n"
        "    ap.add_argument('--pillar-freeze', action='store_true',\n"
        "                    help='hold the AynEngine Pillar-I/II gate at EXACTLY 0 and remove it '\n"
        "                         'from the optimizer, so the only root score term is the declared '\n"
        "                         'score bias.  Required with --sdpa: the pillar block needs the '\n"
        "                         'post-softmax weights a fused kernel does not return.')\n",
        'P2 flags')

    # ---------------------------------------------------------------- P3: install SDPA
    text = sub_once(
        text,
        "    args = ap.parse_args()\n\n    if args.seed >= 0:\n",
        "    args = ap.parse_args()\n"
        "\n"
        "    # P3 -- Phase 0a.  MUST run before any model is constructed: `unified_rootformer_v13`\n"
        "    # binds `IshtiqaqAttentionV12` at import time, so patching the module afterwards would\n"
        "    # leave the already-built backbone on the eager path.\n"
        "    if args.sdpa:\n"
        "        import sdpa_attention as _sdpa\n"
        "        _sdpa.install(verbose=True)\n"
        "        print('[*] SDPA installed BEFORE model construction (Phase 0a)', flush=True)\n"
        "        # PROVE the fused kernel is used instead of assuming it.  One attention call at\n"
        "        # B=8, T=1024 must allocate O(T); a MATH-backend fallback would allocate the whole\n"
        "        # [B,14,T,T] matrix (224 MB) and show up as ratio >= 1.\n"
        "        for _b in (False, True):\n"
        "            try:\n"
        "                _v = _sdpa.verify_dispatch(bias=_b)\n"
        "                print(f'[*] SDPA dispatch bias={_b}: peak {_v[\"peak_MB\"]:.2f} MB vs eager'\n"
        "                      f' [B,14,T,T] {_v[\"eager_matrix_MB\"]:.1f} MB (ratio {_v[\"ratio\"]:.3f})'\n"
        "                      f' -> fused kernel used = {_v[\"fused_kernel_used\"]}', flush=True)\n"
        "                if not _v['fused_kernel_used']:\n"
        "                    print('[warn] SDPA materialises [B,14,T,T]; the memory saving will NOT'\n"
        "                          ' materialise', flush=True)\n"
        "            except Exception as _e:\n"
        "                print(f'[warn] SDPA dispatch verification failed: {_e}', flush=True)\n"
        "\n"
        "    if args.seed >= 0:\n",
        'P3 sdpa install')

    # ---------------------------------------------------------------- P4: pillar freeze
    text = sub_once(
        text,
        "        for _p in ishtiqaq_params:\n"
        "            _p.requires_grad_(True)\n"
        "        print(f'[*] NATIVE root score bias on trunk layers {ishtiqaq_layers} '\n",
        "        for _p in ishtiqaq_params:\n"
        "            _p.requires_grad_(True)\n"
        "        if args.pillar_freeze:\n"
        "            # P4 -- see make_trainer_v13_sdpa.py.  Structural, not numeric: the gate is\n"
        "            # zeroed AND made non-trainable AND dropped from every optimizer group, so it\n"
        "            # can never leave 0 and the fused kernel never has to reproduce the pillar\n"
        "            # block.  `_ig_ids` (below) is built AFTER this, so the parameter cannot leak\n"
        "            # into `trunk_params` either.\n"
        "            _pg = set()\n"
        "            for _m in ishtiqaq_mods:\n"
        "                with torch.no_grad():\n"
        "                    _m.pillar_gate.data.zero_()\n"
        "                _m.pillar_gate.requires_grad_(False)\n"
        "                _pg.add(id(_m.pillar_gate))\n"
        "            ishtiqaq_params[:] = [p for p in ishtiqaq_params if id(p) not in _pg]\n"
        "            print(f'[*] PILLAR-FREEZE: {len(_pg)} pillar gates held at exactly 0.0 and '\n"
        "                  f'removed from the optimizer (coverage_weight / governance_strength '\n"
        "                  f'stay at their checkpoint values but are multiplied by 0)', flush=True)\n"
        "        print(f'[*] NATIVE root score bias on trunk layers {ishtiqaq_layers} '\n",
        'P4 pillar freeze')

    # ---------------------------------------------------------------- P6: clone the root ids
    text = sub_once(
        text,
        "        if rca_stack is not None:\n"
        "            rca_stack.set_root_ids(Tr_w)\n"
        "        # the ONLY wiring the shipped native root path has ever needed\n"
        "        for _i in ishtiqaq_layers:\n"
        "            model.backbone.layers[_i].self_attn.active_root_ids = Tr_w\n",
        "        # P6 -- `root_cross_attn.py:136` does `safe = root_ids.clamp_(...)` IN PLACE on the\n"
        "        # tensor it is handed.  Under `--unfreeze-trunk-all` the `morphemic_embed` lookup\n"
        "        # above has ALREADY SAVED Tr_w for backward, so that in-place bump kills backward\n"
        "        # with: 'one of the variables needed for gradient computation has been modified\n"
        "        # by an inplace operation: [torch.cuda.LongTensor [1024]] is at version 24;\n"
        "        # expected version 23'.  The recorded arms never hit it because their\n"
        "        # `morphemic_embed` was FROZEN (no embedding_backward -> indices not saved -> no\n"
        "        # version check).  Handing the pathways a CLONE leaves the saved tensor untouched.\n"
        "        # This is an UPSTREAM bug in another agent's module: reported, not silently fixed.\n"
        "        _rid = Tr_w.clone() if (rca_stack is not None or ishtiqaq_layers) else Tr_w\n"
        "        if rca_stack is not None:\n"
        "            rca_stack.set_root_ids(_rid)\n"
        "        # the ONLY wiring the shipped native root path has ever needed\n"
        "        for _i in ishtiqaq_layers:\n"
        "            model.backbone.layers[_i].self_attn.active_root_ids = _rid\n",
        'P6 clone root ids')

    # ---------------------------------------------------------------- P7: rca clamp compat
    text = sub_once(
        text,
        """        from root_cross_attn import RootCrossAttentionStack
        from early_root_path import parse_layer_spec_v13 as parse_layer_spec
""",
        """        # P7 -- defuse the IN-PLACE `safe = root_ids.clamp_(...)` at root_cross_attn.py:136
        # WITHOUT editing that verified module.  `clamp_` returns self, so `safe` IS the shared
        # root-id tensor; every attached layer then saves a VIEW of it in IndexSelectBackward0, and
        # the NEXT layer's clamp_ bumps its version, so the first backward dies with
        #   "modified by an inplace operation ... [LongTensor [512]] is at version 24; expected 23"
        # It only fires once `morphemic_embed` is trainable (index_select builds no grad node while
        # root_embed.weight is frozen), which is exactly this experiment.  rca_compat wraps the
        # method so each layer clamps a tensor it owns.  Numerically identical.
        import rca_compat
        rca_compat.install(verbose=True)
        from root_cross_attn import RootCrossAttentionStack
        from early_root_path import parse_layer_spec_v13 as parse_layer_spec
""",
        'P7 rca_compat')

    # ---------------------------------------------------------------- P5: root-count preflight
    text = sub_once(
        text,
        "    vocab = V(bp)\n",
        "    vocab = V(bp)\n"
        "    # P5 -- the silent-shrink trap.  nrmp_vocab reports 9445 instead of 9490 when a release\n"
        "    # module is missing from sys.path; NO downstream table raises, so every number computed\n"
        "    # afterwards is quietly wrong.  Fail here, before the 5-minute model build.\n"
        "    if (int(vocab.num_roots), int(vocab.num_awzan)) != (9490, 142):\n"
        "        raise SystemExit(\n"
        "            f'[PREFLIGHT] SILENT VOCAB SHRINK: nrmp_vocab gives {vocab.num_roots} roots / '\n"
        "            f'{vocab.num_awzan} awzan, expected 9490 / 142.  Refusing to train on a '\n"
        "            f'shrunken root space.')\n"
        "    print(f'[*] PREFLIGHT root count OK: roots={vocab.num_roots} awzan={vocab.num_awzan}',\n"
        "          flush=True)\n",
        'P5 preflight')
    return text


def main():
    if not SRC.exists():
        raise SystemExit(f'[make_trainer_v13_sdpa] missing {SRC}')
    src = SRC.read_text()
    out = patch(src)
    if (int(sys.argv[1]) if len(sys.argv) > 1 else 1) and DST.exists() and DST.read_text() == out:
        print(f'[make_trainer_v13_sdpa] {DST.name} already up to date')
    else:
        DST.write_text(out)
    DIFF.write_text(''.join(difflib.unified_diff(
        src.splitlines(keepends=True), out.splitlines(keepends=True),
        fromfile='_ref_nrmt_train_v13.py', tofile='nrmt_train_v13_sdpa.py')))
    print(f'[make_trainer_v13_sdpa] source  {SRC}  md5={md5(SRC)}')
    print(f'[make_trainer_v13_sdpa] target  {DST}  md5={md5(DST)}  '
          f'({len(out.splitlines())} lines)')
    print(f'[make_trainer_v13_sdpa] diff    {DIFF}  ({sum(1 for _ in DIFF.open())} lines)')
    import ast
    ast.parse(out)                      # the generated trainer must PARSE
    for probe in ('--sdpa', '--pillar-freeze', '_sdpa.install', 'PILLAR-FREEZE',
                  'SILENT VOCAB SHRINK', '_rid = Tr_w.clone()', 'rca_compat.install'):
        assert probe in out, f'patch did not land: {probe}'
    # `'verify_dispatch' in out` is NOT enough -- an earlier draft emitted the whole block with
    # literal backslash-n, so it became ONE COMMENT LINE and the check still passed while the code
    # never ran.  Assert on the PARSED AST instead: a `for` loop whose body CALLS verify_dispatch.
    tree = ast.parse(out)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'verify_dispatch']
    assert calls, ('the generated trainer does not actually CALL verify_dispatch -- it is present '
                   'only as text (most likely commented out by a bad \\n escape)')
    n_verify_lines = sum(1 for ln in out.splitlines() if 'verify_dispatch' in ln
                         and not ln.lstrip().startswith('#'))
    print(f'[make_trainer_v13_sdpa] verify_dispatch is LIVE CODE: {len(calls)} call site(s), '
          f'{n_verify_lines} non-comment line(s)')
    print('[make_trainer_v13_sdpa] all 7 patches verified present')


if __name__ == '__main__':
    main()
