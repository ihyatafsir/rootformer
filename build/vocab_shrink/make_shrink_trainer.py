#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_shrink_trainer.py -- generate `nrmt_train_shrink.py` from the SHIPPED
`/workspace/hf_v19_2_release/nrmt_train.py` by a small, explicit, exactly-once patch set.

The shipped module is NEVER modified.  The generated file lives in /workspace/vocab_shrink/ and
is the only thing the shrunk-vocab arm executes.  Emits a unified diff for review.
"""
import difflib
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else '/workspace/hf_v19_2_release/nrmt_train.py')
DST = Path(sys.argv[2] if len(sys.argv) > 2 else '/workspace/vocab_shrink/nrmt_train_shrink.py')
DIFF = DST.with_suffix('.diff')

PATCHES = [
    # ---- 1. module handle for the head subclass -----------------------------------------
    ("    from nrmt_arch import RootformerNRMT, build_operator_table, ngram_order_vocabs\n",
     "    from nrmt_arch import RootformerNRMT, build_operator_table, ngram_order_vocabs\n"
     "    import nrmt_arch as _na                      # ARTIFACT: shrunk-root-head patch\n"),

    # ---- 2. the shrunk root head, installed before the model is built --------------------
    ("    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)\n",
     '''    # ================= ARTIFACT: SHRUNK ROOT-VOCAB ARM ==================================
    # Hypothesis under test: (A) cardinality / data-per-class -- 9490 root classes at ~25M
    # words is simply too many -- versus (B) a deeper architecture problem, i.e. the root slot
    # is not being optimised or the signal is unreachable from the shared trunk.
    #
    # WHAT CHANGES: `root_head` emits K classes (top-K real roots by TRAIN frequency, map built
    # by analyze_roots.py from the train stream ONLY) and the root CE / root metrics use those K
    # CLASS ids as targets.
    # WHAT DOES NOT CHANGE: the image-aligned cache, the frozen backbone, the cached hidden
    # states, the morphemic/history/operator conditioning (all still indexed by ORIGINAL root
    # ids, so the trunk input is byte-identical), the optimiser, LR schedule, batch size, step
    # count (20 000), eval windows/positions, novelty masking, and the impossibility hinge (its
    # forbidden set is re-expressed in the K classes; on the top-500 roots it is EMPTY, so the
    # hinge is exactly as active as it is on those same positions in the full-vocab arm, i.e.
    # not at all).
    root_map = None
    if getattr(args, 'root_classes', 0):
        _rm = torch.load(args.root_map, map_location='cpu')
        class_to_orig = _rm['class_to_orig'].long()            # (K,) rank -> original root id
        orig_to_class = _rm['orig_to_class'].long()            # (Nfull,) -1 => out of vocabulary
        assert int(_rm['topk']) == int(args.root_classes), (_rm['topk'], args.root_classes)
        assert int(orig_to_class.numel()) == vocab.num_roots
        root_map = (class_to_orig, orig_to_class)
        assert args.oov == 'drop', (
            'the OTHER design was REJECTED: on the identical eval positions the fold-in class '
            'holds 26.85% of the mass and would dominate as a trivial catch-all, while raising '
            'data-per-real-class only 1.67x (426 -> 711 tokens) against 12.3x for drop '
            '(426 -> 5237).  Only drop is a real cardinality reduction.')
        print(f'[*] SHRUNK ROOT VOCAB: K={args.root_classes} classes from train-frequency map '
              f'{args.root_map} (design={args.oov})')

        class _ShrunkRootHead(_na.NRMTHead):
            """NRMTHead whose `root_head` emits K classes instead of num_roots.

            `cond_roots` and the free-running argmax are CLASS ids; they are mapped back to the
            representative ORIGINAL root id before the shared root embedding, so the root
            conditioning channel keeps exactly the semantics it has in the full-vocab arm.
            """
            _MAP = None

            def __init__(self, d_model, num_roots, *a, **kw):
                super().__init__(d_model, num_roots, *a, **kw)
                c2o = type(self)._MAP
                self.root_head = torch.nn.Linear(d_model, int(c2o.numel()), bias=False)
                self.num_root_classes = int(c2o.numel())
                object.__setattr__(self, 'class_to_orig', c2o)

            def forward(self, h, root_ids, op_ids, w_ids, p_ids, s_ids, cond_roots=None):
                h_aug = h + self.build_features(root_ids, op_ids, w_ids, p_ids, s_ids).to(h.dtype)
                root_logits = self.root_head(h_aug)
                cls = root_logits.argmax(-1) if cond_roots is None else cond_roots
                cls = cls.clamp(min=0, max=self.num_root_classes - 1)
                e_root = self._root_embed(self.class_to_orig.to(cls.device)[cls])
                h_cond = self.cond_proj(torch.cat([h_aug, e_root.to(h_aug.dtype)], dim=-1))
                return {'root_logits': root_logits, 'wazn_logits': self.wazn_head(h_cond),
                        'prefix_logits': self.prefix_head(h_cond),
                        'suffix_logits': self.suffix_head(h_cond), 'root_states': h_aug}

        _ShrunkRootHead._MAP = class_to_orig
        _na.NRMTHead = _ShrunkRootHead                                   # patch BEFORE build
    # =====================================================================================

    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
'''),

    # ---- 3. remap gains a K-row warm start for the shrunk root_head ----------------------
    ("def remap_legacy_head(sd, head_state, d_model, cond_init='truncate', allow_local=False):\n",
     "def remap_legacy_head(sd, head_state, d_model, cond_init='truncate', allow_local=False,\n"
     "                      class_to_orig=None):\n"),

    ("        elif local == 'cond_proj.0.weight':\n",
     "        elif local == 'root_head.weight' and class_to_orig is not None:\n"
     "            # SHRUNK ARM: keep the trained row of each retained root, selected by rank.\n"
     "            out[local] = v.to(tgt.dtype)[class_to_orig.to(v.device)]\n"
     "            applied.append(f'{local} {tuple(v.shape)} -> {tuple(tgt.shape)} '\n"
     "                           f'(SHRUNK WARM START: {tgt.shape[0]} kept-root rows by rank)')\n"
     "        elif local == 'cond_proj.0.weight':\n"),

    ("        remapped, applied, exempt = remap_legacy_head(sd, head_state, model.d_model, args.cond_init)\n",
     "        remapped, applied, exempt = remap_legacy_head(\n"
     "            sd, head_state, model.d_model, args.cond_init,\n"
     "            class_to_orig=(root_map[0] if root_map is not None else None))\n"),

    ("        remapped, applied, exempt = remap_legacy_head(hsd, head_state, model.d_model,\n"
     "                                                      'truncate', allow_local=True)\n",
     "        remapped, applied, exempt = remap_legacy_head(\n"
     "            hsd, head_state, model.d_model, 'truncate', allow_local=True,\n"
     "            class_to_orig=(root_map[0] if root_map is not None else None))\n"),

    # ---- 4. hinge forbidden set re-expressed in the K classes ----------------------------
    ("    print(f'[*] static forbidden roots (impossibility hinge): {int(fmask_root.sum())}')\n",
     "    print(f'[*] static forbidden roots (impossibility hinge): {int(fmask_root.sum())}')\n"
     "    if root_map is not None:\n"
     "        fmask_cls = torch.zeros(args.root_classes, dtype=torch.bool)\n"
     "        fmask_cls[torch.isin(root_map[0], torch.tensor(forbidden_ids))] = True\n"
     "    else:\n"
     "        fmask_cls = fmask_root\n"
     "    print(f'[*] forbidden classes actually active in the K={args.root_classes} space: '\n"
     "          f'{int(fmask_cls.sum())}'\n"
     "          + ('' if root_map is None else\n"
     "             ' (none of the top-K roots is in al-Khalil forbidden set, so this term is '\n"
     "             'exactly 0 here and on these same positions in the full arm)'))\n"),

    # ---- 5. eval masks/targets in the shrunk label space + the shrunk marginal -----------
    ("    keep_all = ~torch.isin(Rt_va.reshape(-1), spec_t)\n"
     "    keep_novel = keep_all & novel_mask\n",
     "    if root_map is not None:\n"
     "        tgt_va_cls = root_map[1][Rt_va.reshape(-1)]\n"
     "        keep_all = tgt_va_cls >= 0\n"
     "    else:\n"
     "        tgt_va_cls = Rt_va.reshape(-1)\n"
     "        keep_all = ~torch.isin(tgt_va_cls, spec_t)\n"
     "    keep_novel = keep_all & novel_mask\n"),

    ("    print(f'[*] val radical positions: all={int(keep_all.sum())} novel={int(keep_novel.sum())} '\n"
     "          f'({100*int(keep_novel.sum())/max(int(keep_all.sum()),1):.1f}%)')\n",
     "    print(f'[*] val radical positions: all={int(keep_all.sum())} novel={int(keep_novel.sum())} '\n"
     "          f'({100*int(keep_novel.sum())/max(int(keep_all.sum()),1):.1f}%)')\n"
     "    marg_shrunk = None\n"
     "    if root_map is not None:\n"
     "        _cva = torch.bincount(tgt_va_cls[keep_all], minlength=args.root_classes)\n"
     "        marg_shrunk = float(_cva.max()) / max(int(keep_all.sum()), 1)\n"
     "        _fva = ~torch.isin(Rt_va.reshape(-1), spec_t)\n"
     "        print(f'[*] SHRUNK marginal on the IDENTICAL eval positions: {100*marg_shrunk:.3f}% '\n"
     "              f'(majority class of {args.root_classes}, n={int(keep_all.sum())}); '\n"
     "              f'full-space marginal on the same windows = '\n"
     "              f'{100*float(torch.bincount(Rt_va.reshape(-1)[_fva]).max())/max(int(_fva.sum()),1):.3f}% '\n"
     "              f'(n={int(_fva.sum())})')\n"),

    # ---- 6. step_loss: class targets, class keep-mask, class conditioning ---------------
    ("        cond = None\n"
     "        if p_ss <= 0.0:\n"
     "            cond = tgt_r\n"
     "        elif p_ss < 1.0:\n"
     "            with torch.no_grad():\n"
     "                h_aug = h + head.build_features(R, O, W, P, S).to(h.dtype)\n"
     "                pred = head.root_head(h_aug).argmax(-1)\n"
     "            use_pred = torch.rand(pred.shape, device=pred.device) < p_ss\n"
     "            cond = torch.where(use_pred, pred, tgt_r)\n"
     "        out = head(h, R, O, W, P, S, cond_roots=cond)\n"
     "        keep = ~torch.isin(tgt_r, spec_t.to(tgt_r.device))\n",
     "        if root_map is not None:\n"
     "            tgt_cls = root_map[1].to(tgt_r.device)[tgt_r]      # -1 => out of the K classes\n"
     "            keep = tgt_cls >= 0\n"
     "        else:\n"
     "            tgt_cls = tgt_r\n"
     "            keep = ~torch.isin(tgt_r, spec_t.to(tgt_r.device))\n"
     "        cond = None\n"
     "        if p_ss <= 0.0:\n"
     "            cond = tgt_cls.clamp(min=0)\n"
     "        elif p_ss < 1.0:\n"
     "            with torch.no_grad():\n"
     "                h_aug = h + head.build_features(R, O, W, P, S).to(h.dtype)\n"
     "                pred = head.root_head(h_aug).argmax(-1)\n"
     "            use_pred = torch.rand(pred.shape, device=pred.device) < p_ss\n"
     "            cond = torch.where(use_pred, pred, tgt_cls.clamp(min=0))\n"
     "        out = head(h, R, O, W, P, S, cond_roots=cond)\n"),

    ("        l_root = (F.cross_entropy(rl[keep], tgt_r[keep]) if keep.any()\n"
     "                  else rl.sum() * 0.0)\n",
     "        l_root = (F.cross_entropy(rl[keep], tgt_cls[keep]) if keep.any()\n"
     "                  else rl.sum() * 0.0)\n"),

    ("            info['root_raw'] = float(F.cross_entropy(rr, tgt_r[keep]).detach())\n"
     "            rs = rr.std(-1, keepdim=True).clamp_min(1e-6)\n"
     "            info['root_cez'] = float(\n"
     "                F.cross_entropy((rr - rr.mean(-1, keepdim=True)) / rs, tgt_r[keep]).detach())\n",
     "            info['root_raw'] = float(F.cross_entropy(rr, tgt_cls[keep]).detach())\n"
     "            rs = rr.std(-1, keepdim=True).clamp_min(1e-6)\n"
     "            info['root_cez'] = float(\n"
     "                F.cross_entropy((rr - rr.mean(-1, keepdim=True)) / rs, tgt_cls[keep]).detach())\n"),

    ("        fm = fmask_root.to(tgt_r.device)[tgt_r]\n",
     "        fm = (fmask_cls.to(tgt_r.device)[tgt_cls.clamp(min=0)] if root_map is not None\n"
     "              else fmask_root.to(tgt_r.device)[tgt_r])\n"),

    # ---- 7. evaluate() in the shrunk label space ----------------------------------------
    ("        LG = torch.cat(lg).reshape(-1, vocab.num_roots)\n"
     "        tg = Rt_va.reshape(-1)\n",
     "        LG = torch.cat(lg).reshape(\n"
     "            -1, args.root_classes if root_map is not None else vocab.num_roots)\n"
     "        tg = tgt_va_cls\n"),

    # ---- 8. CLI -------------------------------------------------------------------------
    ("    ap.add_argument('--tag', default='full')\n",
     "    ap.add_argument('--root-classes', type=int, default=0,\n"
     "                    help='ARTIFACT: 0 = shipped full-vocab behaviour.  >0 = shrink the ROOT '\n"
     "                         'CLASSIFIER to this many classes (top-K by train frequency); the '\n"
     "                         'cache, backbone, hidden states and conditioning are untouched.')\n"
     "    ap.add_argument('--root-map', default='/workspace/vocab_shrink/root_map_top500.pt')\n"
     "    ap.add_argument('--oov', choices=('drop', 'other'), default='drop',\n"
     "                    help='CHOSEN: drop -- exclude positions whose gold root is outside the top-K '\n"
     "                         '(the only design that gives a real cardinality reduction); '\n"
     "                         'other: REFUSED (assert in the model-build block).')\n"
     "    ap.add_argument('--tag', default='full')\n"),

    # ---- 9. persist the shrink metadata next to the history ----------------------------
    ("    json.dump({'history': hist, 'args': vars(args)}, open(args.out, 'w'), indent=2)\n",
     "    json.dump({'history': hist, 'args': vars(args),\n"
     "               'shrunk': {'root_classes': args.root_classes, 'root_map': args.root_map,\n"
     "                          'design': args.oov, 'marginal_shrunk_eval': marg_shrunk,\n"
     "                          'n_eval_kept': int(keep_all.sum()),\n"
     "                          'n_eval_total': int(keep_all.numel()),\n"
     "                          'forbidden_classes_active': int(fmask_cls.sum())\n"
     "                          if root_map is not None else None}},\n"
     "              open(args.out, 'w'), indent=2)\n"),
]


def main():
    src = SRC.read_text()
    out = src
    for i, (old, new) in enumerate(PATCHES, 1):
        n = out.count(old)
        if n != 1:
            raise SystemExit(f'PATCH {i}: anchor matched {n} times (need exactly 1):\n{old[:200]}')
        out = out.replace(old, new)
    DST.write_text(out)
    d = difflib.unified_diff(src.splitlines(keepends=True), out.splitlines(keepends=True),
                             fromfile=str(SRC), tofile=str(DST), n=3)
    DIFF.write_text(''.join(d))
    print(f'wrote {DST} ({len(out.splitlines())} lines, {len(PATCHES)} patches applied)')
    print(f'wrote {DIFF}')


if __name__ == '__main__':
    main()
