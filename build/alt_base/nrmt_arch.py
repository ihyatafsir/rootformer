#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmt_arch.py -- an architecture fitted for Next-Root-Morph-Token prediction.

Why a new head
--------------
The shipped head is `root_logits = W · h_t`: a single linear map from the final hidden state.
Root prediction is then a *byproduct* of the backbone. But the measurements say next-root is
mostly a function of the recent ROOT stream (a 4-gram over roots ~33% acc@1 vs the head's ~7.5%),
and Sībawayh's operator (al-ʿāmil) restricts what may follow. So NRMT must be a first-class,
autoregressive, operator-conditioned object -- not a readout.

What this architecture makes explicit
-------------------------------------
1. ROOT HISTORY      h'_t = h_t + W_h [E(r_{t-1}); E(r_{t-2}); E(r_{t-3})]      (zero-init)
2. OPERATOR STATE    + W_o E(op_t),  op_t in Sibawayh's states
                     (HARF_JARR, HARF_JAZM, HARF_NASB, INNA, KANA, FUTURE)     (zero-init)
3. PREV MORPH TUPLES + W_m [E(w_{t-1}); E(p_{t-1}); E(s_{t-1})]                 (zero-init)
   All three projections are zero-initialised, so loading an existing checkpoint reproduces the
   old behaviour exactly and the new capacity is then learned.
4. FACTORISED DECODE root -> cond_proj([h'; E(root)]) -> wazn / prefix / suffix
5. NEGATIVE CONSTRAINTS  impossibility hinge loss over forbidden roots per position,
   so the head is taught what CANNOT come next, not merely what does.
6. AL-TAQALIB            orbit_consistency_loss() over al-Khalil's S^3 permutation families.
   ENGINEERING MEASUREMENT: the pair list is built by `khalil_orbits.KhalilPermutationOrbits`,
   which keeps only the permutations `data/khalil_attest_v4.json` records as مستعمل and drops
   the مهمل ones -- 4,305 pairs over 967 orbits on the shipped inventory (it was 7,727 pairs
   when the list was "every permutation that happens to be in the root inventory").
   FIDELITY CAVEAT: this is orbit/parameter consistency, NOT al-ishtiqāq al-akbar.  The
   tradition rejects the semantic claim that an orbit shares one MEANING (Ibn ʿUsfūr,
   al-Mumtiʿ fī al-Taṣrīf: «والصحيح أن هذا النحو من الاشتقاق غير مأخوذ به؛ لعدم اطراده»);
   nothing here asserts or optimises shared meaning.  See `khalil_orbits.py` for the sources.
7. TRANSLATION HANDOFF   `root_states` (h'_t) is exposed as the interface a bilingual decoder
   should cross-attend to -- the root of the alignment thesis.

Everything is warm-startable from `rootformer_v20_nrmp_master.safetensors`.
"""
import itertools
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

OP_STATES = ['NONE', 'HARF_JARR', 'HARF_JAZM', 'HARF_NASB', 'INNA', 'KANA', 'FUTURE']
OP2ID = {s: i for i, s in enumerate(OP_STATES)}


def build_operator_table(vocab, gov=None) -> torch.Tensor:
    """
    Map every root id to the Sibawayh operator state it establishes (0 = NONE).

    Operator state is a function of the PREVIOUS word, so at training/inference time
    `op_ids = table[r_ids].roll(1)`. Doing this as a table keeps it vectorised over millions of
    positions instead of calling the Python governance engine per token.
    """
    table = torch.zeros(vocab.num_roots, dtype=torch.long)
    surface_to_op = {}
    for w in ('في', 'من', 'إلى', 'على', 'عن', 'مع', 'حتى', 'منذ', 'مذ', 'رب', 'ب', 'ل', 'ك'):
        surface_to_op[w] = 'HARF_JARR'
    for w in ('لم', 'لما'):
        surface_to_op[w] = 'HARF_JAZM'
    for w in ('لن', 'كي', 'إذن'):
        surface_to_op[w] = 'HARF_NASB'
    for w in ('إن', 'أن', 'كأن', 'لكن', 'ليت', 'لعل'):
        surface_to_op[w] = 'INNA'
    for w in ('كان', 'أصبح', 'أمسى', 'أضحى', 'ظل', 'بات', 'صار', 'ليس'):
        surface_to_op[w] = 'KANA'
    for w in ('سوف', 'س'):
        surface_to_op[w] = 'FUTURE'
    for i, r in enumerate(vocab.roots_list):
        name = r[3:-1] if r.startswith('<P:') else r
        op = surface_to_op.get(name)
        if op:
            table[i] = OP2ID[op]
    return table


class NRMTHead(nn.Module):
    """Operator- and history-conditioned, factorised next-root-morph head."""

    def __init__(self, d_model: int, num_roots: int, num_awzan: int, num_prefixes: int,
                 num_suffixes: int, d_root: int, hist: int = 3, d_hist: int = 64,
                 n_ops: int = len(OP_STATES), d_op: int = 32, d_morph: int = 32,
                 dropout: float = 0.1, use_features: bool = True,
                 feat_gate: bool = False):
        super().__init__()
        self.d_model, self.d_root, self.hist = d_model, d_root, hist
        self.num_roots = num_roots
        self.use_features = use_features
        # MEASURED (v18fix, see `feat_gate` below): `feat_norm` is a LayerNorm, so the branch
        # output is EXACTLY invariant to the scale of `feat_proj.weight` once that weight is
        # more than ~1e-2 in norm.  The zero-init of feat_proj therefore buys NO warm start:
        # the branch output jumps discontinuously from 0 to its full unit-RMS magnitude in one
        # Adam step (measured: step-1 loss 11.698 -> step-2 loss 90.533; prefix CE 2.006 ->
        # 41.751; wazn CE 6.924 -> 45.836; max grad_norm 12,137).  `feat_gate=True` multiplies
        # the branch by a zero-initialised scalar, which restores a real fade-in.
        self.feat_gate_enabled = feat_gate
        if feat_gate:
            self.feat_gate = nn.Parameter(torch.zeros(()))

        self.hist_emb = nn.ModuleList([nn.Embedding(num_roots, d_hist) for _ in range(hist)])
        self.op_emb = nn.Embedding(n_ops, d_op)
        self.prev_wazn_emb = nn.Embedding(num_awzan, d_morph)
        self.prev_pref_emb = nn.Embedding(num_prefixes, d_morph)
        self.prev_suff_emb = nn.Embedding(num_suffixes, d_morph)
        d_extra = hist * d_hist + d_op + 3 * d_morph
        self.feat_proj = nn.Linear(d_extra, d_model, bias=False)
        # keeps the conditioning from growing unboundedly (a raw additive term blew PPL up
        # to 3e4 on held-out data: the feature norm had already reached 41 by step 2000)
        self.feat_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self.root_head = nn.Linear(d_model, num_roots, bias=False)
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model + d_root, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.wazn_head = nn.Linear(d_model, num_awzan, bias=False)
        self.prefix_head = nn.Linear(d_model, num_prefixes, bias=False)
        self.suffix_head = nn.Linear(d_model, num_suffixes, bias=False)

        # zero-init => the conditioning is a no-op until trained
        nn.init.zeros_(self.feat_proj.weight)

    # ------------------------------------------------------------------ features
    def build_features(self, root_ids, op_ids, w_ids, p_ids, s_ids) -> torch.Tensor:
        """Roll the history and the PREVIOUS morph tuple internally, so callers cannot
        accidentally pass the current position's tuple (a bug that silently fed the head the
        wrong word's morphology). With use_features=False this is the h-only CONTROL."""
        if not self.use_features:
            return torch.zeros(*root_ids.shape, self.d_model,
                               device=root_ids.device, dtype=self.feat_proj.weight.dtype)
        parts = []
        for k in range(1, self.hist + 1):
            sh = torch.roll(root_ids, shifts=k, dims=1)
            sh[:, :k] = 0
            parts.append(self.hist_emb[k - 1](sh))
        pw = torch.roll(w_ids, 1, dims=1); pw[:, 0] = 0
        pp = torch.roll(p_ids, 1, dims=1); pp[:, 0] = 0
        ps = torch.roll(s_ids, 1, dims=1); ps[:, 0] = 0
        parts += [self.op_emb(op_ids), self.prev_wazn_emb(pw),
                  self.prev_pref_emb(pp), self.prev_suff_emb(ps)]
        out = self.drop(self.feat_norm(self.feat_proj(torch.cat(parts, dim=-1))))
        if self.feat_gate_enabled:
            out = out * self.feat_gate
        return out

    def forward(self, h, root_ids, op_ids, w_ids, p_ids, s_ids,
                cond_roots: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h_aug = h + self.build_features(root_ids, op_ids, w_ids, p_ids, s_ids).to(h.dtype)
        root_logits = self.root_head(h_aug)
        chosen = cond_roots if cond_roots is not None else root_logits.argmax(-1)
        e_root = self._root_embed(chosen)
        h_cond = self.cond_proj(torch.cat([h_aug, e_root.to(h_aug.dtype)], dim=-1))
        return {'root_logits': root_logits, 'wazn_logits': self.wazn_head(h_cond),
                'prefix_logits': self.prefix_head(h_cond),
                'suffix_logits': self.suffix_head(h_cond), 'root_states': h_aug}

    def extra_norm(self) -> float:
        """||feat_proj.weight|| -- the historical "extra conditioning norm" diagnostic.

        READ THIS BEFORE USING IT AS A HEALTH METRIC.  `feat_norm` (a LayerNorm) sits directly
        downstream of `feat_proj`, and LayerNorm is invariant to any positive rescaling of its
        input.  The branch output -- and therefore every logit -- is therefore EXACTLY unchanged
        when this norm is multiplied by 1e3 or 1e8.  MEASURED on the shipped head: sweeping
        ||feat_proj.weight|| over 0.5 -> 5.4e8 leaves max|dlogit| constant at 6.55938.
        This quantity is a pure GAUGE that AdamW random-walks upward; it is a proxy for "how hard
        the head has been driven", NOT a measure of the conditioning branch's influence.
        `feat_scale()` below is the quantity that actually controls the branch magnitude.
        """
        return float(self.feat_proj.weight.detach().float().norm().item())

    def feat_scale(self) -> float:
        """||feat_norm.weight|| (the LayerNorm gain) -- the REAL branch-magnitude control.

        LayerNorm output y = gamma * x_hat + beta with x_hat unit-variance per position, so the
        branch's per-position RMS is ||gamma||/sqrt(d_model) (plus the small beta term).  This is
        the number that can actually inflate h_aug and hence the logits; ||feat_proj.weight||
        cannot.  MEASURED on the shipped head at step 20,000: ||gamma|| = 24.14 (init 29.93),
        i.e. the branch RMS went 1.000 -> 0.807 -- it never inflated.
        """
        return float(self.feat_norm.weight.detach().float().norm().item())


class RootformerNRMT(nn.Module):
    """
    Wraps the v19.2 backbone (morphemic embedding + 24-layer IshtiqaqAttention backbone) with the
    NRMT head. Keeps `root_states` as the public interface for a downstream bilingual decoder.
    """

    def __init__(self, base_v17_model, vocab, device='cuda', dtype=torch.bfloat16,
                 hist=3, d_hist=64, d_op=32, d_morph=32, dropout=0.1, use_features=True,
                 feat_gate=False):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.vocab = vocab

        # backbone (same unwrapping as the shipped model)
        self.backbone = base_v17_model.model.backbone.model
        self.d_model = getattr(self.backbone.config, 'hidden_size', 896)

        # keep the flash wrapper WITHOUT registering it as a submodule
        object.__setattr__(self, '_flash', base_v17_model)

        # reuse the released morphemic embedding so checkpoints transfer
        self.morphemic_embed = base_v17_model.model.morphemic_embed if hasattr(
            base_v17_model.model, 'morphemic_embed') else None
        if self.morphemic_embed is None:
            from rootformer_v18_nrmp_model import FarahidianWordEmbedding
            self.morphemic_embed = FarahidianWordEmbedding(
                vocab.num_roots, vocab.num_awzan, vocab.num_prefixes, vocab.num_suffixes,
                self.d_model).to(device=device, dtype=dtype)
        self.final_norm = nn.LayerNorm(self.d_model).to(device=device, dtype=dtype)

        self.nrmt_head = NRMTHead(self.d_model, vocab.num_roots, vocab.num_awzan,
                                 vocab.num_prefixes, vocab.num_suffixes,
                                 self.morphemic_embed.d_root, hist=hist, d_hist=d_hist,
                                 d_op=d_op, d_morph=d_morph, dropout=dropout,
                                 use_features=use_features, feat_gate=feat_gate
                                 ).to(device=device, dtype=dtype)
        # bind the SHARED root embedding without registering it a second time
        object.__setattr__(self.nrmt_head, '_root_embed', self.morphemic_embed.root_embed)

        self.sibawayh_gov = getattr(base_v17_model.model, 'sibawayh_gov', None) \
            if hasattr(base_v17_model, 'model') else None

        self.orbit_pairs = self._build_orbit_pairs(vocab)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _build_orbit_pairs(vocab) -> List[Tuple[int, int]]:
        """al-Taqālīb pairs from al-Khalīl's orbit record: ATTESTED permutations only.

        al-Khalīl's own filter (Kitāb al-ʿAyn: «يُكَتَب مُسْتَعْمَلها. ويُلغى مُهْمَلها») keeps the
        used permutations and cancels the unused ones, so a permutation al-ʿAyn marks مهمل is
        NOT paired with its orbit-mates.  The record is `data/khalil_attest_v4.json`; membership
        is `membership='attested'` (positive attestation), which reproduces al-ʿAyn's own chapter
        counts -- e.g. the ع ل م chapter («... ع ل م، ع م ل، م ع ل، ل م ع مستعملات») gives علم a
        4-face orbit, not 6.

        ENGINEERING: falls back to the legacy "every permutation present in the inventory"
        builder if `khalil_orbits` (or its record) is unavailable, so the head still trains.
        NOT al-ishtiqāq al-akbar: no shared meaning is asserted.  See `khalil_orbits.py`.
        """
        try:
            from khalil_orbits import KhalilPermutationOrbits
            return KhalilPermutationOrbits(vocab).pairs()
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f'[nrmt_arch] orbit pairs: falling back to inventory-only permutations '
                  f'({type(exc).__name__}: {exc})')
        pairs, seen = [], set()
        for r in vocab.roots_list:
            if r.startswith('<') or len(r) != 3:
                continue
            ids = sorted({vocab.root2id[p] for p in
                          (''.join(x) for x in itertools.permutations(r)) if p in vocab.root2id})
            for a, b in itertools.combinations(ids, 2):
                if (a, b) not in seen:
                    seen.add((a, b)); pairs.append((a, b))
        return pairs

    def apply_orbit_tying(self, method: str = 'mean') -> Dict[str, Any]:
        """Make al-Khalīl orbit-mates share ONE root-embedding row (parameter tying).

        CALL THIS AFTER `load_state_dict`.  It is shape-preserving (rows are overwritten in
        place, `num_roots x d_root` unchanged), so it invalidates no existing checkpoint; but
        loading a checkpoint afterwards would overwrite the tying, hence the ordering.

        It is deliberately NOT applied automatically: it is a real change to the parameter
        geometry, and the mechanism it implements is al-Khalīl's مستعمل/مهمل filter, not
        al-ishtiqāq al-akbar (whose semantic claim the tradition rejects; see khalil_orbits.py).
        Returns the `tie_embeddings` stats dict, or {} if orbit tying is unavailable.
        """
        try:
            from khalil_orbits import KhalilPermutationOrbits
            orbits = KhalilPermutationOrbits(self.vocab)
            stats = orbits.tie_embeddings(self.morphemic_embed.root_embed.weight, method=method)
            print(f"[nrmt_arch] orbit tying: {stats['orbits_tied']} orbits, "
                  f"{stats['rows_collapsed']} rows collapsed (method={method})")
            return stats
        except Exception as exc:  # pragma: no cover - defensive
            print(f'[nrmt_arch] orbit tying skipped: {type(exc).__name__}: {exc}')
            return {}

    def orbit_consistency_loss(self, max_pairs=4096) -> torch.Tensor:
        if not self.orbit_pairs:
            return torch.zeros((), device=self.device)
        n = min(max_pairs, len(self.orbit_pairs))
        idx = torch.randint(0, len(self.orbit_pairs), (n,), device=self.device)
        a = torch.tensor([self.orbit_pairs[i][0] for i in idx.tolist()], device=self.device)
        b = torch.tensor([self.orbit_pairs[i][1] for i in idx.tolist()], device=self.device)
        ea = F.normalize(self.morphemic_embed.root_embed(a).float(), dim=-1)
        eb = F.normalize(self.morphemic_embed.root_embed(b).float(), dim=-1)
        return (1.0 - (ea * eb).sum(-1)).mean()

    def _set_flash(self, r_ids, w_ids):
        fl = getattr(self, '_flash', None)
        if fl is not None and hasattr(fl, 'current_root_ids'):
            fl.current_root_ids = r_ids
            fl.current_wazn_ids = w_ids

    # ------------------------------------------------------------------ forward
    def forward(self, p_ids, r_ids, w_ids, s_ids,
                op_ids=None, target_roots=None, target_awzan=None,
                target_prefixes=None, target_suffixes=None,
                ss_prob=0.0, lambda_orbit=0.0, forbidden_mask=None,
                margin=0.0) -> Dict[str, Any]:
        emb = self.morphemic_embed(p_ids, r_ids, w_ids, s_ids)
        self._set_flash(r_ids, w_ids)
        out = self.backbone(inputs_embeds=emb)
        h = self.final_norm(out.last_hidden_state)

        if op_ids is None:
            op_ids = torch.zeros_like(r_ids)
        cond = target_roots
        if self.training and ss_prob > 0:
            with torch.no_grad():
                pred = self.nrmt_head.root_head(
                    h + self.nrmt_head.build_features(
                        r_ids, op_ids, w_ids, p_ids, s_ids).to(h.dtype)
                ).argmax(-1)
            use_pred = torch.rand_like(pred.float()) < ss_prob
            cond = pred if target_roots is None else torch.where(use_pred, pred, target_roots)

        heads = self.nrmt_head(h, r_ids, op_ids, w_ids, p_ids, s_ids, cond_roots=cond)

        loss, info = None, {}
        if target_roots is not None and target_awzan is not None:
            def sh(x): return x[:, :-1, :].contiguous()
            def tg(x): return x[:, 1:].contiguous()

            tr_, tw_ = tg(target_roots), tg(target_awzan)
            tp_ = tg(target_prefixes if target_prefixes is not None else p_ids)
            ts_ = tg(target_suffixes if target_suffixes is not None else s_ids)

            ignore = [self.vocab.PAD_ROOT, self.vocab.BOS_ROOT, self.vocab.EOS_ROOT,
                      self.vocab.UNK_ROOT, self.vocab.root2id['<PARTICLE>']]
            keep = ~torch.isin(tr_, torch.tensor(ignore, device=tr_.device))
            rl = sh(heads['root_logits'])
            l_root = F.cross_entropy(rl[keep], tr_[keep]) if keep.any() else rl.sum() * 0.0
            l_wazn = F.cross_entropy(sh(heads['wazn_logits']).view(-1, self.vocab.num_awzan),
                                     tw_.view(-1), ignore_index=self.vocab.PAD_WAZN)
            l_pref = F.cross_entropy(sh(heads['prefix_logits']).view(-1, self.vocab.num_prefixes),
                                     tp_.view(-1), ignore_index=self.vocab.PAD_PREFIX)
            l_suff = F.cross_entropy(sh(heads['suffix_logits']).view(-1, self.vocab.num_suffixes),
                                     ts_.view(-1), ignore_index=self.vocab.PAD_SUFFIX)
            loss = l_root + 0.5 * l_wazn + 0.25 * l_pref + 0.25 * l_suff
            info = {'loss_root': float(l_root.detach()), 'loss_wazn': float(l_wazn.detach()),
                    'loss_prefix': float(l_pref.detach()), 'loss_suffix': float(l_suff.detach()),
                    'root_ppl': math.exp(min(float(l_root.detach()), 20.0))}

            # (5) negative impossibility hinge: push forbidden roots below `margin`
            if forbidden_mask is not None and margin > 0:
                rl2 = sh(heads['root_logits'])
                fm = forbidden_mask[:, 1:].bool()
                if fm.any():
                    pen = F.relu(rl2[fm] - margin) ** 2
                    loss = loss + 0.1 * pen.mean()
                    info['loss_impossible'] = float(pen.mean().detach())

            if lambda_orbit > 0:
                l_orb = self.orbit_consistency_loss()
                loss = loss + lambda_orbit * l_orb
                info['loss_orbit'] = float(l_orb.detach())

        return {'loss': loss, 'loss_dict': info, 'root_states': heads['root_states'], **heads}
