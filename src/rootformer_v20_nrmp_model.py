#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rootformer_v20_nrmp_model.py -- architecture fixes for NRMP.

The pristine `rootformer_v18_nrmp_model.py` is left untouched; this subclasses it so the two can
be A/B tested against the same checkpoint.

Fix A -- EXPLICIT ROOT HISTORY (the dominant defect)
    The shipped head computes root_logits = W h_t, so the root stream is only implicit in h_t.
    A count-based 4-gram over that stream scores ~33% acc@1 against the head's ~7.5%.
    Here the head reads  h'_t = h_t + W_h [E(r_{t-1}); E(r_{t-2}); E(r_{t-3})].
    `W_h` is ZERO-INITIALISED, so at load time this model is numerically identical to the
    shipped one -- it warm-starts from the v20 checkpoint and then learns the history term.
    `hist_proj.weight` norms are reported so you can confirm the term wakes up.

Fix B -- ENGRAM LOOKED UP BY SEQUENCE POSITION INSTEAD OF ROOT
    `RootformerV18_NRMP.__init__` sets self.backbone = base.model.backbone.model, bypassing
    `UnifiedRootformerV17_DeepSeekFlash.forward()`, which is the only place that sets
    `current_root_ids` / `current_wazn_ids`. The layer-1 and layer-14 hooks therefore receive
    None and DeepSeekEngramModule falls back to torch.arange(seq_len). We keep a reference to the
    flash wrapper and set those fields from the morphemic streams before every backbone call, so
    the radical-root memory is actually addressed by root.

Fix C -- TRAIN/INFERENCE CONDITIONING MISMATCH
    FarahidianNRMPHead conditions on `target_roots` while self.training else on argmax. With root
    acc@1 ~7% the affix heads are trained on gold roots and deployed on mostly-wrong ones.
    `forward` takes `ss_prob` (scheduled sampling): with that probability the conditioning root is
    the model's own prediction even in training mode.

Fix D -- AL-TAQALIB (Al-Khalil S^3 PERMUTATION ORBITS)
    Permutation-equivalent roots share one semantic core, but they are independent one-hot ids
    with zero parameter sharing. `orbit_consistency_loss()` pulls the embeddings of the six
    permutations of a root together (a regulariser, so no vocabulary change is needed).
"""
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

from rootformer_v18_nrmp_model import RootformerV18_NRMP, FarahidianNRMPHead  # noqa: E402


class RootformerV20_NRMP(RootformerV18_NRMP):
    def __init__(self, base_v17_model, vocab, device='cuda', dtype=torch.bfloat16,
                 hist=3, d_hist=64):
        super().__init__(base_v17_model, vocab, device, dtype)

        # ---- Fix B: keep the flash wrapper so its hooks can be fed root ids --------------
        # NB: assigning an nn.Module to self registers it as a submodule, which would
        # duplicate every flash parameter in this model's state_dict (581 spurious
        # 'missing' keys) and hand them gradients. Store it unregistered instead.
        object.__setattr__(self, 'flash', base_v17_model)

        # ---- Fix A: root-history augmentation, zero-initialised (warm start) --------------
        self.hist_n = hist
        self.hist_emb = nn.ModuleList(
            [nn.Embedding(vocab.num_roots, d_hist) for _ in range(hist)]
        ).to(device=device, dtype=dtype)
        self.hist_proj = nn.Linear(hist * d_hist, self.d_model, bias=False).to(
            device=device, dtype=dtype)
        nn.init.normal_(self.hist_proj.weight, std=0.0)      # exactly a no-op at load time

        # ---- Fix D: precompute Al-Taqalib permutation orbits ----------------------------
        self.orbit_pairs = self._build_orbit_pairs(vocab)

    # ------------------------------------------------------------------ Fix D
    @staticmethod
    def _build_orbit_pairs(vocab) -> List[Tuple[int, int]]:
        pairs, seen = [], set()
        for r in vocab.roots_list:
            if r.startswith('<') or len(r) != 3:
                continue
            perms = {''.join(p) for p in itertools.permutations(r)}
            ids = [vocab.root2id[p] for p in perms if p in vocab.root2id]
            ids = sorted(set(ids))
            for a, b in itertools.combinations(ids, 2):
                if (a, b) not in seen:
                    seen.add((a, b))
                    pairs.append((a, b))
        return pairs

    def orbit_consistency_loss(self, max_pairs=4096) -> torch.Tensor:
        """Pull permutation-equivalent root embeddings together (Al-Taqiq -> Al-Taqalib)."""
        if not self.orbit_pairs:
            return torch.zeros((), device=self.hist_proj.weight.device)
        idx = torch.randint(0, len(self.orbit_pairs), (min(max_pairs, len(self.orbit_pairs)),))
        a = torch.tensor([self.orbit_pairs[i][0] for i in idx], device=self.device)
        b = torch.tensor([self.orbit_pairs[i][1] for i in idx], device=self.device)
        ea = F.normalize(self.morphemic_embed.root_embed(a).float(), dim=-1)
        eb = F.normalize(self.morphemic_embed.root_embed(b).float(), dim=-1)
        return (1.0 - (ea * eb).sum(-1)).mean()

    # ------------------------------------------------------------------ Fix A
    def _augment(self, h: torch.Tensor, root_ids: torch.Tensor) -> torch.Tensor:
        """h: [B,T,d] ; root_ids: [B,T] -> h + W_h[E(r_{t-1});E(r_{t-2});E(r_{t-3})]."""
        if self.hist_n == 0:
            return h
        embs = []
        for k in range(1, self.hist_n + 1):
            shifted = torch.roll(root_ids, shifts=k, dims=1)
            if k > 0:
                shifted[:, :k] = 0          # pad: no history available
            embs.append(self.hist_emb[k - 1](shifted))
        return h + self.hist_proj(torch.cat(embs, dim=-1)).to(h.dtype)

    def hist_wake_up(self) -> float:
        return float(self.hist_proj.weight.detach().float().norm().item())

    # ------------------------------------------------------------------ Fix B helper
    def _set_flash_ids(self, r_ids: torch.Tensor, w_ids: torch.Tensor):
        if self.flash is not None and hasattr(self.flash, 'current_root_ids'):
            self.flash.current_root_ids = r_ids
            self.flash.current_wazn_ids = w_ids

    # ------------------------------------------------------------------ forward
    def forward(self, p_ids, r_ids, w_ids, s_ids,
                target_roots=None, target_awzan=None,
                target_prefixes=None, target_suffixes=None,
                ss_prob: float = 0.0, lambda_orbit: float = 0.0) -> Dict[str, Any]:
        inputs_embeds = self.morphemic_embed(p_ids, r_ids, w_ids, s_ids)

        # Fix B: address the engram memory by ROOT, not by sequence position
        self._set_flash_ids(r_ids, w_ids)

        outputs = self.backbone(inputs_embeds=inputs_embeds)
        hidden = self.final_norm(outputs.last_hidden_state)

        # Fix A: augment the hidden state with the root history (no-op until trained)
        hidden = self._augment(hidden, r_ids)

        # Fix C: scheduled sampling for the morphological conditioning
        cond_roots = target_roots
        if target_roots is not None and ss_prob > 0 and self.training:
            raw = self.nrmp_head.root_head(hidden.detach())
            pred = raw.argmax(-1)
            use_pred = torch.rand_like(pred.float()) < ss_prob
            cond_roots = torch.where(use_pred, pred, target_roots)

        heads = self.nrmp_head(h=hidden, root_embed_layer=self.morphemic_embed.root_embed,
                               target_roots=cond_roots)

        loss, loss_dict = None, {}
        if target_roots is not None and target_awzan is not None:
            def sh(x):
                return x[:, :-1, :].contiguous()

            def tg(x):
                return x[:, 1:].contiguous()

            tr_, tw_ = tg(target_roots), tg(target_awzan)
            tp_ = tg(target_prefixes if target_prefixes is not None else p_ids)
            ts_ = tg(target_suffixes if target_suffixes is not None else s_ids)

            ignore = torch.tensor([self.vocab.PAD_ROOT, self.vocab.BOS_ROOT, self.vocab.EOS_ROOT,
                                   self.vocab.UNK_ROOT, self.vocab.root2id['<PARTICLE>']],
                                  device=p_ids.device)
            if self.sibawayh_gov is not None:
                ignore = torch.cat([ignore, torch.tensor(
                    sorted(self.sibawayh_gov.particle_root_ids), device=p_ids.device)])
            keep = ~torch.isin(tr_, ignore)
            l_root = (F.cross_entropy(sh(heads['root_logits'])[keep], tr_[keep])
                      if keep.any() else sh(heads['root_logits']).sum() * 0.0)
            l_wazn = F.cross_entropy(sh(heads['wazn_logits']).view(-1, self.vocab.num_awzan),
                                     tw_.view(-1), ignore_index=self.vocab.PAD_WAZN)
            l_pref = F.cross_entropy(sh(heads['prefix_logits']).view(-1, self.vocab.num_prefixes),
                                     tp_.view(-1), ignore_index=self.vocab.PAD_PREFIX)
            l_suff = F.cross_entropy(sh(heads['suffix_logits']).view(-1, self.vocab.num_suffixes),
                                     ts_.view(-1), ignore_index=self.vocab.PAD_SUFFIX)
            import math
            loss = l_root + 0.5 * l_wazn + 0.25 * l_pref + 0.25 * l_suff
            loss_dict = {'total_loss': float(loss.detach()), 'loss_root': float(l_root.detach()),
                         'loss_wazn': float(l_wazn.detach()), 'loss_prefix': float(l_pref.detach()),
                         'loss_suffix': float(l_suff.detach()),
                         'root_ppl': math.exp(min(float(l_root.detach()), 20.0))}
            if lambda_orbit > 0:
                l_orb = self.orbit_consistency_loss()
                loss = loss + lambda_orbit * l_orb
                loss_dict['loss_orbit'] = float(l_orb.detach())

        return {'loss': loss, 'loss_dict': loss_dict, 'hidden_states': hidden, **heads}

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate_words(self, prompt_text: str, max_new_words: int = 20, min_new_words: int = 5):
        self.eval()
        # a new prompt starts a new ʿāmil chain (inqiṭāʿ al-ʿamal, al-Kitab 1/421)
        if self.sibawayh_gov:
            self.sibawayh_gov.reset_chain()
        encoded = self.vocab.encode_sentence(prompt_text)
        if not encoded:
            encoded = [(self.vocab.NONE_PREFIX, self.vocab.BOS_ROOT,
                        self.vocab.NONE_WAZN, self.vocab.NONE_SUFFIX)]
        cp = [t[0] for t in encoded]
        cr = [t[1] for t in encoded]
        cw = [t[2] for t in encoded]
        cs = [t[3] for t in encoded]
        generated = []
        for step in range(max_new_words):
            p = torch.tensor([cp], dtype=torch.long, device=self.device)
            r = torch.tensor([cr], dtype=torch.long, device=self.device)
            w = torch.tensor([cw], dtype=torch.long, device=self.device)
            s = torch.tensor([cs], dtype=torch.long, device=self.device)
            emb = self.morphemic_embed(p, r, w, s)
            self._set_flash_ids(r, w)                      # Fix B during generation too
            out = self.backbone(inputs_embeds=emb)
            h = self._augment(self.final_norm(out.last_hidden_state), r)[:, -1:, :]  # Fix A

            r_logits = self.nrmp_head.root_head(h)[0, 0, :].clone()
            if self.sibawayh_gov:
                r_logits = self.sibawayh_gov.apply_root_exclusion_mask(
                    r_logits, self.sibawayh_gov.get_operator_state((cp[-1], cr[-1], cw[-1], cs[-1])),
                    cr[-6:], step, min_new_words)
            else:
                for bad in (self.vocab.PAD_ROOT, self.vocab.BOS_ROOT, self.vocab.UNK_ROOT,
                            self.vocab.root2id['<PARTICLE>']):
                    r_logits[bad] = -1e9
                if step < min_new_words:
                    r_logits[self.vocab.EOS_ROOT] = -1e9
                for rid in set(cr[-6:]):
                    r_logits[rid] -= cr[-6:].count(rid) * 2.5
            next_r = int(torch.argmax(r_logits).item())
            if next_r == self.vocab.EOS_ROOT:
                break
            root_str = self.vocab.id2root.get(next_r, '')
            is_particle = root_str.startswith('<P:') or root_str in ('<PARTICLE>', '<UNK>', '<PAD>')

            r_t = torch.tensor([[next_r]], dtype=torch.long, device=self.device)
            e_root = self.morphemic_embed.root_embed(r_t)
            h_cond = self.nrmp_head.cond_proj(torch.cat([h, e_root], dim=-1))
            if is_particle:
                next_w = self.vocab.NONE_WAZN
            else:
                wl = self.nrmp_head.wazn_head(h_cond)[0, 0, :].clone()
                if self.sibawayh_gov:
                    wl = self.sibawayh_gov.apply_wazn_exclusion_mask(
                        wl, 'NONE', self.vocab.NONE_PREFIX, is_radical_root=True)
                else:
                    for bad in (self.vocab.NONE_WAZN, self.vocab.PAD_WAZN,
                                self.vocab.wazn2id['<UNK>']):
                        wl[bad] = -1e9
                next_w = int(torch.argmax(wl).item())
            pl = self.nrmp_head.prefix_head(h_cond)[0, 0, :].clone()
            pl[self.vocab.PAD_PREFIX] = -1e9
            next_p = int(torch.argmax(pl).item())
            sl = self.nrmp_head.suffix_head(h_cond)[0, 0, :].clone()
            sl[self.vocab.PAD_SUFFIX] = -1e9
            next_s = int(torch.argmax(sl).item())

            cp.append(next_p); cr.append(next_r); cw.append(next_w); cs.append(next_s)
            generated.append((next_p, next_r, next_w, next_s))
        return {'generated_text': self.vocab.decode_sentence(generated, apply_syntax=True),
                'generated_tuples': generated, 'num_words': len(generated)}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='/workspace/nrmp_trained_final.safetensors')
    args = ap.parse_args()
    print('[*] loading v20 architecture...')
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    import nrmp_vocab as nv
    cls = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = cls(str(ROOT_DIR / 'data/rootformer_v12_arabic_blueprint.json'))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = UnifiedRootformerV12(str(ROOT_DIR / 'data/rootformer_v12_arabic_blueprint.json'),
                               'Qwen/Qwen2.5-0.5B', str(dev), torch.bfloat16).to(dev)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, dev, torch.bfloat16).to(dev)
    m = RootformerV20_NRMP(flash, vocab, dev, torch.bfloat16).to(dev)
    from safetensors.torch import load_file
    missing, unexpected = m.load_state_dict(load_file(args.checkpoint), strict=False)
    print(f'[*] loaded: missing={len(missing)} unexpected={len(unexpected)}')
    print(f'[*] orbit pairs (Al-Taqalib): {len(m.orbit_pairs)}')
    print(f'[*] hist_proj norm at init (must be 0.0 to preserve v20 behaviour): '
          f'{m.hist_wake_up():.6f}')
    g = m.generate_words('العلم نور يضيء العقل')
    print('[*] sample generation:', g['generated_text'][:110])
