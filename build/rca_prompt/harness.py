#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harness.py -- qualitative prompt-read of the RCA arm vs its own gate-ablation.

Loads /workspace/root_attn/head_RCA_UNFREEZE_A2.pt (+ .trunk.pt) on top of the shipped
v19.2 checkpoint -- the same load path as repro_eval.py -- then:

  A. aggregate self-test: recompute ALL_val / NOVEL_only acc@1/acc@5/CE_z for BOTH
     conditions over the trainer's own 300 val windows, plus the majority-class marginal.
  B. rebuild the ALIGNED val stream from the corpus (file-level split, seed 1337) with the
     surface word of every word event, and ASSERT element-wise equality with the shipped
     cache val.pt -- so the surfaces provably line up with the positions the model was
     scored on.
  C. prompt N windows sampled uniformly at random (default seed 20261002) from the same
     300 val starts; for each: context text, RCA-on top-1/top-5, RCA-off top-1/top-5,
     truth -- all decoded with the project's own realiser (nrmp_vocab.decode_word).
  D. collapse / leak / frequency-rank metrics over all 300 windows for both conditions.

Read-only w.r.t. every existing module and checkpoint.  Writes only under --out-dir.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REL = '/workspace/hf_v19_2_release'
WIN, STRIDE, CTX = 128, 64, 12
TGT_LOCAL = 64          # head position at which we read the prompt prediction
                        # -> the target word is window position TGT_LOCAL+1 = 65


def now():
    return time.strftime('%H:%M:%S')


# ---------------------------------------------------------------------------- aggregate fit
def majority_marginal(true_roots, keep):
    v = true_roots[keep]
    if v.size == 0:
        return None
    c = np.bincount(v.astype(np.int64))
    return float(c.max()) / float(v.size)


def confusion_free_acc(pred, true, keep):
    if keep.sum() == 0:
        return None
    return float((pred[keep] == true[keep]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', default=REL)
    ap.add_argument('--checkpoint',
                    default='/workspace/hf_v19_2_release/checkpoints/'
                            'rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors')
    ap.add_argument('--cache', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--head', default='/workspace/root_attn/head_RCA_UNFREEZE_A2.pt')
    ap.add_argument('--trunk', default='/workspace/root_attn/head_RCA_UNFREEZE_A2.pt.trunk.pt')
    ap.add_argument('--out-dir', default='/workspace/rca_prompt')
    ap.add_argument('--n-prompts', type=int, default=30)
    ap.add_argument('--seed', type=int, default=20261002)
    ap.add_argument('--head-dtype', choices=('bf16', 'fp32'), default='fp32',
                    help='dtype of the NRMT head module (repro_eval.py uses fp32)')
    ap.add_argument('--feat-gate', choices=('auto', 'on', 'off'), default='auto',
                    help='build the head WITH its trained feat_gate scalar.  repro_eval.py '
                         'silently builds feat_gate=False, dropping the trained multiplier '
                         '(A2 saved 0.2198) -- auto reads it off the checkpoint keys')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--skip-rebuild', action='store_true',
                    help='reuse /workspace/rca_prompt/val_stream_rebuild.pt if present')
    args = ap.parse_args()

    os.environ.setdefault('ROOTFORMER_VALIDATED_SEG', '0')
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'run.log', 'a', buffering=1)

    def log(*a):
        s = ' '.join(str(x) for x in a)
        print(f'[{now()}] {s}', flush=True)
        logf.write(f'[{now()}] {s}\n')

    sys.path.insert(0, args.release)
    sys.path.insert(0, str(Path(args.release) / 'models'))
    sys.path.insert(0, '/workspace/root_attn')

    import nrmp_vocab as nv
    import nrmp_train as NP
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT, build_operator_table
    from root_cross_attn import RootCrossAttentionStack
    from safetensors.torch import load_file
    import nrmt_train as NT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'device={device}')
    bp = str(Path(args.release) / 'data/rootformer_v12_arabic_blueprint.json')
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V(bp)
    num_roots = vocab.num_roots
    log(f'vocab: roots={num_roots} awzan={vocab.num_awzan} prefixes={vocab.num_prefixes} '
        f'suffixes={vocab.num_suffixes}')

    PAD, BOS, EOS, UNK = vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT
    PARTICLE = vocab.root2id['<PARTICLE>']
    specials = sorted({PAD, BOS, EOS, UNK, PARTICLE} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec_t = torch.tensor(specials)

    def root_name(rid):
        return vocab.roots_list[int(rid)]

    def decode(p, r, w, s):
        try:
            return vocab.decode_word(int(p), int(r), int(w), int(s))
        except Exception as e:
            return f'<DECODE_ERR:{type(e).__name__}>'

    def leak_class(p, r, w, s):
        """Where does a decoded tuple stand w.r.t. control tokens / form failure?"""
        surf = decode(p, r, w, s)
        rn = vocab.roots_list[int(r)]
        if int(r) in (PAD, BOS, EOS):
            return 'control_root_(PAD/BOS/EOS)->empty', surf
        if int(r) == UNK:
            return 'UNK_root_(affixes only)', surf
        if rn == '<PARTICLE>' or int(r) == PARTICLE:
            return '<PARTICLE>_root_(affixes only)', surf
        if '<' in surf or '>' in surf:
            return 'LITERAL_CONTROL_TOKEN_IN_TEXT', surf
        if surf == '':
            return 'empty_surface', surf
        return 'well_formed', surf

    # ================================================================= load the arm
    tr4 = [t.long() for t in torch.load(f'{args.cache}/train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(f'{args.cache}/val.pt', map_location='cpu')]
    log(f'cache: train={tr4[1].numel()} val={va4[1].numel()} root tokens')

    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    hsd = torch.load(args.head, map_location='cpu')
    ck_feat_gate = float(hsd['feat_gate']) if 'feat_gate' in hsd else None
    if args.feat_gate == 'auto':
        use_feat_gate = ck_feat_gate is not None
    else:
        use_feat_gate = (args.feat_gate == 'on')
    log(f'head ckpt feat_gate={ck_feat_gate} -> building head with feat_gate={use_feat_gate}')
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16,
                           feat_gate=use_feat_gate).to(device)
    r0 = model.load_state_dict(load_file(args.checkpoint), strict=False)
    for p in model.parameters():
        p.requires_grad = False
    log(f'base checkpoint: missing={len(r0.missing_keys)} unexpected={len(r0.unexpected_keys)}')

    d = torch.load(args.trunk, map_location='cpu')
    log('trunk meta: ' + json.dumps({k: v for k, v in d.items() if k != 'state'}))
    layers = d.get('rca_layers') or []
    stack = RootCrossAttentionStack(
        model.d_model, model.morphemic_embed.root_embed, layers,
        num_heads=int(d.get('rca_heads', 8)), dropout=float(d.get('rca_dropout', 0.0)),
        dtype=torch.float32, out_norm=bool(d.get('rca_out_norm', False))).to(device)
    stack.exclude_current = bool(d.get('rca_exclude_current', False))
    model.root_cross = stack
    res = model.load_state_dict(
        {k: (v.to(model.dtype) if v.is_floating_point() and 'root_cross' not in k else v)
         for k, v in d['state'].items()}, strict=False)
    log(f'trunk state loaded: {len(d["state"])} tensors, missing={len(res.missing_keys)} '
        f'unexpected={len(res.unexpected_keys)}')
    if res.missing_keys:
        log('  MISSING KEYS: ' + ', '.join(res.missing_keys[:10]))
    stack.attach(model.backbone.layers)

    if args.head_dtype == 'fp32':
        model.nrmt_head.to(torch.float32)
    model.eval()
    model.backbone.eval()
    model.nrmt_head.eval()
    hres = model.nrmt_head.load_state_dict(hsd, strict=False)
    fg = (float(model.nrmt_head.feat_gate) if getattr(model.nrmt_head, 'feat_gate_enabled', False)
          else None)
    log(f'head: {len(hsd)} tensors, missing={len(hres.missing_keys)} '
        f'unexpected={len(hres.unexpected_keys)} dtype={next(model.nrmt_head.parameters()).dtype} '
        f'feat_gate_loaded={fg} feat_scale={model.nrmt_head.feat_scale():.4f}')
    if hres.unexpected_keys:
        log('  HEAD UNEXPECTED KEYS: ' + ', '.join(hres.unexpected_keys))
    if hres.missing_keys:
        log('  HEAD MISSING KEYS (stay at init!): ' + ', '.join(hres.missing_keys))
    log(f'RCA gates (loaded): {stack.gate_values()}')
    gates_on = stack.gate_values()

    # ================================================================= val windows
    def starts_for(t, n, seed):
        s = list(range(0, t.numel() - WIN - 1, STRIDE))
        np.random.default_rng(seed).shuffle(s)
        return sorted(s[:n])

    va_st = starts_for(va4[1], 300, 1)
    log(f'val starts: {len(va_st)} (trainer-identical)')

    def win_tensor(t, starts):
        return torch.stack([t[j:j + WIN] for j in starts])

    Tv, Wv, Pv, Sv = (win_tensor(va4[i], va_st) for i in (1, 2, 0, 3))
    Rt_va = Tv[:, 1:].contiguous()
    N, T1 = Rt_va.shape
    log(f'val windows: {N} x {T1} positions')

    nov = NT.Novelty(tr4[1].tolist())
    per = WIN - 1
    novel_mask = torch.zeros(N * per, dtype=torch.bool)
    for w in range(N):
        seg = Tv[w].tolist()
        for t in range(CTX, WIN):
            novel_mask[w * per + (t - 1)] = nov.novel(seg[t - CTX:t])
    keep_all = ~torch.isin(Rt_va.reshape(-1), spec_t)
    keep_novel = keep_all & novel_mask
    log(f'val positions: all={int(keep_all.sum())} novel={int(keep_novel.sum())}')

    op_table = build_operator_table(vocab)
    try:
        from sibawayh_governor import SibawayhGovernor
        gov = SibawayhGovernor(vocab)
        rows = [gov.op_ids_for_ids(rr, ww, pp)
                for pp, rr, ww in zip(Pv.tolist(), Tv.tolist(), Wv.tolist())]
        Ova_w = torch.tensor(rows, dtype=torch.long)[:, :-1]
        log('ʿāmil stream: SibawayhGovernor (persistent) -- as the trainer')
    except Exception as e:
        log(f'[warn] governor unavailable ({e}); using t-1 operator table')
        Ova_w = op_table[Tv.clamp(min=0)][:, :-1]

    true_roots = Rt_va.reshape(-1).numpy()
    gold_p = Pv[:, 1:].reshape(-1)
    gold_w = Wv[:, 1:].reshape(-1)
    gold_s = Sv[:, 1:].reshape(-1)

    # ================================================================= trunk + head driver
    def trunk_forward(p, r, w, s):
        emb = model.morphemic_embed(p, r, w, s)
        stack.set_root_ids(r)
        h = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state)
        return h.float()[:, :-1].contiguous()

    def run(ablate, batch=None):
        """Full 300-window pass. Returns per-position argmax / top5 / morph argmax."""
        batch = batch or args.batch
        old = stack.zero_gates() if ablate else None
        pr = np.empty(N * per, dtype=np.int32)
        t5 = np.empty((N * per, 5), dtype=np.int32)
        pw = np.empty(N * per, dtype=np.int32)
        pp = np.empty(N * per, dtype=np.int32)
        ps = np.empty(N * per, dtype=np.int32)
        cez = np.empty(N * per, dtype=np.float32)
        tgt_flat = Rt_va.reshape(-1).to(device)
        with torch.no_grad():
            for i in range(0, N, batch):
                sl = slice(i, min(i + batch, N))
                p, r, w, s = (t[sl].to(device) for t in (Pv, Tv, Wv, Sv))
                h = trunk_forward(p, r, w, s)
                o = model.nrmt_head(h, r[:, :-1], Ova_w[sl].to(device),
                                    w[:, :-1], p[:, :-1], s[:, :-1])
                rl = o['root_logits'].float().cpu()
                n = rl.shape[0] * rl.shape[1]
                pr[i * per:i * per + n] = rl.argmax(-1).reshape(-1).numpy()
                t5[i * per:i * per + n] = rl.topk(5, -1).indices.reshape(-1, 5).numpy()
                pw[i * per:i * per + n] = o['wazn_logits'].float().argmax(-1).reshape(-1).cpu().numpy()
                pp[i * per:i * per + n] = o['prefix_logits'].float().argmax(-1).reshape(-1).cpu().numpy()
                ps[i * per:i * per + n] = o['suffix_logits'].float().argmax(-1).reshape(-1).cpu().numpy()
                sd = rl.std(-1, keepdim=True).clamp_min(1e-6)
                z = (rl - rl.mean(-1, keepdim=True)) / sd
                tgt_b = tgt_flat[i * per:i * per + n].cpu()
                cez[i * per:i * per + n] = torch.nn.functional.cross_entropy(
                    z.reshape(-1, z.shape[-1]), tgt_b, reduction='none').cpu().numpy()
                del o, rl
        if old is not None:
            stack.restore_gates(old)
        return dict(pred=pr, top5=t5, pw=pw, pp=pp, ps=ps, cez=cez)

    # sanity: which pipeline is really loaded?
    log('=== aggregate pass: RCA ON ===')
    on = run(False)
    log('=== aggregate pass: RCA OFF (gates forced to 0) ===')
    off = run(True)

    def score(r, tag):
        out = {}
        for name, m in (('ALL_val', keep_all.numpy()), ('NOVEL_only', keep_novel.numpy())):
            if m.sum() == 0:
                continue
            pred, true = r['pred'][m], true_roots[m]
            a1 = float((pred == true).mean())
            a5 = float((r['top5'][m] == true[:, None]).any(-1).mean())
            ce_z = float(r['cez'][m].mean())
            out[name] = {'n': int(m.sum()), 'acc@1': a1, 'acc@5': a5, 'ce_z': ce_z,
                         'ppl_z': math.exp(min(ce_z, 25.0))}
        return out

    maj = majority_marginal(true_roots, keep_all.numpy())
    agg = {'RCA_ON': score(on, 'on'), 'RCA_OFF': score(off, 'off'),
           'majority_class_marginal_ALL_val': maj,
           'published_in_trainer_log': {'RCA_ON': {'ALL_val_acc@1': 0.19529, 'acc@5': 0.36414},
                                        'RCA_OFF': {'ALL_val_acc@1': 0.02825, 'acc@5': 0.05273}},
           'repro_eval_published': {'RCA_ON': {'ALL_val_acc@1': 0.11140, 'acc@5': 0.19370},
                                    'RCA_OFF': {'ALL_val_acc@1': 0.014362, 'acc@5': 0.037893}},
           'head_dtype': args.head_dtype,
           'feat_gate_built': bool(use_feat_gate),
           'feat_gate_value': fg,
           'rca_gates_loaded': gates_on,
           'note': 'repro_eval.py builds the head with feat_gate=False, which DROPS the trained '
                   'feat_gate scalar and makes the linear conditioning branch ~4.5x too strong; '
                   'that is the likely cause of its 11.14%/1.44% instead of 19.53%/2.83%.'}
    (out / 'aggregate_check.json').write_text(json.dumps(agg, indent=2))
    for k in ('RCA_ON', 'RCA_OFF'):
        d1 = agg[k]['ALL_val']
        log(f"  {k}: acc@1 {100*d1['acc@1']:.3f}%  acc@5 {100*d1['acc@5']:.3f}%  "
            f"CE_z {d1['ce_z']:.4f}  n={d1['n']}")
    log(f'  majority-class marginal: {100*maj:.3f}%')

    # ================================================================= rebuild val stream + surfaces
    reb_path = out / 'val_stream_rebuild.pt'
    if args.skip_rebuild and reb_path.exists():
        rb = torch.load(reb_path, map_location='cpu')
        streams, surfaces, word_sent, sent_meta = (rb['streams'], rb['surfaces'],
                                                   rb['word_sent'], rb['sent_meta'])
        log(f'reused rebuild: {len(surfaces)} word events')
    else:
        log('=== rebuilding the ALIGNED val stream from the corpus (surfaces included) ===')
        tt = time.time()
        all_pairs = []
        for c in NP.CORPORA:
            if not Path(c).exists():
                log(f'  [skip missing corpus] {c}')
                continue
            got = NP.sentences_from(c, min_words=4, max_words=64, limit=120000)
            all_pairs += got
            log(f'  {c}: {len(got)} sentences')
        seen, uniq = set(), []
        for f, s in all_pairs:
            if s in seen:
                continue
            seen.add(s)
            uniq.append((f, s))
        files = sorted({f for f, _ in uniq})
        random.Random(NP.SEED).shuffle(files)
        n_val = max(1, int(len(files) * 0.08))
        val_files = set(files[:n_val])
        log(f'  split(seed={NP.SEED}): {len(files)} files -> {len(val_files)} val files: '
            f'{sorted(Path(x).name for x in val_files)}')
        assert len(val_files) == 3, 'expected the documented 3 val files'
        streams = [[], [], [], []]
        surfaces = []
        word_sent = []
        sent_meta = []
        counters = {}
        for i, (f, s) in enumerate(uniq):
            if f not in val_files:
                continue
            enc = vocab.encode_sentence(s)
            if len(enc) < 2:
                continue
            counters[f] = counters.get(f, 0) + 1
            sid = len(sent_meta)
            sent_meta.append({'file': Path(f).name, 'sentence_ord': counters[f]})
            ws = s.split()
            for j, (p, r, w, sf) in enumerate(enc):
                streams[0].append(p); streams[1].append(r)
                streams[2].append(w); streams[3].append(sf)
                surfaces.append(ws[j] if j < len(ws) else '')
                word_sent.append(sid)
            if i and i % 100000 == 0:
                log(f'    {i}/{len(uniq)} val_words={len(surfaces)} [{time.time()-tt:.0f}s]')
        streams = [torch.tensor(x, dtype=torch.long) for x in streams]
        word_sent = torch.tensor(word_sent, dtype=torch.long)
        torch.save({'streams': streams, 'surfaces': surfaces, 'word_sent': word_sent,
                    'sent_meta': sent_meta}, reb_path)
        log(f'  rebuilt {len(surfaces)} val word events in {time.time()-tt:.0f}s')

    # prove the rebuild is the stream the model was scored on
    sh4 = [t.long() for t in torch.load(f'{args.cache}/val.pt', map_location='cpu')]
    names = ('prefix', 'root', 'wazn', 'suffix')
    ver = {}
    for k in range(4):
        same = bool(sh4[k].numel() == streams[k].numel()) and bool(
            torch.equal(sh4[k], streams[k]))
        ver[names[k]] = {'shipped_len': int(sh4[k].numel()),
                         'rebuilt_len': int(streams[k].numel()), 'identical': same}
        log(f'  rebuild check [{names[k]}]: shipped={sh4[k].numel()} '
            f'rebuilt={streams[k].numel()} identical={same}')
    assert all(v['identical'] for v in ver.values()), 'REBUILD DOES NOT MATCH THE CACHE'
    (out / 'rebuild_check.json').write_text(json.dumps(ver, indent=2))

    # ================================================================= gold round-trip (tokenizer defect)
    gold_rt = {'n_keep_all': int(keep_all.sum()), 'fail': 0, 'samples': []}
    for pos in np.nonzero(keep_all.numpy())[0][:4000]:
        w = pos // per
        t = pos % per
        gidx = va_st[w] + 1 + t
        gp, gr, gw, gs = (int(gold_p[pos]), int(true_roots[pos]), int(gold_w[pos]),
                          int(gold_s[pos]))
        dsurf = decode(gp, gr, gw, gs)
        if dsurf != surfaces[gidx]:
            gold_rt['fail'] += 1
            if len(gold_rt['samples']) < 8:
                gold_rt['samples'].append({'corpus': surfaces[gidx], 'realiser': dsurf,
                                           'root': root_name(gr)})
    gold_rt['checked'] = int(min(4000, keep_all.sum()))
    gold_rt['fail_pct'] = 100.0 * gold_rt['fail'] / max(gold_rt['checked'], 1)
    log(f"  gold-tuple round trip: {gold_rt['fail']}/{gold_rt['checked']} fail "
        f"({gold_rt['fail_pct']:.1f}%) -- the tokenizer's own form coverage")
    (out / 'gold_roundtrip.json').write_text(json.dumps(gold_rt, ensure_ascii=False, indent=2))

    # ================================================================= collapse metrics
    tr_counts = np.bincount(tr4[1].numpy().astype(np.int64), minlength=num_roots)
    order = np.argsort(-tr_counts, kind='stable')
    rank = np.empty(num_roots, dtype=np.int64)
    rank[order] = np.arange(num_roots)
    keep_np = keep_all.numpy()

    # ---- condition-independent reference predictors on the SAME positions ---------------
    Tvnp = Tv.numpy()
    true_np = true_roots
    # cur[w,t]  = the root fed in at window position t (the model sees it); target = t+1
    cur = Tvnp[:, :-1].reshape(-1)
    p1 = np.concatenate([np.full((N, 1), -1, dtype=np.int64), Tvnp[:, :WIN - 2]], axis=1).reshape(-1)
    p2 = np.concatenate([np.full((N, 2), -1, dtype=np.int64), Tvnp[:, :WIN - 3]], axis=1).reshape(-1)
    mask_p1 = keep_np & (p1 >= 0)
    mask_p2 = keep_np & (p2 >= 0)

    def rad_set(rid):
        s = root_name(rid)
        return set(s) if not s.startswith('<') else set()

    def overlap_profile(pred_arr, mask):
        rs = [rad_set(x) for x in pred_arr[mask]]
        ts = [rad_set(x) for x in true_np[mask]]
        c3 = sum(1 for a, b in zip(rs, ts) if a == b and len(b) == 3)
        c2 = sum(1 for a, b in zip(rs, ts) if a != b and len(a & b) == 2)
        c1 = sum(1 for a, b in zip(rs, ts) if len(a & b) <= 1)
        n = max(int(mask.sum()), 1)
        return {'n': int(mask.sum()), 'same_root_pct': 100.0 * c3 / n,
                'shares_exactly_2_radicals_pct': 100.0 * c2 / n,
                'shares_<=1_radical_pct': 100.0 * c1 / n}

    ref = {
        'prior_most_frequent_root_acc@1_pct': 100.0 * maj,
        'copy_current_root_r_t_acc@1_pct': 100.0 * float((cur[keep_np] == true_np[keep_np]).mean()),
        'copy_previous_root_r_t-1_acc@1_pct': 100.0 * float(
            (p1[mask_p1] == true_np[mask_p1]).mean()) if mask_p1.sum() else None,
        'copy_root_r_t-2_acc@1_pct': 100.0 * float(
            (p2[mask_p2] == true_np[mask_p2]).mean()) if mask_p2.sum() else None,
        'radical_overlap_of_copy_current_root': overlap_profile(cur, keep_np),
    }
    log('reference predictors: ' + ', '.join(
        f'{k}={v:.2f}%' for k, v in ref.items() if isinstance(v, float)))

    def metrics(r, tag):
        pred, top5 = r['pred'], r['top5']
        m = keep_np
        # distinctness over radical positions
        vals, cnts = np.unique(pred[m], return_counts=True)
        modal_root = int(vals[cnts.argmax()])
        # modality of the FULL emitted tuple
        tup = list(zip(r['pp'][m].tolist(), pred[m].tolist(), r['pw'][m].tolist(),
                       r['ps'][m].tolist()))
        tc = Counter(tup)
        (mp, mr, mw, ms), mn = tc.most_common(1)[0]
        modal_surface = decode(mp, mr, mw, ms)
        # adjacent-argmax change rate, per window (in-window adjacency only)
        P = pred.reshape(N, per)
        chg = (P[:, 1:] != P[:, :-1])
        inw_all = float(chg.mean())
        K = keep_all.reshape(N, per).numpy()
        pair_ok = K[:, 1:] & K[:, :-1]
        inw_rad = float(chg[pair_ok].mean()) if pair_ok.sum() else None
        # leak profile
        lc = Counter()
        for p_, r_, w_, s_ in zip(r['pp'][m].tolist(), pred[m].tolist(),
                                  r['pw'][m].tolist(), r['ps'][m].tolist()):
            lc[leak_class(p_, r_, w_, s_)[0]] += 1
        # frequency rank vs correctness
        mrank = rank[pred[m]]
        correct = pred[m] == true_roots[m]
        buckets = [(0, 9), (10, 99), (100, 999), (1000, 9999), (10000, 10 ** 9)]
        buck = {}
        for lo, hi in buckets:
            b = (mrank >= lo) & (mrank <= hi)
            if b.sum() == 0:
                continue
            buck[f'rank_{lo+1}_{hi if hi < 10**9 else "inf"}'] = {
                'n': int(b.sum()), 'share_pct': 100.0 * float(b.mean()),
                'acc@1_pct': 100.0 * float(correct[b].mean())}
        return {
            'n_positions': int(m.sum()),
            'distinct_top1_roots': int(len(vals)),
            'distinct_top1_pct': 100.0 * len(vals) / max(int(m.sum()), 1),
            'modal_root': root_name(modal_root), 'modal_root_count': int(cnts.max()),
            'modal_root_share_pct': 100.0 * float(cnts.max()) / max(int(m.sum()), 1),
            'distinct_tuples': len(tc),
            'modal_tuple': [int(mp), int(mr), int(mw), int(ms)],
            'modal_tuple_roots': root_name(mr),
            'modal_tuple_surface': modal_surface,
            'modal_tuple_count': int(mn),
            'modal_tuple_share_pct': 100.0 * mn / max(int(m.sum()), 1),
            'adjacent_argmax_change_rate_inwindow_pct': 100.0 * inw_all,
            'adjacent_argmax_change_rate_radical_pairs_pct': (
                None if inw_rad is None else 100.0 * inw_rad),
            'leak_profile_counts': dict(lc),
            'leak_literal_control_token_pct': 100.0 * lc.get('LITERAL_CONTROL_TOKEN_IN_TEXT', 0)
                                             / max(int(m.sum()), 1),
            'leak_empty_surface_pct': 100.0 * lc.get('empty_surface', 0) / max(int(m.sum()), 1),
            'control_root_pct': 100.0 * (
                lc.get('control_root_(PAD/BOS/EOS)->empty', 0)
                + lc.get('UNK_root_(affixes only)', 0)
                + lc.get('<PARTICLE>_root_(affixes only)', 0)) / max(int(m.sum()), 1),
            'acc_by_pred_root_freq_rank': buck,
            'mean_pred_root_rank': float(mrank.mean()),
            'median_pred_root_rank': float(np.median(mrank)),
            'mean_pred_root_rank_CORRECT': (
                float(mrank[correct].mean()) if correct.any() else None),
            'mean_pred_root_rank_WRONG': (
                float(mrank[~correct].mean()) if (~correct).any() else None),
            'median_pred_root_rank_CORRECT': (
                float(np.median(mrank[correct])) if correct.any() else None),
            'median_pred_root_rank_WRONG': (
                float(np.median(mrank[~correct])) if (~correct).any() else None),
            'pred_in_top10_frequent_pct': 100.0 * float((mrank < 10).mean()),
            'pred_in_top100_frequent_pct': 100.0 * float((mrank < 100).mean()),
            'pred_in_top1000_frequent_pct': 100.0 * float((mrank < 1000).mean()),
            'true_root_median_rank': float(np.median(rank[true_roots[m]])),
            'acc@1_pct': 100.0 * float(correct.mean()),
            'echo_diagnostics': {
                'pred_equals_current_root_r_t_pct': 100.0 * float((pred[m] == cur[m]).mean()),
                'pred_equals_prev_root_r_t-1_pct': (
                    100.0 * float((pred[mask_p1] == p1[mask_p1]).mean())
                    if mask_p1.sum() else None),
                'pred_equals_root_r_t-2_pct': (
                    100.0 * float((pred[mask_p2] == p2[mask_p2]).mean())
                    if mask_p2.sum() else None),
                'radical_overlap_of_prediction': overlap_profile(pred, m),
            },
        }

    met = {'RCA_ON': metrics(on, 'on'), 'RCA_OFF': metrics(off, 'off'),
           'majority_class_marginal_pct': 100.0 * maj,
           'reference_condition_independent_predictors': ref,
           'head_dtype': args.head_dtype, 'rca_gates': gates_on}
    np.save(out / 'pred_on_all_positions.npy', on['pred'])
    np.save(out / 'pred_off_all_positions.npy', off['pred'])
    np.save(out / 'keep_all_mask.npy', keep_all.numpy())

    # RCA-OFF emission profile with counts (the qualitative form of "below the marginal")
    def emission_profile(r):
        m = keep_np
        rc = Counter(r['pred'][m].tolist())
        tupc = Counter(zip(r['pp'][m].tolist(), r['pred'][m].tolist(),
                           r['pw'][m].tolist(), r['ps'][m].tolist()))
        mod_root_decode = {}
        for t, c in tupc.most_common():
            mod_root_decode.setdefault(t[1], (decode(*t), c))
        tot = int(m.sum())
        return {
            'n': tot,
            'distinct_roots': len(rc),
            'top_roots': [{'root': root_name(k), 'root_id': int(k), 'count': int(c),
                           'share_pct': 100.0 * c / tot,
                           'modal_tuple_surface_for_this_root': mod_root_decode.get(k, ('', 0))[0]}
                          for k, c in rc.most_common(12)],
            'PARTICLE_count': int(rc.get(PARTICLE, 0)),
            'PARTICLE_share_pct': 100.0 * rc.get(PARTICLE, 0) / tot,
            'PAD_count': int(rc.get(PAD, 0)), 'BOS_count': int(rc.get(BOS, 0)),
            'EOS_count': int(rc.get(EOS, 0)), 'UNK_count': int(rc.get(UNK, 0)),
            'top_tuples': [{'tuple': [int(x) for x in t], 'root': root_name(t[1]),
                            'surface': decode(*t), 'count': int(c), 'share_pct': 100.0 * c / tot}
                           for t, c in tupc.most_common(10)],
        }

    met['RCA_OFF_emission_profile'] = emission_profile(off)
    met['RCA_ON_emission_profile'] = emission_profile(on)
    (out / 'metrics.json').write_text(json.dumps(met, ensure_ascii=False, indent=2))

    for k in ('RCA_ON', 'RCA_OFF'):
        d1 = met[k]
        log(f"--- {k}: distinct top1 {d1['distinct_top1_roots']} "
            f"({d1['distinct_top1_pct']:.2f}%) modal '{d1['modal_root']}' "
            f"{d1['modal_root_share_pct']:.2f}% | modal tuple '{d1['modal_tuple_surface']}' "
            f"{d1['modal_tuple_share_pct']:.2f}% | adj-change {d1['adjacent_argmax_change_rate_inwindow_pct']:.2f}% | "
            f"literal-leak {d1['leak_literal_control_token_pct']:.2f}% | "
            f"empty {d1['leak_empty_surface_pct']:.2f}% | control-root {d1['control_root_pct']:.2f}% | "
            f"pred-rank mean {d1['mean_pred_root_rank']:.0f}")
    ep = met['RCA_OFF_emission_profile']
    log(f"RCA_OFF top roots: " + ', '.join(
        f"{x['root']}x{x['count']}" for x in ep['top_roots'][:8]))

    # ================================================================= prompts
    log(f'=== prompts: {args.n_prompts} windows sampled uniformly from the 300 val starts '
        f'(seed {args.seed}) ===')
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(len(va_st), size=min(args.n_prompts, len(va_st)), replace=False))
    win_idx = [int(x) for x in pick]
    s_idx = torch.tensor([va_st[i] for i in win_idx])
    log(f'picked window indices: {win_idx}')

    def prompt_pass(ablate):
        old = stack.zero_gates() if ablate else None
        outs = []
        with torch.no_grad():
            for i in range(0, len(win_idx), args.batch):
                sl = win_idx[i:i + args.batch]
                p, r, w, s = (t[sl].to(device) for t in (Pv, Tv, Wv, Sv))
                h = trunk_forward(p, r, w, s)
                o = model.nrmt_head(h, r[:, :-1], Ova_w[sl].to(device),
                                    w[:, :-1], p[:, :-1], s[:, :-1])
                # top-5 tuples, each candidate root conditioning its own morph heads
                rl = o['root_logits'][:, TGT_LOCAL, :].float()
                a5 = rl.topk(5, -1).indices                       # [b,5]
                haug = o['root_states'][:, TGT_LOCAL, :].float()  # [b,d]
                outs.append({'logits_t': rl.float().cpu(), 'top5': a5.cpu(),
                             'haug': haug.cpu(),
                             'pw': o['wazn_logits'][:, TGT_LOCAL, :].float().cpu(),
                             'pp': o['prefix_logits'][:, TGT_LOCAL, :].float().cpu(),
                             'ps': o['suffix_logits'][:, TGT_LOCAL, :].float().cpu()})
                del o, rl
        if old is not None:
            stack.restore_gates(old)
        return outs

    pon, poff = prompt_pass(False), prompt_pass(True)

    def top5_tuples(blk):
        """[b,5] roots -> decoded tuple per candidate root (own conditioning)."""
        res = []
        for bi in range(blk['top5'].shape[0]):
            haug = blk['haug'][bi:bi + 1].to(device)
            row = []
            for c in blk['top5'][bi].tolist():
                cid = torch.tensor([c], device=device, dtype=torch.long)
                e = model.nrmt_head._root_embed(cid).to(haug.dtype)
                hc = model.nrmt_head.cond_proj(torch.cat([haug, e], dim=-1))
                w_ = int(model.nrmt_head.wazn_head(hc).float().argmax(-1))
                p_ = int(model.nrmt_head.prefix_head(hc).float().argmax(-1))
                s_ = int(model.nrmt_head.suffix_head(hc).float().argmax(-1))
                row.append({'root': root_name(c), 'root_id': int(c),
                            'tuple': [p_, c, w_, s_], 'surface': decode(p_, c, w_, s_),
                            'logit': float(blk['logits_t'][bi, c])})
            res.append(row)
        return res

    def gather(blocks):
        return {k: torch.cat([b[k] for b in blocks]) for k in blocks[0]}

    gon, goff = gather(pon), gather(poff)
    t5on = top5_tuples(gon)
    t5off = top5_tuples(goff)

    prompts = []
    for bi, wi in enumerate(win_idx):
        s = va_st[wi]
        tgt_idx = s + TGT_LOCAL + 1
        ctx_words = [surfaces[j] for j in range(max(0, tgt_idx - 24), tgt_idx)]
        gp, gr, gw, gs = (int(streams[0][tgt_idx]), int(streams[1][tgt_idx]),
                          int(streams[2][tgt_idx]), int(streams[3][tgt_idx]))
        truth_surface = surfaces[tgt_idx]
        gold_realised = decode(gp, gr, gw, gs)
        r_on = int(gon['logits_t'][bi].argmax(-1))
        r_off = int(goff['logits_t'][bi].argmax(-1))
        # top-1 tuples for both conditions
        def top1(blk, r):
            haug = blk['haug'][bi:bi + 1].to(device)
            cid = torch.tensor([r], device=device, dtype=torch.long)
            e = model.nrmt_head._root_embed(cid).to(haug.dtype)
            hc = model.nrmt_head.cond_proj(torch.cat([haug, e], dim=-1))
            w_ = int(model.nrmt_head.wazn_head(hc).float().argmax(-1))
            p_ = int(model.nrmt_head.prefix_head(hc).float().argmax(-1))
            s_ = int(model.nrmt_head.suffix_head(hc).float().argmax(-1))
            return [p_, r, w_, s_], decode(p_, r, w_, s_)
        ton, son_ = top1(gon, r_on)
        toff, soff_ = top1(goff, r_off)
        sid = int(word_sent[tgt_idx])
        prompts.append({
            'i': bi + 1, 'window_start': int(s), 'target_stream_index': int(tgt_idx),
            'source_file': sent_meta[sid]['file'], 'sentence_ord_in_file': sent_meta[sid]['sentence_ord'],
            'context_words': ctx_words,
            'truth_surface': truth_surface,
            'truth_root': root_name(gr), 'truth_root_id': gr,
            'truth_tuple': [gp, gr, gw, gs], 'truth_realised': gold_realised,
            'truth_root_freq_rank': int(rank[gr]) + 1,
            'truth_root_train_count': int(tr_counts[gr]),
            'truth_is_radical_not_counted_by_metric': bool(gr in set(specials)),
            'truth_is_novel_12gram_context': bool(novel_mask[wi * per + TGT_LOCAL - 1]),
            'rca_on': {'root': root_name(r_on), 'root_id': r_on,
                       'root_freq_rank': int(rank[r_on]) + 1, 'tuple': ton, 'surface': son_,
                       'correct': bool(r_on == gr), 'top5': t5on[bi]},
            'rca_off': {'root': root_name(r_off), 'root_id': r_off,
                        'root_freq_rank': int(rank[r_off]) + 1, 'tuple': toff, 'surface': soff_,
                        'correct': bool(r_off == gr), 'top5': t5off[bi]},
            'same_root_prediction_in_both_conditions': bool(r_on == r_off),
        })

    (out / 'prompts.json').write_text(json.dumps(prompts, ensure_ascii=False, indent=2))

    # readable blocks
    L = []
    L.append('# RCA prompt read -- head_RCA_UNFREEZE_A2 vs its own gate-ablation\n')
    L.append(f'- sampling: {len(win_idx)} of the trainer\'s 300 held-out val windows, drawn '
             f'uniformly at random without replacement, seed **{args.seed}** (`numpy '
             f'default_rng({args.seed}).choice(300, {len(win_idx)}, replace=False)`).\n')
    L.append(f'- every prompt feeds the model the full 128-word held-out window '
             f'(identical windows to the ones the 19.53 % aggregate was computed on); the '
             f'prediction is read at window position {TGT_LOCAL}, i.e. for window word '
             f'{TGT_LOCAL+1}.\n')
    L.append('- the 24 words before the target are shown as the "context"; `…` marks the fact '
             'that the model saw further words to the left.\n')
    L.append('- surfaces are produced by `nrmp_vocab.MorphemicVocab.decode_word` (the '
             'project\'s own realiser), applied identically to the model\'s tuple and to the '
             'gold tuple.\n')
    nc_rad = sum(1 for p in prompts if not p['truth_is_radical_not_counted_by_metric'])
    L.append(f'- {nc_rad}/{len(prompts)} prompts have a target that the project\'s own metric '
             'counts as a radical (non-special) position; the other '
             f'{len(prompts)-nc_rad} have a particle/control target root, kept in because the '
             'sample is uncurated.\n')
    L.append('')
    for p in prompts:
        rad = ('**not** counted by the metric (particle/control root)'
               if p['truth_is_radical_not_counted_by_metric'] else 'radical, counted by the metric')
        L.append(f"## P{p['i']:02d} — {p['source_file']} (sentence {p['sentence_ord_in_file']}), "
                 f"window {p['window_start']}, target stream idx {p['target_stream_index']}")
        L.append(f"*target: {rad}; 12-gram context novel: "
                 f"{p['truth_is_novel_12gram_context']}*")
        L.append('')
        L.append('**Context (last 24 of the 128 window words; … = more words to the left):**')
        L.append('')
        L.append('… ' + ' '.join(p['context_words']))
        L.append('')
        L.append(f"**True next word:** `{p['truth_surface']}`  "
                 f"(root `{p['truth_root']}` = `{p['truth_root_id']}`, freq rank "
                 f"{p['truth_root_freq_rank']} of {num_roots}, seen "
                 f"{p['truth_root_train_count']}x in train)  — gold tuple "
                 f"`{tuple(p['truth_tuple'])}` realises as `{p['truth_realised']}`")
        L.append('')
        for cond in ('rca_on', 'rca_off'):
            c = p[cond]
            tag = 'RCA ON  (gates loaded)' if cond == 'rca_on' else 'RCA OFF (gates forced to 0)'
            L.append(f"**{tag} — top-1:** root `{c['root']}` (id {c['root_id']}, freq rank "
                     f"{c['root_freq_rank']}) → tuple `{tuple(c['tuple'])}` → "
                     f"`{c['surface']}`   {'✅ correct root' if c['correct'] else '❌ wrong root'}")
            L.append('')
            L.append(f"  top-5: " + ' | '.join(
                f"{j+1}. `{x['surface']}` (root `{x['root']}`)" for j, x in enumerate(c['top5'])))
            L.append('')
        L.append('---')
        L.append('')
    (out / 'prompts.md').write_text('\n'.join(L), encoding='utf-8')

    # prompt-level summary
    sumr = {
        'n_prompts': len(prompts),
        'n_target_radical': int(sum(1 for p in prompts
                                    if not p['truth_is_radical_not_counted_by_metric'])),
        'RCA_ON_root_top1_correct': int(sum(1 for p in prompts if p['rca_on']['correct'])),
        'RCA_OFF_root_top1_correct': int(sum(1 for p in prompts if p['rca_off']['correct'])),
        'RCA_ON_top5_contains_truth': int(sum(
            1 for p in prompts if any(x['root_id'] == p['truth_root_id'] for x in p['rca_on']['top5']))),
        'RCA_OFF_top5_contains_truth': int(sum(
            1 for p in prompts if any(x['root_id'] == p['truth_root_id'] for x in p['rca_off']['top5']))),
        'RCA_ON_distinct_top1_surfaces': len({p['rca_on']['surface'] for p in prompts}),
        'RCA_OFF_distinct_top1_surfaces': len({p['rca_off']['surface'] for p in prompts}),
        'RCA_ON_empty_surface_prompts': int(sum(1 for p in prompts if p['rca_on']['surface'] == '')),
        'RCA_OFF_empty_surface_prompts': int(sum(1 for p in prompts if p['rca_off']['surface'] == '')),
        'RCA_OFF_modal_surface': Counter(p['rca_off']['surface'] for p in prompts).most_common(3),
        'RCA_ON_modal_surface': Counter(p['rca_on']['surface'] for p in prompts).most_common(3),
        'RCA_OFF_modal_root': Counter(p['rca_off']['root'] for p in prompts).most_common(3),
        'RCA_ON_modal_root': Counter(p['rca_on']['root'] for p in prompts).most_common(3),
        'same_root_prediction_in_both_conditions': int(
            sum(1 for p in prompts if p['same_root_prediction_in_both_conditions'])),
        'aggregate_self_test': {'RCA_ON_acc@1_pct': 100 * agg['RCA_ON']['ALL_val']['acc@1'],
                                'RCA_OFF_acc@1_pct': 100 * agg['RCA_OFF']['ALL_val']['acc@1'],
                                'majority_marginal_pct': 100 * maj},
    }
    (out / 'prompt_summary.json').write_text(json.dumps(sumr, ensure_ascii=False, indent=2))
    log('prompt summary: ' + json.dumps(sumr, ensure_ascii=False))
    log(f'TOTAL {time.time()-t0:.0f}s -> {out}')
    logf.close()


if __name__ == '__main__':
    main()
