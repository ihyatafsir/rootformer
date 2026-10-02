#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prompt_predict.py -- readable next-word prediction from sf_morph.pt.

Loads the checkpoint the same way scratch_lm.py does and exposes ONE prompt interface:
give it an Arabic word context, it returns the predicted next word's tuple
(prefix, root, wazn, suffix) and the surface Arabic, realised with the project's own
realiser (nrmp_vocab.FarahidianMorphemicVocab.decode_word).  Nothing here is a parallel format.

The joint top-5 is a real 5*5*5*5 = 625-way exhaustive beam over the four head marginals,
scored with the SAME weighted sum training minimises:
    logp(root) + 0.5*logp(wazn) + 0.25*logp(prefix) + 0.25*logp(suffix)

Modes
  interactive          one context per line on stdin
  sample N [SEED]      N uniform-random real prompts from the three prompt corpora
  heldout_corpus N     real corpus sentences whose target is a word from the 38 held-out roots
  heldout_stream N     contexts taken from test_deriv (guaranteed held-out-root targets)
"""
import json
import random
import sys

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/scratch_prompt')

D = '/workspace/sf_data'
OUT = '/workspace/scratch_prompt'
CKPT = '/workspace/scratch_lm_run/sf_morph.pt'
CTX = 128

meta = json.load(open(f'{D}/meta.json'))
NR, NW, NP, NS = meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes']
SPECIAL = set(meta['special_root_ids'])
HOLD = set(meta['hold_roots'])

import scratch_lm as S        # noqa: E402
import nrmp_vocab as nv       # noqa: E402

_V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
VOCAB = _V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
device = torch.device('cpu')

model = S.MorphemicLM((NR, NW, NP, NS)).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()

HEADS = ('p', 'r', 'w', 's')
HNAME = {'p': 'prefix', 'r': 'root', 'w': 'wazn', 's': 'suffix'}
HDESC = {'p': 'prefix', 'r': 'root', 'w': 'wazn', 's': 'suffix'}
W_JOINT = {'r': 1.0, 'w': 0.5, 'p': 0.25, 's': 0.25}


def name_of(h, i):
    return (VOCAB.id2prefix.get(i, '?') if h == 'p' else
            VOCAB.id2root.get(i, '?') if h == 'r' else
            VOCAB.id2wazn.get(i, '?') if h == 'w' else VOCAB.id2suffix.get(i, '?'))


def tag_of_root(i):
    return 'SPECIAL' if i in SPECIAL else ('HELD-OUT' if i in HOLD else 'real')


LIMITS = {'p': NP, 'r': NR, 'w': NW, 's': NS}
OVER = Counter()          # how many encoded tokens fell outside the checkpoint id space


def check_enc(enc):
    """Trim an encoded sequence at the first id the CHECKPOINT cannot embed.

    The live nrmp_vocab was rebuilt on 2026-10-01 18:07 and now carries 9,490 roots / 142
    awzan, while sf_morph.pt was trained under the earlier 9,114 / 130 id space.  New ids
    were appended, so old ids are stable (verified in verify_ids.py) -- but an encoded
    token can still land on a NEW id, which would index past the checkpoint's embedding.
    """
    keep = []
    for t in enc:
        if any(t[k] >= LIMITS[h] for k, h in ((0, 'p'), (1, 'r'), (2, 'w'), (3, 's'))):
            OVER['trimmed'] += 1
            break
        keep.append(t)
    return keep


def _logp(lg, h):
    return torch.log_softmax(lg[h][0, -1], -1)


def predict(words, beam=6, max_ctx=CTX):
    """words: list of surface Arabic words.  Predicts the tuple of the NEXT word."""
    enc = check_enc(VOCAB.encode_sentence(' '.join(words))[:max_ctx])
    if not enc:
        return None
    ctx = {h: [[t[k] for t in enc]] for h, k in
           (('p', 0), ('r', 1), ('w', 2), ('s', 3))}
    with torch.no_grad():
        lr, lw, lp, ls = model(S.pad(ctx['p'], 0), S.pad(ctx['r'], 0),
                               S.pad(ctx['w'], 0), S.pad(ctx['s'], 0))
    lg = {'r': lr, 'w': lw, 'p': lp, 's': ls}
    lp_h = {h: _logp(lg, h) for h in HEADS}
    per_head = {}
    for h in HEADS:
        v, i = lp_h[h].topk(beam)
        per_head[h] = [{'id': int(ii), 'name': name_of(h, int(ii)), 'logp': float(vv)}
                       for vv, ii in zip(v, i)]
    # exhaustive joint beam over the top-`beam` of each head
    cand = []
    for pr in per_head['p']:
        for ro in per_head['r']:
            for wa in per_head['w']:
                for su in per_head['s']:
                    sc = (W_JOINT['p'] * pr['logp'] + W_JOINT['r'] * ro['logp']
                          + W_JOINT['w'] * wa['logp'] + W_JOINT['s'] * su['logp'])
                    cand.append((sc, (pr['id'], ro['id'], wa['id'], su['id'])))
    cand.sort(key=lambda x: -x[0])
    joint = [{'tuple': list(t), 'score': float(s_), 'surface': VOCAB.decode_word(*t),
              'prefix': VOCAB.id2prefix.get(t[0], '?'),
              'root': VOCAB.id2root.get(t[1], '?'),
              'wazn': VOCAB.id2wazn.get(t[2], '?'),
              'suffix': VOCAB.id2suffix.get(t[3], '?'),
              'root_tag': tag_of_root(t[1])} for s_, t in cand[:5]]
    t1 = tuple(int(lg[h][0, -1].argmax()) for h in HEADS)
    return {'enc': enc, 'per_head': per_head, 't1': t1,
            't1_surface': VOCAB.decode_word(*t1), 'joint': joint}


def describe(words, truth=None, label=None, beam=6):
    r = predict(words, beam=beam)
    if r is None:
        return f'--- {label or ""}\nCONTEXT : {" ".join(words)}\n   [SKIPPED: empty encoding]', None
    t1 = r['t1']
    L = []
    if label:
        L.append(f'--- {label}')
    L.append('CONTEXT : ' + ' '.join(words))
    if truth is not None:
        L.append(f'TRUE NEXT: {truth["surface"]!r}   tuple=({truth["p"]},{truth["r"]},'
                 f'{truth["w"]},{truth["s"]})  [{VOCAB.id2prefix.get(truth["p"],"?")}'
                 f'+{VOCAB.id2root.get(truth["r"],"?")}+{VOCAB.id2wazn.get(truth["w"],"?")}'
                 f'+{VOCAB.id2suffix.get(truth["s"],"?")}]  root_tag={tag_of_root(truth["r"])}')
    L.append(f'PRED top1: {r["t1_surface"]!r}   tuple={t1}  '
             f'[{VOCAB.id2prefix.get(t1[0],"?")}+{VOCAB.id2root.get(t1[1],"?")}'
             f'+{VOCAB.id2wazn.get(t1[2],"?")}+{VOCAB.id2suffix.get(t1[3],"?")}]'
             f'  root_tag={tag_of_root(t1[1])}')
    for h in HEADS:
        s = '  '.join(f'{x["name"]!r}:{x["logp"]:.2f}' for x in r['per_head'][h])
        L.append(f'   {HDESC[h]:6s} top5: {s}')
    L.append('   joint top5 (weighted tuple beam):')
    for j in r['joint']:
        L.append(f'      {j["score"]:8.2f}  {j["surface"]!r:20s} '
                 f'[{j["prefix"]}+{j["root"]}+{j["wazn"]}+{j["suffix"]}] root_tag={j["root_tag"]}')
    if truth is not None:
        h1 = [h for h in HEADS if r['per_head'][h][0]['id'] == truth[h]]
        h5 = [h for h in HEADS if truth[h] in [x['id'] for x in r['per_head'][h]]]
        jhit = [i for i, j in enumerate(r['joint'])
                if tuple(j['tuple']) == (truth['p'], truth['r'], truth['w'], truth['s'])]
        jr = [i for i, j in enumerate(r['joint']) if j['tuple'][1] == truth['r']]
        L.append(f'   SCORE: per-head hit@1={h1 or "none"}  hit@5={h5 or "none"}  '
                 f'| joint tuple hit@5={"yes@"+str(jhit[0]) if jhit else "no"}  '
                 f'| joint root-in-top5={"yes" if jr else "no"}')
    return '\n'.join(L), r


# ---------------------------------------------------------------- sampling
def _cut(ws, frac=0.7):
    c = max(3, int(len(ws) * frac))
    return ws[:c], ws[c]


def _tuple_at(ws, k):
    """Tuple of word k, encoded so that position k is itself inside the checkpoint id space.

    Words BEFORE k that encode to a post-checkpoint id are the only problem; if one occurs,
    this returns None and the prompt is dropped (recorded in OVER['dropped']).
    """
    if k >= len(ws):
        return None
    e = VOCAB.encode_sentence(' '.join(ws[:k + 1]))
    if len(e) != k + 1:
        return None
    for t in e:
        if any(t[i] >= LIMITS[h] for i, h in ((0, 'p'), (1, 'r'), (2, 'w'), (3, 's'))):
            OVER['dropped'] += 1
            return None
    t = e[k]
    return {'p': t[0], 'r': t[1], 'w': t[2], 's': t[3], 'surface': ws[k]}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'interactive'
    if mode == 'interactive':
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            txt, _ = describe(line.split())
            print(txt, flush=True)
            print(flush=True)
        return
    import corpus_read as CR

    if mode == 'sample':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261002
        pool = list(CR.sentences())
        print(f'[*] {len(pool)} candidate sentences in the three prompt corpora', flush=True)
        rng = random.Random(seed)
        rng.shuffle(pool)
        picks, seen = [], set()
        for corpus, rel, s in pool:
            ws = s.split()
            if len(ws) < 8 or len(ws) > 40 or s in seen:
                continue
            seen.add(s)
            picks.append((corpus, rel, s))
            if len(picks) >= n:
                break
        print(f'[sampling] uniform random over {len(pool)} sentences (seed {seed}); '
              f'only filters: 8<=words<=40 and exact-duplicate removal.  '
              f'No quality curation.  kept={len(picks)}')
        out = []
        for corpus, rel, s in picks:
            ws = s.split()
            ctxlen = len(ws) - 1
            if ctxlen > 40:
                ctxlen = max(3, int(ctxlen * 0.7))
            tt = _tuple_at(ws, ctxlen)
            if tt is None:
                continue
            ctxw = ws[:ctxlen]
            label = f'{corpus.split("/")[-1]}/{rel}   target="{ws[ctxlen]}"'
            txt, r = describe(ctxw, truth=tt, label=label)
            if r is None:
                continue
            out.append({'corpus': corpus, 'file': rel, 'context': ' '.join(ctxw),
                        'context_words': ctxw, 'true': ws[ctxlen],
                        'true_tuple': [tt['p'], tt['r'], tt['w'], tt['s']],
                        'pred_top1_tuple': list(r['t1']),
                        'pred_top1_surface': r['t1_surface'],
                        'joint_top5': r['joint'],
                        'per_head': {h: r['per_head'][h] for h in HEADS},
                        'text': txt})
            print(txt); print()
        json.dump(out, open(f'{OUT}/prompts_sample.json', 'w'), ensure_ascii=False, indent=2)
        print(f'[over-range id trims={OVER["trimmed"]} dropped prompts={OVER["dropped"]}]')
        print(f'wrote {OUT}/prompts_sample.json')
        _summarise(out, 'sample')

    elif mode == 'heldout_corpus':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        # find real corpus sentences containing a word that ENCODES to a held-out root
        want = {}
        scanned = 0
        for corpus, rel, s in CR.sentences():
            scanned += 1
            enc = VOCAB.encode_sentence(s)
            hits = [(k, t) for k, t in enumerate(enc) if t[1] in HOLD]
            for k, t in hits:
                if k >= 3:
                    want.setdefault(t[1], []).append((corpus, rel, s, k))
            if sum(len(v) for v in want.values()) > 4000:
                break
        print(f'[*] scanned {scanned} corpus sentences; held-out-root targets found for '
              f'{len(want)} of the {len(HOLD)} held-out roots', flush=True)
        roots = sorted(want, key=lambda r: -len(want[r]))
        out = []
        for rid in roots:
            if len(out) >= n:
                break
            corpus, rel, s, k = want[rid][0]
            ws = s.split()
            tt = _tuple_at(ws, k)
            if tt is None or tt['r'] != rid:
                continue
            ctxw = [w for w in ws[:k] if w]
            if len(ctxw) < 2:
                continue
            label = (f'{corpus.split("/")[-1]}/{rel}  held-out root '
                     f'{VOCAB.id2root.get(rid,"?")!r} (id {rid}) at target position {k}')
            txt, r = describe(ctxw, truth=tt, label=label)
            out.append({'corpus': corpus, 'file': rel, 'context': ' '.join(ctxw),
                        'context_words': ctxw, 'true': ws[k],
                        'true_tuple': [tt['p'], tt['r'], tt['w'], tt['s']],
                        'heldout_root_id': rid, 'heldout_root': VOCAB.id2root.get(rid, '?'),
                        'pred_top1_tuple': list(r['t1']),
                        'pred_top1_surface': r['t1_surface'],
                        'joint_top5': r['joint'],
                        'text': txt})
            print(txt); print()
        json.dump(out, open(f'{OUT}/heldout_corpus_prompts.json', 'w'),
                  ensure_ascii=False, indent=2)
        print(f'[over-range id trims={OVER["trimmed"]} dropped prompts={OVER["dropped"]}]')
        print(f'wrote {OUT}/heldout_corpus_prompts.json')
        _summarise(out, 'heldout_corpus')

    elif mode == 'heldout_stream':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        d = None
        print('[*] loading streams.pt ...', flush=True)
        d = torch.load(f'{D}/streams.pt', weights_only=False)['test_deriv']
        idx = [i for i, seq in enumerate(d['R']) if set(seq) & HOLD]
        print(f'[*] test_deriv sentences containing a held-out root: {len(idx)}', flush=True)
        rng = random.Random(5)
        rng.shuffle(idx)
        out = []
        for i in idx:
            if len(out) >= n:
                break
            P_, R_, W_, S_ = d['P'][i], d['R'][i], d['W'][i], d['S'][i]
            pos = [k for k in range(3, len(R_)) if R_[k] in HOLD]
            if not pos:
                continue
            k = pos[0]
            ctxw = [VOCAB.decode_word(*t) for t in zip(P_[:k], R_[:k], W_[:k], S_[:k])]
            ctxw = [w for w in ctxw if w]
            if len(ctxw) < 2:
                continue
            # the model only ever sees what the CURRENT encoder produces, so re-encode the
            # realised context and require the re-encoding to stay inside the checkpoint space
            # AND to still end on the held-out root we meant to test.
            enc = check_enc(VOCAB.encode_sentence(' '.join(ctxw)))
            if len(enc) != len(ctxw):
                OVER['dropped'] += 1
                continue
            if len(enc) + 1 > CTX:
                enc = enc[-(CTX - 1):]
                ctxw = ctxw[-(CTX - 1):]
            nxt = VOCAB.encode_word(VOCAB.decode_word(P_[k], R_[k], W_[k], S_[k]))
            if any(nxt[i] >= LIMITS[h] for i, h in ((0, 'p'), (1, 'r'), (2, 'w'), (3, 's'))):
                OVER['dropped'] += 1
                continue
            if nxt[1] != R_[k]:
                OVER['dropped'] += 1
                continue
            tt = {'p': nxt[0], 'r': nxt[1], 'w': nxt[2], 's': nxt[3],
                  'surface': VOCAB.decode_word(*nxt)}
            label = (f'test_deriv sentence {i} @ target {k}  held-out root '
                     f'{VOCAB.id2root.get(R_[k],"?")!r} (id {R_[k]})')
            txt, r = describe(ctxw, truth=tt, label=label)
            out.append({'sent': i, 'pos': k, 'context_words': ctxw,
                        'context': ' '.join(ctxw),
                        'true_tuple': [tt['p'], tt['r'], tt['w'], tt['s']],
                        'true_surface': tt['surface'],
                        'heldout_root_id': R_[k], 'heldout_root': VOCAB.id2root.get(R_[k], '?'),
                        'pred_top1_tuple': list(r['t1']),
                        'pred_top1_surface': r['t1_surface'],
                        'joint_top5': r['joint'], 'text': txt})
            print(txt); print()
        json.dump(out, open(f'{OUT}/heldout_stream_prompts.json', 'w'),
                  ensure_ascii=False, indent=2)
        print(f'[over-range id trims={OVER["trimmed"]} dropped prompts={OVER["dropped"]}]')
        print(f'wrote {OUT}/heldout_stream_prompts.json')
        _summarise(out, 'heldout_stream')


def _summarise(out, tag):
    """Aggregate the per-prompt sample.  Small n -> reported as counts, not rates."""
    if not out:
        return
    n = len(out)

    def top1(o):
        return o['pred_top1_tuple']

    root1 = sum(1 for o in out if top1(o)[1] == o['true_tuple'][1])
    root5 = sum(1 for o in out if any(j['tuple'][1] == o['true_tuple'][1]
                                      for j in o.get('joint_top5', [])))
    jroot = sum(1 for o in out if any(j['tuple'][1] == o['true_tuple'][1]
                                      for j in o.get('joint_top5', [])))
    jtuple = sum(1 for o in out if any(tuple(j['tuple']) == tuple(o['true_tuple'])
                                       for j in o.get('joint_top5', [])))
    p1 = sum(1 for o in out if top1(o) == o['true_tuple'])
    tags = {}
    for o in out:
        rt = tag_of_root(top1(o)[1])
        tags[rt] = tags.get(rt, 0) + 1
    # error taxonomy on the root head: right root wrong pattern, wrong root, exact
    rw = sum(1 for o in out if top1(o)[1] == o['true_tuple'][1] and top1(o) != o['true_tuple'])
    print(f'\n[summary {tag}] n={n}')
    print(f'  model top-1 root correct      : {root1}/{n}')
    print(f'  true root anywhere in top-5   : {root5}/{n}')
    print(f'  true TUPLE anywhere in top-5  : {jtuple}/{n}')
    print(f'  model top-1 tuple exact       : {p1}/{n}')
    print(f'  right root, wrong pattern     : {rw}/{n}')
    print(f'  model top-1 root class        : {tags}')
    json.dump({'tag': tag, 'n': n, 'top1_root_correct': root1, 'true_root_in_top5': root5,
               'true_tuple_in_top5': jtuple, 'top1_exact': p1,
               'right_root_wrong_pattern': rw, 'top1_root_class_hist': tags},
              open(f'{OUT}/summary_{tag}.json', 'w'), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
