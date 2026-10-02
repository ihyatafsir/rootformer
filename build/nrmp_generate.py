#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmp_generate.py -- NRMP decoder governed by the classical grammar engines.

This is the decoder the release was supposed to have. Instead of ad-hoc constraints it
delegates to the project's own implementations of the classical algorithms:

  * Al-Khalil ibn Ahmad (Kitab al-'Ayn) -- SibawayhNRMPGovernance.khalil_forbidden_roots
      radical phonotactic compatibility (C1==C2, deep-guttural incompatibility, bare alif).
  * Sibawayh (Al-Kitab) -- Nazariyyat al-'Amil operator states:
      HARF_JARR / HARF_JAZM / HARF_NASB / INNA / KANA / FUTURE drive hard exclusion masks
      on the root, wazn, prefix and suffix heads (apply_*_exclusion_mask).
  * Ibn Malik (Lamiyyat al-Af'al) -- IbnMalikVerbTransmuter.predict_imperfect_wazn:
      the pharyngeal / assimilated / hollow decision tree picks the imperfect wazn.
  * Ibn Malik (Alfiyyah) -- IbnMalikPOSAutomaton:
      "كلامنا لفظ مفيد كاستقم واسم وفعل ثم حرف الكلم" -- no Harf->Harf, no Fi'l->Fi'l.
  * Ibn Mada' (Kitab al-Radd) -- IbnMadaRealismFilter for the English actuator.
  * Al-Basriyyun surface realization -- BasranSyntacticRealizer via
      FarahidianMorphemicVocab.decode_sentence(apply_syntax=True).

Usage:
  python nrmp_generate.py                                  # built-in prompts
  python nrmp_generate.py "العلم نور يضيء العقل"
  python nrmp_generate.py --checkpoint /workspace/nrmp_trained_final.safetensors --n 2
  python nrmp_generate.py --no-grammar                     # ablation: engines off
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models'))


# --------------------------------------------------------------------------------------------
# VISIBILITY (pillar 3): the decode guards below run once per candidate word / per finished
# sentence.  Each reports its FIRST failure on stderr and keeps its previous default (''), so a
# decode bug can no longer be published as a blank token without a trace.
# --------------------------------------------------------------------------------------------
_WARNED_SITES = set()


def _warn_once(site, message):
    if site not in _WARNED_SITES:
        _WARNED_SITES.add(site)
        print(f'[nrmp_generate] {message}', file=sys.stderr)

try:
    from sibawayh_governance_engine import SibawayhNRMPGovernance  # noqa: F401
except Exception as e:  # noqa: BLE001
    SibawayhNRMPGovernance = None
    print(f'[warn] SibawayhNRMPGovernance unavailable: {e}')

try:
    from andalusian_grammatical_algorithms import (  # noqa: F401
        IbnMalikVerbTransmuter, IbnMalikPOSAutomaton, IbnMadaRealismFilter)
except Exception as e:  # noqa: BLE001
    IbnMalikVerbTransmuter = IbnMalikPOSAutomaton = IbnMadaRealismFilter = None
    print(f'[warn] Andalusian algorithms unavailable: {e}')

DEFAULT_PROMPTS = [
    'العلم نور يضيء العقل ويهدي إلى الحق',
    'رأس الحكمة مخافة الله',
    'اليقين لا يزول بالشك',
    'إنما الأعمال بالنيات وإنما لكل امرئ ما نوى',
    'الواجب الوجود بذاته لا علة له وهو مبدأ كل وجود',
    'النظم ليس شيئا غير توخي معاني النحو فيما بين الكلم',
    'الأصل في الأشياء الإباحة حتى يقوم دليل على التحريم',
    'النفس الناطقة جوهر مجرد عن المادة يدرك الكليات',
]

_DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640]')
_ARABIC = re.compile(r'[\u0600-\u06FF]')
# genuine prefix morphemes (the rest of canonical_prefixes are particles living in the root slot)
TRUE_PREFIXES = {'<NONE>', 'ال', 'و', 'وال', 'ف', 'فال', 'ب', 'بال', 'ل', 'لل', 'ك', 'كال', 'س'}


def strip_diac(s):
    return _DIAC.sub('', s)


def repair_orthography(word):
    w = word
    for bad in ('أا', 'إا', 'آا', 'اا', 'أآ', 'إآ'):
        w = w.replace(bad, 'آ')
    return w.replace('ىا', 'ى')


def valid_surface(word, seen):
    if not word or '<' in word or '>' in word:
        return False
    if len(_ARABIC.findall(word)) < 2:
        return False
    if word.count('ة') > 1:
        return False
    return word not in seen


def _sample(logits, temperature, top_k, top_p, generator):
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    finite = int(torch.isfinite(logits).sum().item())
    if finite <= 0:
        return int(torch.argmax(logits).item())
    if top_k and top_k > 0:
        k = min(top_k, finite)
        thresh = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < thresh, torch.full_like(logits, -1e9), logits)
    if top_p and 0 < top_p < 1.0:
        srt, idx = torch.sort(logits, descending=True)
        probs = F.softmax(srt, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        srt = torch.where(cum - probs > top_p, torch.full_like(srt, -1e9), srt)
        logits = torch.full_like(logits, -1e9).scatter(0, idx, srt)
    probs = F.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs, 1, generator=generator).item())


def pos_category(vocab, gov, r_id, w_id):
    """Ibn Malik's four categories: 1 Ism, 2 Fi'l, 3 Harf, 4 Sifah."""
    root = vocab.id2root.get(r_id, '')
    if root.startswith('<P:') or root in ('<PARTICLE>', '<UNK>', '<PAD>'):
        return 3
    if gov is not None and w_id in gov.all_verbal_awzan_ids:
        return 2
    pattern = strip_diac(vocab.id2wazn.get(w_id, ''))
    if pattern in ('فعيل', 'فعول', 'فعال', 'فاعل', 'مفعول'):
        return 4
    return 1


@torch.no_grad()
def generate_nrmp(model, prompt, max_new_words=16, min_new_words=6,
                  temperature=0.85, top_k=48, top_p=0.92,
                  root_rep_penalty=1.6, no_repeat_root_window=3,
                  max_retries=12, use_grammar=True, seed=None):
    vocab = model.vocab
    dev = model.device
    dev = dev if isinstance(dev, torch.device) else torch.device(dev)
    g = torch.Generator(device=dev)
    g.manual_seed(seed if seed is not None else 1234)

    gov = getattr(model, 'sibawayh_gov', None) if use_grammar else None
    use_alim = use_grammar and IbnMalikPOSAutomaton is not None
    use_im = use_grammar and IbnMalikVerbTransmuter is not None

    encoded = vocab.encode_sentence(prompt) or [
        (vocab.NONE_PREFIX, vocab.BOS_ROOT, vocab.NONE_WAZN, vocab.NONE_SUFFIX)]
    cp = [t[0] for t in encoded]
    cr = [t[1] for t in encoded]
    cw = [t[2] for t in encoded]
    cs = [t[3] for t in encoded]

    true_pref = {vocab.prefix2id[p] for p in TRUE_PREFIXES if p in vocab.prefix2id}
    stats = Counter()
    gen_tuples, seen_words = [], set()
    prev_pos = 0
    n_forward = 0
    # a new prompt starts a new ʿāmil chain (inqiṭāʿ al-ʿamal, al-Kitab 1/421)
    if gov is not None:
        gov.reset_chain()
    t0 = time.perf_counter()

    for step in range(max_new_words):
        ip = torch.tensor([cp], dtype=torch.long, device=dev)
        ir = torch.tensor([cr], dtype=torch.long, device=dev)
        iw = torch.tensor([cw], dtype=torch.long, device=dev)
        isu = torch.tensor([cs], dtype=torch.long, device=dev)
        emb = model.morphemic_embed(ip, ir, iw, isu)
        out = model.backbone(inputs_embeds=emb)
        h = model.final_norm(out.last_hidden_state)[:, -1:, :]
        n_forward += 1

        prev_tuple = (cp[-1], cr[-1], cw[-1], cs[-1])
        op_state = gov.get_operator_state(prev_tuple) if gov is not None else 'NONE'

        base_r = model.nrmp_head.root_head(h)[0, 0, :].clone().float()
        if gov is not None:
            base_r = gov.apply_root_exclusion_mask(base_r, op_state, cr[-6:], step, min_new_words)
            stats[f'op_{op_state}'] += 1
        else:
            for bid in (vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.UNK_ROOT,
                        vocab.root2id.get('<PARTICLE>', -1)):
                if bid >= 0:
                    base_r[bid] = -1e9
            if step < min_new_words:
                base_r[vocab.EOS_ROOT] = -1e9
        for rid in cr[-no_repeat_root_window:]:
            base_r[rid] = -1e9
        for rid in cr[-12:]:
            base_r[rid] = base_r[rid] / root_rep_penalty if base_r[rid] > 0 else base_r[rid] * root_rep_penalty
        # Ibn Malik: Harf -> Harf is forbidden; pre-emptively mask particles
        if use_alim and prev_pos == 3 and gov is not None:
            for pid in gov.particle_root_ids:
                base_r[pid] = -1e9

        chosen = None
        for _attempt in range(max_retries):
            next_r = _sample(base_r, temperature, top_k, top_p, g)
            if next_r == vocab.EOS_ROOT and step >= min_new_words:
                stats['eos'] += 1
                return _finish(prompt, gen_tuples, vocab, t0, n_forward, stats)
            root_str = vocab.id2root.get(next_r, '')

            if root_str.startswith('<P:'):
                cand = (vocab.NONE_PREFIX, next_r, vocab.NONE_WAZN, vocab.NONE_SUFFIX)
            else:
                r_t = torch.tensor([[next_r]], dtype=torch.long, device=dev)
                e_root = model.morphemic_embed.root_embed(r_t)
                h_cond = model.nrmp_head.cond_proj(torch.cat([h, e_root], dim=-1))

                w_logits = model.nrmp_head.wazn_head(h_cond)[0, 0, :].clone().float()
                if gov is not None:
                    w_logits = gov.apply_wazn_exclusion_mask(w_logits, op_state, -1, is_radical_root=True)
                if use_im and op_state in ('HARF_JAZM', 'HARF_NASB', 'FUTURE'):
                    pred = IbnMalikVerbTransmuter.predict_imperfect_wazn(root_str)
                    wid = vocab.wazn2id.get(pred['wazn'])
                    if wid is not None:
                        w_logits[wid] += 2.5
                        stats['ibn_malik_wazn'] += 1
                next_w = _sample(w_logits, temperature, top_k, top_p, g)

                p_logits = model.nrmp_head.prefix_head(h_cond)[0, 0, :].clone().float()
                if gov is not None:
                    p_logits = gov.apply_prefix_exclusion_mask(p_logits, op_state)
                for i in range(p_logits.numel()):
                    if i not in true_pref:
                        p_logits[i] = -1e9
                next_p = _sample(p_logits, temperature, top_k, top_p, g)

                s_logits = model.nrmp_head.suffix_head(h_cond)[0, 0, :].clone().float()
                if gov is not None:
                    s_logits = gov.apply_suffix_exclusion_mask(s_logits, next_w)
                    if next_w in gov.all_verbal_awzan_ids:
                        stats['sibawayh_suffix_mask'] += 1
                next_s = _sample(s_logits, temperature, top_k, top_p, g)
                cand = (next_p, next_r, next_w, next_s)

            cat = pos_category(vocab, gov, cand[1], cand[2])
            if use_alim:
                # Ibn Malik, Alfiyyah, bāb al-ʿaṭf: «وعطفك الفعل على الفعل يصح» -- Fiʿl -> Fiʿl
                # is legal ONLY through a coordinator, which the morphological tuple carries
                # in its PREFIX slot (و/ف/وال/فال).
                coord = IbnMalikPOSAutomaton.has_coordinator_prefix(
                    vocab.id2prefix.get(cand[0], ''))
                if not IbnMalikPOSAutomaton.transition_allowed(prev_pos, cat, coord):
                    stats[f'pos_blocked_{prev_pos}->{cat}'] += 1
                    continue
            try:
                word = repair_orthography(vocab.decode_word(*cand))
            except Exception as exc:  # noqa: BLE001
                # VISIBILITY: '' is then rejected by valid_surface(), so a decode bug silently
                # removed this candidate from the search instead of reporting itself.  '' is
                # unchanged.  Rate-limited: this runs once per candidate word.
                _warn_once('decode_word', f'vocab.decode_word({cand!r}) raised: {exc!r} -- '
                                          f'the candidate is scored with an empty surface')
                word = ''
            if valid_surface(word, seen_words):
                chosen = (cand, word, cat)
                break
            stats['invalid_surface'] += 1
            base_r[next_r] = -1e9

        if chosen is None:
            stats['dropped'] += 1
            continue
        cand, word, cat = chosen
        seen_words.add(word)
        cp.append(cand[0]); cr.append(cand[1]); cw.append(cand[2]); cs.append(cand[3])
        gen_tuples.append(cand)
        prev_pos = cat

    return _finish(prompt, gen_tuples, vocab, t0, n_forward, stats)


def _finish(prompt, gen_tuples, vocab, t0, n_forward, stats):
    dt = (time.perf_counter() - t0) * 1000
    words = []
    for t in gen_tuples:
        try:
            words.append(repair_orthography(vocab.decode_word(*t)))
        except Exception as exc:  # noqa: BLE001
            # VISIBILITY: an empty string used to be appended to the DECODED SENTENCE in
            # silence, so a decode failure was published as a blank token.  '' is unchanged.
            _warn_once('finish_decode', f'vocab.decode_word({t!r}) raised while finishing the '
                                        f'sentence: {exc!r} -- a blank token is emitted')
            words.append('')
    try:
        surf = vocab.decode_sentence(gen_tuples, apply_syntax=True)
    except Exception as e:  # noqa: BLE001
        surf = f'<syntax-error {type(e).__name__}>'
    roots = [vocab.id2root.get(r, '?') for (_, r, _, _) in gen_tuples]
    n = max(len(roots), 1)
    cnt = Counter(roots)
    bigrams = [tuple(roots[i:i + 2]) for i in range(len(roots) - 1)]
    return {
        'prompt': prompt,
        'text': surf,
        'text_raw': ' '.join(w for w in words if w),
        'words': words,
        'num_words': len(gen_tuples),
        'latency_ms': dt,
        'ms_per_word': dt / n,
        'forward_passes': n_forward,
        'grammar_events': dict(stats),
        'distinct_roots': len(set(roots)),
        'max_root_repeat': max(cnt.values()) if cnt else 0,
        'stutter_rate': (max(cnt.values()) - 1) / n if cnt and max(cnt.values()) > 1 else 0.0,
        'max_root_bigram_repeat': max(Counter(bigrams).values()) if bigrams else 0,
        'literal_token_leak': sum(1 for w in words if not w or '<' in w or '>' in w),
        'tuples': [{'prefix': vocab.id2prefix.get(p, '?'), 'root': vocab.id2root.get(r, '?'),
                    'wazn': vocab.id2wazn.get(w, '?'), 'suffix': vocab.id2suffix.get(s, '?')}
                   for (p, r, w, s) in gen_tuples],
    }


def report(rec, verbose=True):
    print(f"\n  PROMPT : {rec['prompt']}")
    print(f"  NRMP   : {rec['text']}")
    print(f"  surface: {rec['text_raw']}")
    print(f"  words  : {rec['num_words']} | {rec['latency_ms']:.0f} ms "
          f"({rec['ms_per_word']:.1f} ms/word)")
    print(f"  health : distinct_roots={rec['distinct_roots']}/{rec['num_words']} "
          f"stutter={rec['stutter_rate']:.2f} leak={rec['literal_token_leak']}")
    if rec.get('grammar_events'):
        ev = {k: v for k, v in rec['grammar_events'].items() if not k.startswith('op_NONE')}
        if ev:
            print(f"  grammar: {ev}")
    if verbose:
        for d in rec['tuples']:
            print(f"      {d['prefix']:<7}{d['root']:<14} {d['wazn']:<12} {d['suffix']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prompt', nargs='*', default=None)
    ap.add_argument('--prompts-file', default=None)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--max-new-words', type=int, default=16)
    ap.add_argument('--min-new-words', type=int, default=6)
    ap.add_argument('--temperature', type=float, default=0.85)
    ap.add_argument('--top-k', type=int, default=48)
    ap.add_argument('--top-p', type=float, default=0.92)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--no-grammar', action='store_true', help='ablation: disable classical engines')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    from nrmp_run import load_engine
    model, vocab, device, _ = load_engine(ckpt=args.checkpoint)

    gov = getattr(model, 'sibawayh_gov', None)
    print(f'\n[engines] SibawayhNRMPGovernance: {"ACTIVE" if gov is not None else "OFF"}'
          + (f' (khalil_forbidden_roots={len(gov.khalil_forbidden_roots)}, '
             f'particle_roots={len(gov.particle_root_ids)}, '
             f'verbal_awzan={len(gov.all_verbal_awzan_ids)})' if gov is not None else ''))
    print(f'[engines] IbnMalikVerbTransmuter: {"ACTIVE" if IbnMalikVerbTransmuter else "OFF"} | '
          f'IbnMalikPOSAutomaton: {"ACTIVE" if IbnMalikPOSAutomaton else "OFF"} | '
          f'BasranSyntacticRealizer: '
          f'{"ACTIVE" if getattr(vocab, "syntactic_realizer", None) else "OFF"}')

    if args.prompts_file:
        prompts = [l.strip() for l in Path(args.prompts_file).read_text(encoding='utf-8').splitlines()
                   if l.strip() and not l.startswith('#')]
    elif args.prompt:
        prompts = args.prompt if len(args.prompt) > 1 else [' '.join(args.prompt)]
    else:
        prompts = DEFAULT_PROMPTS

    temp = 0.0 if args.greedy else args.temperature
    mode = 'ABLATION (engines off)' if args.no_grammar else 'CLASSICAL-GRAMMAR GOVERNED'
    print('\n' + '=' * 78)
    print(f"NRMP DECODER [{mode}] temp={temp} top_k={args.top_k} top_p={args.top_p}")
    print('=' * 78)

    recs = []
    for p in prompts:
        for i in range(args.n):
            rec = generate_nrmp(model, p, max_new_words=args.max_new_words,
                                min_new_words=args.min_new_words, temperature=temp,
                                top_k=args.top_k, top_p=args.top_p,
                                use_grammar=not args.no_grammar, seed=1234 + i)
            report(rec)
            recs.append(rec)

    good = [r for r in recs if r['literal_token_leak'] == 0 and r['max_root_bigram_repeat'] <= 2]
    print('\n' + '=' * 78)
    print(f"SUMMARY {len(good)}/{len(recs)} samples clean (no literal leak, no root-bigram loop)")
    if recs:
        print(f"  mean stutter {sum(r['stutter_rate'] for r in recs)/len(recs):.3f} | "
              f"mean distinct roots {sum(r['distinct_roots'] for r in recs)/len(recs):.1f} | "
              f"mean {sum(r['ms_per_word'] for r in recs)/len(recs):.1f} ms/word")
        agg = Counter()
        for r in recs:
            agg.update(r.get('grammar_events', {}))
        if agg:
            print(f"  grammar firings: {dict(agg)}")
    print('=' * 78)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(recs, ensure_ascii=False, indent=2))
        print(f'wrote {args.json_out}')


if __name__ == '__main__':
    main()
