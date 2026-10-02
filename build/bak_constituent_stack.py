#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
constituent_stack.py -- Sibawayh's 'amal, implemented as a stack, producing CASE and then ROLE.

The governing principle, as Sibawayh states it in al-Kitab:

    العامل يعمل فيما يليه        an operator works upon what follows it
    ولا يعمل عاملان في معمول واحد   and two operators do not work upon one ma'mul
    فإذا عمل العامل عمل فيما بعده    once an operator works, it works upon what comes after,
                                   حتى يقطع عمله                until its action is cut off

So: imposing a CASE is what 'amal IS. The role then follows from the case plus the word's own
morphology -- which is why the (prefix, root, wazn, suffix) split is the right substrate: the ROOT
gives the masdar-same-root test, the WAZN gives the derived-adjective test, and the operator gives
the case.

  role = case(from the governing 'amil)  x  signature(wazn, root-relation, closed lexicon)

Where the table under-determines the answer, Sibawayh and Ibn Malik resolve it by ta'wil -- and that
residual is left to the model rather than forced by a rule.

Usage:
  python constituent_stack.py --n 30000
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

from word_class import WordClassifier, bare, JARR, INNA_FAMILY, KANA_FAMILY, JAZM, NASB, ATF

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
AR = re.compile(r'[\u0600-\u06FF]')
MAFUL_MUTLAQ_MARK = re.compile(r'^(مَصْدَر|مَصْ|اِسْم مَصْ)')
ZARF_LEXICON = {'يوم', 'ليلة', 'سنة', 'شهر', 'ساعة', 'وقت', 'حين', 'زمن', 'أمس', 'غدا',
                'مكة', 'بغداد', 'مصر', 'الشام', 'العراق', 'أرض', 'بلد', 'دار', 'بيت'}


def wazn_role(wazn_name, has_is_fa_il=True):
    """Map a wazn to the morphological class it signals (Ibn Malik's sarf taxonomy)."""
    w = DIAC.sub('', wazn_name or '')
    if not w:
        return None
    w = w.split('_')[-1] if '_' in w else w
    pat = w
    # derived nouns -- the classes the man subat roles key off
    if pat.startswith('مَفْعُول') or pat.startswith('مُفْعَل') or pat.startswith('مُفْتَعَل') \
            or pat.startswith('مُسْتَفْعَل'):
        return 'ism_maf_ul'
    if pat.startswith('فَاعِل') or pat.startswith('مُفْعِل') or pat.startswith('مُتَفَعِّل') \
            or pat.startswith('مُتَفَاعِل') or pat.startswith('مُنْفَعِل') or pat.startswith('مُفْتَعِل') \
            or pat.startswith('مُسْتَفْعِل') or pat.startswith('مُفَعْلِل'):
        return 'ism_fa_il'
    if pat.startswith('مَفْعَل') or pat.startswith('مَفْعِل'):
        return 'ism_makan'
    if pat.startswith('فَعِيل') or pat.startswith('فَعُول') or pat.startswith('فَعْلَى'):
        return 'sifah_mushabbahah'
    if pat.startswith('فَعَّال'):
        return 'sighat_mubalaghah'
    if pat.startswith('أَفْعَل'):
        return 'af_al_tafdil'
    if pat.startswith('فَعْلَل') or pat.startswith('فِعْلِل') or pat.startswith('فُعْلُل'):
        return 'masdar_asl'          # quadriliteral masdar
    if pat.startswith(('تَفْعِيل', 'تَفْعِلَة', 'إِفْعَال', 'تَفَعُّل', 'تَفَاعُل', 'اِنْفِعَال',
                       'اِفْتِعَال', 'اِفْعِلَال', 'اِسْتِفْعَال', 'مُفَاعَلَة', 'فِعَال',
                       'فَعْل', 'فِعْل', 'فُعْل', 'فَعَل', 'فِعَل', 'فُعَل', 'فُعُول',
                       'فِعْلَان', 'فُعْلَان')):
        return 'masdar_asl'
    return None


# ---- the 'amil -> (case, role) requirements, in order (Sibawayh, al-Kitab) -------------------
GOVERNMENT = {
    'harf_jarr': [('jarr', 'majruz_bil_jarr')],
    'inna':      [('nasb', 'mubtada'), ('raf', 'khabar')],      # ism inna / khabar inna
    'kana':      [('raf', 'mubtada'), ('nasb', 'khabar')],      # ism kana / khabar kana
    'khalil_neg': [('raf', 'mubtada'), ('nasb', 'khabar')],     # ma / la (hijaziyya)
    'jazm':      [('jazm', 'fi_l_mudari')],
    'nasb_op':   [('nasb', 'fi_l_mudari')],
    'fi_l':      [('raf', 'fa_il'), ('nasb', 'maf_ul_bihi')],
    'mubtada':   [('raf', 'khabar')],
}

JUMLA_OPERATORS = JARR | KANA_FAMILY | INNA_FAMILY


class ConstituentStack:
    """Walks a sentence assigning case from the governing operator, then role from morphology."""

    def __init__(self, vocab, blueprint_path):
        self.vocab = vocab
        self.wc = WordClassifier(vocab, blueprint_path)
        bp = json.load(open(blueprint_path))
        self.t2i = bp['vocab_token_to_id']

    def cat(self, name):
        return self.t2i.get(f'<{name}>', self.t2i['<mubtada>'])

    def analyze(self, words):
        """Return a list of (category_id, case, role, reason) per word."""
        out = []
        stack = []          # active operators, each a dict(kind, need_index)
        pending_jarr = False
        prev_verb_root = None
        prev_was_tafdil = False

        for w in words:
            p, r, wz, s = self.vocab.encode_word(w)
            cid, reason = self.wc.classify(w, p, r, wz, s, self.vocab)
            b = bare(w)
            wazn_name = self.vocab.id2wazn.get(wz, '')
            kind = reason.split(':')[0]

            # ---------- cut-off: inqita' al-'amal --------------------------------------
            # an exhausted operator stops working; a new jumla opens a new mubtada'
            stack = [o for o in stack if o['i'] < len(GOVERNMENT[o['kind']])]

            if kind == 'particle':
                which = reason.split(':', 1)[1] if ':' in reason else ''
                if which == 'harf_jarr':
                    # governs ONLY the next nominal; it does not linger on the stack
                    pending_jarr = True
                    out.append((cid, '', 'harf_jarr', 'operator:jarr (immediate)'))
                elif which == 'inna':
                    stack.insert(0, {'kind': 'inna', 'i': 0})
                    out.append((cid, '', 'inna', 'operator:inna'))
                elif which == 'kana':
                    stack.insert(0, {'kind': 'kana', 'i': 0})
                    out.append((cid, '', 'kana', 'operator:kana'))
                elif which == 'shart_in':
                    stack.insert(0, {'kind': 'jazm', 'i': 0})
                    out.append((cid, '', 'shart_in', 'operator:shart->jazm'))
                elif which in ('atf_waw',):
                    out.append((cid, '', 'atf_waw', 'coordination (no case change)'))
                else:
                    out.append((cid, '', which or 'particle', f'particle:{which}'))
                continue

            # ---------- verb: opens the fa'il / maf'ul requirement ---------------------
            if kind == 'verb':
                role = 'fi_l_' + reason.split(':')[1]
                # FIDELITY: the perfect (madi) and the imperative (amr) are MABNI -- they carry
                # no case at all. Only the imperfect (mudari') is mu'rab, taking raf' / nasb /
                # jazm from its operator. Marking madi as raf' inflated raf' to 65% of positions.
                if role == 'fi_l_mudari':
                    if stack and stack[0]['kind'] in ('jazm', 'nasb_op'):
                        case = stack[0]['kind']
                        stack[0]['i'] += 1
                    else:
                        case = 'raf'
                    why = f"{reason} (mu'rab, case={case})"
                else:
                    case = ''
                    why = f'{reason} (mabni -- carries no case)'
                prev_verb_root = r
                stack.insert(0, {'kind': 'fi_l', 'i': 0})
                out.append((cid, case, role, why))
                continue

            # ---------- noun: take the case from the top operator ----------------------
            if pending_jarr:
                # the jarr operator claims this position outright, then expires
                pending_jarr = False
                case, base_role = 'jarr', 'majruz_bil_jarr'
                src = 'case=jarr from harf_jarr'
            elif stack:
                top = stack[0]
                reqs = GOVERNMENT[top['kind']]
                case, base_role = reqs[min(top['i'], len(reqs) - 1)]
                top['i'] += 1
                # a verb stays open after its core requirements for zarf / hal / 2nd maf'ul
                if top['i'] >= len(reqs) and top['kind'] != 'fi_l':
                    stack = stack[1:]
                src = f"case={case} from {top['kind']}"
            else:
                case, base_role = 'raf', 'mubtada'
                stack.insert(0, {'kind': 'mubtada', 'i': 0})
                src = 'no operator -> mubtada (raf)'

            # ---- role refinement: case x morphology (the part the operator cannot give) ----
            morph = wazn_role(wazn_name)
            role = base_role
            why = src
            if case == 'nasb':
                if morph == 'masdar_asl' and prev_verb_root is not None and r == prev_verb_root:
                    role, why = 'masdar_asl', f'{src} + masdar of the SAME root -> maf\'ul mutlaq'
                elif morph in ('ism_fa_il', 'ism_maf_ul', 'sifah_mushabbahah'):
                    role, why = 'hal', f'{src} + derived adjective ({morph}) -> hal'
                elif prev_was_tafdil:
                    role, why = 'tamyiz', f'{src} + after af\'al al-tafdil -> tamyiz'
                elif b in ZARF_LEXICON:
                    role, why = 'mudaf_ilayhi', f'{src} + time/place noun -> zarf'
            elif morph:
                role = morph
                why = f'{src} + wazn class {morph}'
            prev_was_tafdil = (morph == 'af_al_tafdil')
            out.append((self.cat(role), case, role, why))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30000)
    ap.add_argument('--out', default='/workspace/constituent_stack.json')
    args = ap.parse_args()

    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    cs = ConstituentStack(vocab, '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')

    files = (sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))
             + sorted(glob.glob('/workspace/andalusian_canon_sanitized/*.txt')))[:6]
    sents = []
    for f in files:
        txt = open(f, encoding='utf-8', errors='ignore').read()
        for s in re.split(r'[\n.!?؟]+', txt):
            ws = [x for x in s.split() if AR.search(x)]
            if 3 <= len(ws) <= 60:
                sents.append(ws)
            if sum(len(x) for x in sents) > args.n:
                break
        if sum(len(x) for x in sents) > args.n:
            break

    roles, cases, whys = Counter(), Counter(), Counter()
    total = 0
    for ws in sents:
        for cid, case, role, why in cs.analyze(ws):
            roles[role] += 1
            cases[case or '-'] += 1
            whys[why.split(' + ')[0][:34]] += 1
            total += 1

    print(f'[*] analysed {total} positions in {len(sents)} sentences\n')
    print('CASE distribution:')
    for k, v in cases.most_common():
        print(f'   {k:<10}{v:>8}{100*v/total:>8.1f}%')
    print('\nROLE distribution (top 18):')
    for k, v in roles.most_common(18):
        print(f'   {k:<22}{v:>8}{100*v/total:>8.1f}%')
    unresolved = roles.get('mubtada', 0)
    print(f'\n[*] roles with a MORPHOLOGICAL signature: '
          f'{total-unresolved}/{total} = {100*(total-unresolved)/total:.1f}%')
    json.dump({'total': total, 'cases': dict(cases), 'roles': dict(roles)},
              open(args.out, 'w'), indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
