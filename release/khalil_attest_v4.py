#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
khalil_attest_v4.py -- the complete al-'Ayn attestation record (مستعمل / مهمل).

v3 held 2,483 attested and 106 unused permutations.  This holds 4,381 and 2,750.  Five defects were
responsible, and each one is a *constraint that was not read* rather than a coding slip:

1. THE MARK VOCABULARY WAS TOO NARROW.  al-'Ayn states attestation with several verbs, and v3 only
   recognised the م-forms.  Measured in the text:

       مستعملات 670 | مستعملان 228 | مستعمل فقط 175 | مستعمل 196
       يستعمل   226 | يستعملان 155 | يستعمل فقط 183

   v3's pattern `(مستعملات|مستعمل فقط|مستعملان|مستعمل|مهملا|مهمل)` cannot match `يستعمل فقط`
   -- ي-س-ت-ع-م-ل does not contain م-س-ت-ع-م-ل -- so 338 chapters that state their attestation
   with يستعمل were dropped entirely.

2. THE MARK DOES NOT ALWAYS FOLLOW THE RUN.  Besides "<run> مستعملات", al-'Ayn writes
   «باب العين والطاء والفاء معهما يستعمل ع ط ف- ع ف ط فقط» -- verdict word first, فقط last.  Read
   as run-then-mark, the first permutation is fused to يستعمل (ي س ت ع م ل ع ط ف) and lost, and the
   chapter then *condemns* a root it had just entered.

3. THE LETTER NAMES WERE READ AS THEIR FIRST LETTER.  al-'Ayn names its letters (العين، اللام،
   الهمزة).  For every name but one the first letter IS the letter; for الهمزة it is not -- the
   letter is ء, not ه.  v3 declared {ه,...} for every hamza chapter, so no permutation of a mahmuz
   root could match its own chapter.  Fixed with an explicit name table, plus canonicalisation to
   bare alif on output, because the shipped lexicon writes the hamza radical as ا (افل not أفل).

4. مهمل WAS ONLY READ WHERE IT WAS WRITTEN.  The commonest construction is the terse one:

       باب التاء والنون   ت ن   يستعمل فقط   تن: ...

   Only the *used* permutation is written and the rest are muhmal by exclusion -- "فقط" asserts the
   listed permutation is the only one.  v3 therefore had no 2-letter unused permutation at all,
   which is exactly why نتت could not be condemned: the pair {ن،ت} is marked only in the 2-letter
   chapter, whose complement v3 never derived.

5. WHOLE CHAPTER FAMILIES WERE INVISIBLE.  Two heading forms carry no verdict at all:
      * the UNMARKED enumeration -- «باب العين والدال والراء معهما ع د ر- ع ر د- ... ر د ع عدر:»,
        where all six permutations (عدر عرد دعر رعد درع ردع) are roots of the language;
      * the WEAK-RADICAL paren -- «باب التاء والنون و (وء ي) معهما ت ي ن، ي ت ن، وت ن، ن تء، ء ت ن
        مستعملات», which declares one 3-letter set per weak letter.  Read as a 2-letter chapter,
        every permutation mismatches and the pair even collects a phantom attestation.  236
        chapters use this form.

THE COMPLEMENT RULE, AND WHERE IT IS SAFE
-----------------------------------------
The complement of a chapter's listed permutations is مuhmal, but only where al-'Ayn asserts it: the
window contains فقط, or the chapter already carries an explicit مهمل group.  Otherwise unlisted
permutations are merely *undiscussed* and are not condemned -- the same distinction the fidelity
ledger draws for the phonotactic rules, where "لا يأتلف" (impossible) is not "not attested".

A chapter's ENTRIES outrank its own terse verdict.  «باب القاف والذال واللام معهما ق ذ ل، ل ذ ق
يستعملان فقط» ("only these two") then enters BOTH قذل and ذلق, ذلق being neither of the two listed.
Every lemma of the form <word>: in the chapter body is therefore read as attested.  This can only
ADD attestation; it can never condemn, so it cannot make the negative record unsafe.

KEYING
------
The record is keyed by the ORDERED sequence, because al-Khalīl marks permutations individually: in
باب العين والدال واللام, ع د ل / ع ل د / د ل ع are مستعمل while د ع ل / ل ع د / ل د ع are مهمل.
Order is the whole point of التقليب.

A repeated-radical root is NOT looked up on its distinct skeleton.  That was tried and is wrong:
نتت and نتن both reduce to نت, the pair {ت،ن} chapter says «ت ن يستعمل فقط», yet al-'Ayn enters نتن
in that very chapter, and the same move condemned the real quadriliterals قنقل، قرقل، لغلغ.  Where a
chapter does enter such a root, the headword pass records it under its exact sequence; otherwise the
verdict is "unknown", which is honest.

Usage: python khalil_attest_v4.py [path-to-Al_Khalil_Al_Ayn.txt] [out.json]
"""
import itertools
import json
import re
import sys
from collections import Counter, defaultdict

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
LET = r'[\u0621-\u064a]'

# al-'Ayn names each letter; take the NAME, not its first letter
LETTER_NAMES = {
    'الف': 'ا', 'ألف': 'ا', 'همزه': 'ء', 'همزة': 'ء',
    'باء': 'ب', 'با': 'ب', 'تاء': 'ت', 'تا': 'ت', 'ثاء': 'ث', 'ثا': 'ث',
    'جيم': 'ج', 'حاء': 'ح', 'حا': 'ح', 'خاء': 'خ', 'خا': 'خ',
    'دال': 'د', 'ذال': 'ذ', 'راء': 'ر', 'را': 'ر', 'زاي': 'ز', 'زا': 'ز', 'زاء': 'ز',
    'سين': 'س', 'شين': 'ش', 'صاد': 'ص', 'ضاد': 'ض', 'طاء': 'ط', 'طا': 'ط',
    'ظاء': 'ظ', 'ظا': 'ظ', 'عين': 'ع', 'غين': 'غ', 'فاء': 'ف', 'فا': 'ف',
    'قاف': 'ق', 'كاف': 'ك', 'لام': 'ل', 'ميم': 'م', 'نون': 'ن', 'هاء': 'ه', 'ها': 'ه',
    'واو': 'و', 'ياء': 'ي', 'يا': 'ي',
}

# every way al-'Ayn states the two verdicts; longest alternative first
MARK = re.compile(
    r'(ومستعملات|أمستعملات|امستعملات|مستعملات|مستعملان|مستعمل|مهملان|مهملات|مهملا|مهمل|'
    r'يستعملان|يستعمل|تستعمل)'
)

HEAD = re.compile(r'باب\s+((?:(?:وال|ال)' + LET + r'+)(?:\s+(?:وال|ال)' + LET + r'+){1,4})')


def norm(s):
    s = DIAC.sub('', s)
    for a, b in (('أ', 'ا'), ('إ', 'ا'), ('آ', 'ا'), ('ٱ', 'ا'), ('ى', 'ي'), ('ة', 'ه'),
                 ('ؤ', 'و'), ('ئ', 'ي')):
        s = s.replace(a, b)
    return s


def heading_letters(h):
    """The letter set the chapter declares, by NAME (so الهمزة -> ء, not ه)."""
    out, unknown = [], []
    for name in re.findall(r'(?:وال|ال)(' + LET + r'+)', h):
        got = LETTER_NAMES.get(name)
        if got is None:
            unknown.append(name)
            got = name[0]
        out.append(got)
    return out, unknown


def permutations(letters):
    return [list(p) for p in itertools.permutations(letters)]


def root_key(root):
    """The chapter key a root is looked up under.

    al-'Ayn declares chapters over sets of DISTINCT letters, so a root that repeats a radical is
    filed under its ordered distinct skeleton:

        ك ت ت (muda''af, C2==C3)  ->  (ك, ت)   -- the 2-letter chapter باب الكاف والتاء
        ث ل ث (C1==C3)           ->  (ث, ل)   -- the 2-letter chapter باب الثاء واللام
        ت ح ت (C1==C3)           ->  (ت, ح)

    Reading only C2==C3 (as the first version did) loses every C1==C3 root -- ثلث، تحت are real
    roots and were being reported as having no chapter at all.
    """
    return tuple(dict.fromkeys(root))


# al-'Ayn's FOURTH construction: a chapter for a letter pair that also accepts one of the weak
# letters as its third radical --
#
#     باب التاء والنون و (وء ي) معهما ت ي ن، ي ت ن، وت ن، ن تء، ء ت ن   مستعملات
#
# The parenthetical is the set of weak letters the chapter admits (و ا ي ء).  239 chapters use this
# form.  Read as a 2-letter chapter (the naive parse) every one of its permutations mismatches, and
# the pair {ت،ن} even collects a phantom attestation.  Alif is listed by convention but is never a
# radical in Arabic, so it contributes no member.
WEAK_GROUP = re.compile(r'\s*(?:و\s*)?\(([\u0621-\u064a\s]{1,10})\)')


def weak_letters(tail):
    m = WEAK_GROUP.match(tail)
    if not m:
        return []
    body = m.group(1)
    if any(ch not in 'وءياأإآ ' for ch in body):
        return []
    out = []
    for ch in ('و', 'ي', 'ء'):
        if ch in body:
            out.append(ch)
    return out


def parse(text):
    attested, unused, ambiguous = [], [], []
    stats = Counter()
    unknown_names = Counter()

    for m in HEAD.finditer(text):
        letters, unknown = heading_letters(m.group(1))
        if unknown:
            # a heading like باب الثلاثي الصحيح / باب المعتل / باب المضاعف names a SECTION, not
            # a letter set.  Falling back to the token's first letter would fabricate a chapter
            # (ثلاثي+صحيح -> {ث،ص}) and its complement would then condemn innocent permutations.
            # An unrecognised name means "not a letter chapter" -- skip it.
            unknown_names.update(unknown)
            stats['reject_nonletter'] += 1
            continue
        if not (2 <= len(letters) <= 4):
            stats['reject_len'] += 1
            continue
        if len(set(letters)) != len(letters):
            stats['reject_dup'] += 1
            continue

        weaks = weak_letters(text[m.end():m.end() + 26])
        if weaks:
            declared = [tuple(letters) + (w,) for w in weaks]
            stats['weak_chapters'] += 1
        else:
            declared = [tuple(letters)]

        # the enumeration lives between the heading and the first entry (the ':'),
        # and cannot run past the next chapter
        end = m.end()
        stop = end + 220
        for j in range(end, min(stop, len(text))):
            if text[j] == ':':
                stop = j
                break
        nxt = text.find('باب ', end)
        if 0 <= nxt < stop:
            stop = nxt
        window = text[end:stop]

        # THE CHAPTER'S OWN ENTRIES ARE GROUND TRUTH.  The terse "يستعمل فقط" chapters can
        # contradict their own headwords -- باب القاف والذال واللام says «ق ذ ل، ل ذ ق يستعملان
        # فقط» ("only these two") and then enters BOTH قذل and ذلق, ذلق not being among the two
        # listed.  Whatever a chapter enters is a root of the language, so every lemma of the form
        # <word>: in the body is read as attested.  This can only ADD attestation; it can never
        # condemn, so it cannot make the negative record unsafe.
        body_end = text.find('باب ', stop)
        body = text[stop:body_end if body_end > 0 else stop + 4000]
        headwords = []
        for hm in re.finditer(r'([\u0621-\u064a]{2,6})\s*:', body):
            hw = hm.group(1)
            for want in declared:
                if set(hw) != set(want):
                    continue
                # an exact permutation of the declared letters, or that permutation with ONE
                # radical repeated -- which is how al-'Ayn enters a muda''af/lafif root in the
                # chapter of its distinct letters (نتن in باب التاء والنون, حرح in باب الحاء).
                if len(hw) == len(want) or (
                        len(hw) == len(want) + 1 and
                        any(hw.count(c) == 2 for c in want)):
                    headwords.append(hw)
                    break

        marks = list(MARK.finditer(window))

        def chunks_of(run):
            """The permutations inside one enumeration run, as ordered letter lists.

            The run writes single letters separated by spaces (ع د ر) or glued (دلع), and the FIRST
            HEADWORD of the chapter is appended to the last chunk with no separator
            (... ر د ع عدر:).  So a chunk may carry more letters than the root has; in that case
            the leading n letters are the permutation, exactly as al-Khalil wrote them.
            """
            for conn in ('معهما', 'معهن', 'معها', 'معهم', 'معه'):
                run = run.replace(conn, ' ')
            # The order is not fixed.  Besides the usual "<run> مستعملات", al-'Ayn also writes
            # "معهما يستعمل ع ط ف- ع ف ط فقط" -- the verdict word FIRST, "فقط" last, so the first
            # permutation is fused to يستعمل and would be read as ي س ت ع م ل ع ط ف.
            run = re.sub(r'(?:و|أ|ا)?(?:يستعمل|مستعمل|تستعمل)[\u0621-\u064a]*', ' ', run)
            run = run.replace('فقط', ' ')
            # the paren that DECLARED the weak letters also sits inside the run, and the conjunction
            # introducing it ("و (وا يء) معهما ك ون") would be read as a letter of the first
            # permutation (و ك و ن).  Remove the whole و (...) construct.
            run = re.sub(r'\s*و\s*\([^)]*\)', ' ', run)
            run = re.sub(r'\([^)]*\)', ' ', run)
            for chunk in re.split(r'[-،,]', run):
                got = re.findall(LET, chunk)
                for want in declared:
                    n = len(want)
                    if len(got) == n and sorted(got) == sorted(want):
                        yield got
                        break
                    if len(got) > n and sorted(got[:n]) == sorted(want):
                        yield got[:n]
                        break

        listed_a, listed_u, pos = [], [], 0
        has_muhmal_group = any('مهمل' in mk.group(1) for mk in marks)
        if not marks or not has_muhmal_group:
            # Either the chapter marks nothing, or it states its verdict with فقط ("only").  Both
            # put EVERY listed permutation on the attested side, and the order of the verdict word
            # relative to the run does not matter.  The complement then follows from فقط alone.
            listed_a = list(chunks_of(window))
            if not listed_a and not headwords:
                stats['reject_nomark'] += 1
                continue
            if not marks:
                stats['unmarked_chapters'] += 1
        else:
            # explicit groups: each run is labelled by the mark that follows it
            for mk in marks:
                run = window[pos:mk.start()]
                pos = mk.end()
                got_any = False
                for got in chunks_of(run):
                    (listed_u if 'مهمل' in mk.group(1) else listed_a).append(got)
                    got_any = True
                if got_any:
                    stats['groups'] += 1

        if not listed_a and not listed_u and not headwords:
            stats['reject_norun'] += 1
            continue

        stats['chapters'] += 1
        stats['headwords'] += len(headwords)
        attested.extend(listed_a)
        attested.extend([list(hw) for hw in headwords])
        unused.extend(listed_u)

        # the complement -- only where al-'Ayn asserts it.  Derived per declared set, because a
        # paren chapter covers one set per weak letter.  Headwords are excluded: a lemma the
        # chapter actually enters is never مuhmal, whatever a terse «فقط» appears to imply.
        if 'فقط' in window or listed_u:
            seen_a = {tuple(x) for x in listed_a} | {tuple(hw) for hw in headwords}
            seen_u = {tuple(x) for x in listed_u}
            for want in declared:
                for p in permutations(list(want)):
                    if tuple(p) not in seen_a and tuple(p) not in seen_u:
                        unused.append(p)
            if 'فقط' in window:
                stats['complement_faqat'] += 1
        else:
            for want in declared:
                for p in permutations(list(want)):
                    if tuple(p) not in {tuple(x) for x in listed_a}:
                        ambiguous.append(p)

    def uniq(xs):
        seen, out = set(), []
        for x in xs:
            k = tuple(x)
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    def canon(x):
        """al-'Ayn writes the hamza radical as الهمزة (ء); the shipped lexicon normalises it to bare
        alif (ا) -- ابق/أبق, اصل/أصل.  (Alif is never a genuine radical, so the two are the same
        letter here.)  Without this, every mahmuz root misses its own chapter: افل، اتن، اسم."""
        return tuple('ا' if c == 'ء' else c for c in x)

    attested = uniq([canon(x) for x in attested])
    unused = uniq([canon(x) for x in unused])
    ambiguous = uniq([canon(x) for x in ambiguous])
    # PRECEDENCE.  A permutation must not be both: al-'Ayn sometimes attests X in one chapter and
    # a terse "يستعمل فقط" elsewhere implies it.  An explicit statement beats an inference, and any
    # conflict is reported rather than silently resolved.
    A, U = {tuple(x) for x in attested}, {tuple(x) for x in unused}
    both = A & U
    if both:
        stats['conflicts'] = len(both)
        stats['conflict_examples'] = [' '.join(x) for x in sorted(both)[:12]]
        unused = [x for x in unused if tuple(x) not in A]
    return attested, unused, ambiguous, stats, unknown_names


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '../corpus/basran/Al_Khalil_Al_Ayn.txt'
    out = sys.argv[2] if len(sys.argv) > 2 else '../results/khalil_attest_v4.json'
    raw = open(path, encoding='utf-8', errors='ignore').read()
    text = re.sub(r'\s+', ' ', norm(raw))

    attested, unused, ambiguous, stats, unknown = parse(text)
    goodset = {tuple(x) for x in attested}
    badset = {tuple(x) for x in unused}

    print("=== al-'Ayn attestation, complete ===")
    print(f"  chapters with a usable enumeration : {stats['chapters']}"
          f"   (of which paren/weak-radical chapters: {stats['weak_chapters']},"
          f" unmarked enumerations: {stats['unmarked_chapters']})")
    print(f"  mark groups                        : {stats['groups']}")
    print(f"  complement derived (فقط chapters)  : {stats['complement_faqat']}")
    print(f"  attestation/مهمل conflicts resolved: {stats.get('conflicts', 0)}")
    if stats.get('conflict_examples'):
        print(f"    e.g. {stats['conflict_examples']}")
    print(f"  attested permutations              : {len(attested)}"
          f"  {dict(Counter(len(x) for x in attested))}")
    print(f"  unused   (muhmal) permutations     : {len(unused)}"
          f"  {dict(Counter(len(x) for x in unused))}")
    print(f"  undiscussed (neither)              : {len(ambiguous)}")
    if unknown:
        print(f"  unrecognised letter names           : {unknown.most_common(8)}")

    print("\n  SANITY -- roots the earlier parses could not see:")
    for probe in (['ت', 'ن'], ['ن', 'ت'], ['ك', 'ت', 'ب'], ['ض', 'ر', 'ب'],
                  ['ع', 'د', 'ل'], ['ع', 'ل', 'م'], ['ق', 'و', 'ل'], ['ح', 'ق', 'ق'],
                  ['ق', 'ن', 'و'], ['ع', 'ن', 'و'], ['ك', 'و', 'ن']):
        p = tuple(probe)
        print(f"    {''.join(probe):<6} attested={p in goodset}  muhmal={p in badset}")

    # -- measure the record against the shipped 9,013-root lexicon -------------------------
    try:
        bp = json.load(open('../results/rootformer_v12_blueprint_137awzan.json'))
        i2t = bp['vocab_id_to_token']
        lo, hi = bp['partitions']['classical_roots']
        lexicon = [re.sub(r'^<root_|>$', '', i2t[str(i)]) for i in range(lo, hi)]
    except Exception as e:                                   # pragma: no cover
        print(f"\n  (lexicon not loaded: {e})")
        lexicon = []

    if lexicon:
        def verdict(root):
            r = ''.join(root)
            # EXACT lookup.  The lexicon writes a root as its own letter sequence, so that is what
            # is looked up.  Reducing a repeated-radical root to its distinct skeleton (نتت and
            # نتن both -> نت) was tried and is WRONG: al-'Ayn enters نتن in باب التاء والنون even
            # though that chapter says «ت ن يستعمل فقط», and the same move condemned the real
            # quadriliterals قنقل، قرقل، لغلغ.  Where a chapter does enter a muda''af root, the
            # headword pass above records it under its exact sequence.
            key = tuple(r)
            if key in badset:
                return 'muhmal'
            if key in goodset:
                return 'attested'
            return 'unknown'

        c = Counter(verdict(r) for r in lexicon)
        n = len(lexicon)
        print(f"\n=== against the {n}-root lexicon ===")
        for k in ('attested', 'muhmal', 'unknown'):
            print(f"  {k:<9}: {c[k]:>5}  ({100.0*c[k]/n:.1f}%)")
        print("\n  the muhmal roots (al-Khalil condemns these):")
        bad = [r for r in lexicon if verdict(r) == 'muhmal']
        print('   ', ' '.join(bad[:60]), '...' if len(bad) > 60 else '')
        for probe in ('نتت', 'كتب', 'ضرب', 'بلل', 'كمم', 'عنن'):
            print(f"    {probe}: {verdict(probe) if probe in set(lexicon) else 'not in lexicon'}")

    json.dump({'attested': attested, 'unused': unused, 'ambiguous': ambiguous,
               'counts': {'chapters': stats['chapters'],
                          'attested': len(attested), 'unused': len(unused)}},
              open(out, 'w'), ensure_ascii=False, indent=1)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
