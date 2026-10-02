# al-ʿAyn attestation record, v4 — what was wrong and what it now supports

Built by `khalil_attest_v4.py` from `corpus/basran/Al_Khalil_Al_Ayn.txt`. Output:
`results/khalil_attest_v4.json`.

## The record

| | v3 | v4 |
|---|---|---|
| attested permutations | 2,483 | **4,381** (2, 3 and 4 letters) |
| unused (مهمل) permutations | 106 | **2,750** |
| 2-letter pairs marked مهمل | **0** | **41** |
| chapters with a usable enumeration | 1,183 | **1,428** |

Against the shipped 9,013-root lexicon: **3,675 attested (40.8 %), 425 condemned (4.7 %),
4,913 unknown (54.5 %)**.

## Five defects in v3

Each is a constraint that was not *read*, not a coding slip.

1. **The mark vocabulary was too narrow.** al-ʿAyn states its verdict with several verbs
   (`مستعملات` 670, `مستعملان` 228, `يستعمل` 226, `يستعمل فقط` 183, `يستعملان` 155 …). v3's pattern
   `(مستعملات|مستعمل فقط|مستعملان|مستعمل|مهملا|مهمل)` cannot match `يستعمل فقط` — ي-س-ت-ع-م-ل does
   not contain م-س-ت-ع-م-ل — so **338 chapters were dropped**.

2. **The mark does not always follow the run.** Besides `‹run› مستعملات`, al-ʿAyn writes
   «باب العين والطاء والفاء معهما **يستعمل** ع ط ف- ع ف ط **فقط**» — verdict first, `فقط` last. Read
   as run-then-mark, `عطف` fuses to `يستعمل` and the chapter then *condemns a root it had just
   entered*.

3. **The letter names were read as their first letter.** Every name but one works that way
   (العين → ع). **الهمزة → ء, not ه.** So every hamza chapter declared the wrong letter set and no
   mahmuz root could match its own chapter. Also canonicalised to bare alif on output, because the
   lexicon writes `افل`, not `أفل`.

4. **مهمل was only read where it was written.** The commonest construction is terse and positional:

       باب التاء والنون   ت ن   يستعمل فقط   تن:

   Only the *used* permutation is written; the rest are مuhmal **by exclusion**. v3 derived no
   complement at all, so it had **no 2-letter unused pair** — which is exactly why `نتت` could not
   be condemned. The complement is now derived wherever `فقط` appears.

5. **Two whole heading families were invisible (236 + 38 chapters).**
   * **Unmarked enumerations** — «باب العين والدال والراء معهما ع د ر- ع ر د- … ر د ع عدر:». No
     mark means every listed permutation is used (all six are roots: عدر عرد دعر رعد درع ردع).
   * **Weak-radical parens** — «باب التاء والنون و (وء ي) معهما ت ي ن، ي ت ن، وت ن، ن تء، ء ت ن
     مستعملات», which declares one 3-letter set per weak letter. Read as a 2-letter chapter, every
     permutation mismatches *and the pair collects a phantom attestation*.

## Two guards that keep the negative record safe

**The complement is derived only where asserted** — `فقط` present, or an explicit مهمل group. Unlisted
permutations in a chapter that says nothing are *undiscussed*, not forbidden. This is the same line
the fidelity ledger draws for the phonotactic rules: «لا يأتلف» is not "not attested".

**A chapter's entries outrank its own terse verdict.** «باب القاف والذال واللام معهما ق ذ ل، ل ذ ق
يستعملان فقط» ("only these two") then enters **both** قذل and ذلق — and `ذلق` is neither of the two
listed. Every `<word>:` lemma in the chapter body is therefore read as attested. This can only *add*
attestation, never condemn.

## What was tried and rejected

**Skeleton lookup for repeated-radical roots.** `نتت` and `نتن` both reduce to the pair `نت`, and the
pair chapter says «ت ن يستعمل فقط». That rule condemned `نتت` — and also `نتن`, which al-ʿAyn *enters
in that very chapter* («نتن ينتن نتنا … وهذه المادة من الثلاثي») — plus the real quadriliterals
`قنقل`, `قرقل`, `لغلغ`. Rejected; repeated-radical roots resolve by exact sequence or stay *unknown*.

## Independent validation

Scored each lexicon root by how often it appears as a whole word in four other dictionaries held
locally (Sībawayh-era and later: Sihah, Tahdhib, Maqayis, Jamharat):

| verdict | n | median occurrences elsewhere |
|---|---|---|
| attested | 3,675 | **23** |
| condemned | 425 | **3** |

The condemned list is dominated by surface forms the lexicon wrongly stores as roots — `ذين` (761),
`تلك` (364), `كفه`, `ضال`, `خلت`, `فات`, `شقه`, `دعت`, `كلت` — which are inflected words, not roots.
Known false condemnations found and fixed during this pass: `ذلق` (55), `عطف` (140), `نتن`, `قنقل`,
`لغلغ`, `ملهم`. One residual doubt is recorded: `حثر` (29) is condemned on al-ʿAyn's explicit
«باب الحاء والثاء والراء … ح ر ث يستعمل فقط», though later dictionaries list it.

## The result that matters, and it is negative

Swapping the record into the shipped analyzer changed **nothing**. `analyzer_segmentation_v3.py` and
`analyzer_segmentation_v4.py` produce byte-identical output on the nine-word probe set, because
`_real_root()` returns `True` on **mere lexicon membership** before it ever consults al-ʿAyn:

```python
if rt in self.unused: return False
if r in self.lexicon:  return True     # <-- decides everything
if rt in self.attested: return True
```

So a 26× larger, independently validated negative record was invisible to the pipeline. That — not
the completeness of the extraction — was the real blocker for `كنت`.

## Step 2 — the mudaʿʿaf pair gate

`analyzer_segmentation_v5.py` adds one rule: a doubled root `C1 C2 C2` *is* the doubling of the
ordered pair `(C1, C2)`, and al-ʿAyn's two-letter chapters enumerate exactly which pairs are used.
`نتت` is the tokenizer's doubling of the stem `نت`; the pair `نت` is مهمل; so `نتت` is not a root of
the language — although the blueprint lexicon lists it. Applied to `C2==C3` **only**, which is what
keeps `نتن` (entered by al-ʿAyn in that very chapter) safe.

`نتت` stopped being invented. `كنت` was still not `كون`.

## Step 3 — al-Khalīl's calculation, and ranking the evidence

Three further pieces, in `analyzer_segmentation_v9.py`:

**1. The evidence is ranked.** `_real_root()` now returns a strength instead of a boolean:

| strength | meaning | weight |
|---|---|---|
| `ATTESTED` = 2 | al-ʿAyn enters this root | **+8** |
| `LEXICON` = 1 | in the 9,013 lexicon, nothing else vouches | **+3** |
| `UNKNOWN` = 0 | not covered | — |
| `MUHMAL` = −1 | al-ʿAyn marks the permutation unused | −12 |

This is the fix for the null result above, and it is what finally separates `كون` (**attested**)
from `كنن` (lexicon only), and the particle `كم` (attested) from `كمم` (lexicon only). Both had
scored identically at +6.

**2. Candidates come from al-Khalīl's calculation, not the lookup.** The tokenizer only proposes
roots it can read off the surface, so a root whose radical was *deleted* is unreachable — no amount
of lexicon filtering finds `كون` in `كنت`. `TasrifEngine._align()` walks the wazn template against
the surface and is allowed to spend a slot on a radical that left no letter behind. Note
`candidate_pairs()` cannot be used for this: it validates with exact equality, and
`generate(كون, فَعَلَ)` is `كان`, not `كن` — so it discards the very root we need.

**3. Elision is constrained by the ajwaf / nāqiṣ division.** Admitting *any* weak-letter deletion
let `كني` win the tie for `كنت`, because deleting the final `ى` from `كنى` also yields `كن`. But:

- **الأجوف** the **medial** radical is weak and *is* deleted before a suffix — `كان + ت → كنت`
- **الناقص** the **final** radical is weak and is *retained* — `كني + ت → كنيت`

So only a **non-final** weak letter may be elided in the loose match.

### Probe set, 60 words

| word | v5 | v9 |
|---|---|---|
| **كنت** | `كنن` | **`كون`** ✓ |
| **قال** | `قال` (wazn read off the surface) | **`قول`** ✓ |
| **دعا** | `دعا` | **`دعو`** ✓ |
| فكم | `كم` ✓ | `كم` ✓ |
| عنهم | `عن` ✓ | `عن` ✓ |
| باع | `بعع` | `بوع` (should be `بيع`) |
| يعد | `عدد` | `عود` (should be `وعد`) |
| يصلي | `ي + صلل + ي` | `ي + صلو` (loses the final clitic) |
| school, zakāt, etc. | — | unchanged |

3 clear improvements, 2 sidegrades, 1 clitic loss, 1 neutral (`بالله` now returns the untouched
baseline instead of a fabricated root — a proper noun with no root to find). Runtime 0.73 s.

### Two regressions found and fixed on the way

- **`مدرسة` / `زكاة` silently lost the ta' marbuta.** Raising al-ʿAyn's weight to +8 was enough for
  `مَفْعَل` on `مدرسة` — which generates `مدرس`, not `مدرسة` — to displace the correct split.
  Fixed by asserting the docstring's own intent: **a reading that cannot produce the stem it was
  read from may not replace the baseline.**
- **`كمم` (score 23) outbid the particle `كم` (score 7).** The particle bonus and a root bonus were
  both being awarded on a particle stem. Fixed: a recognised particle *is* the analysis, and
  `readings` for a particle stem are not calculated at all.

## Step 4 — wired into the shipped tokenizer (and why it defaults OFF)

`models/morphemic_tokenizer_v12_arabic.py` now routes `decompose_arabic_word()` through
`models/validated_segmentation.py`. The patch is **one hunk, +18 lines**: the original method is
kept verbatim as `_decompose_greedy()`, and the new entry point calls it first.

```python
def decompose_arabic_word(self, word):
    base = self._decompose_greedy(word)
    try:    from validated_segmentation import validated_decompose
    except Exception:
        try: from models.validated_segmentation import validated_decompose
        except Exception: return base
    return validated_decompose(self, word, base)
```

The greedy reading is the fallback in **every** failure mode — flag off, record missing, engine
unavailable, or any exception — so the change cannot make a word worse by failing.

### Two modes, because the two halves measured very differently

```
ROOTFORMER_VALIDATED_SEG=0   off — byte-identical to the previous tokenizer
ROOTFORMER_VALIDATED_SEG=1   DEFAULT — the closed class only
ROOTFORMER_VALIDATED_SEG=2   also re-assign roots via al-Khalil's calculation
```

| | mode 0 | **mode 1 (default)** | mode 2 |
|---|---|---|---|
| roots changed, 20,000 corpus words | — | **1,103 (5.51%)** | 3,006 (15.03%) |
| hand-checked sample of 40 changes | — | **40/40 correct** | ~24/40 correct |
| throughput | 75,342 w/s | **48,729 w/s** | 7,285 w/s |
| كنت / قال / دعا | ✗ | ✗ (needs mode 2) | ✓ |
| فكم / عنهم / عليه / منها / فهذه / مني | ✗ | **✓** | ✓ |

The split is the point: **the closed-class half is definitional and measured clean; the root
re-assignment half is heuristic and is not.**

**Mode 1 is the default.** al-Khalīl's first division is ḥarf / ism / fiʿl, and a ḥarf has no root —
so a word consumed by clitics plus a particle cannot have one, whatever a root-and-pattern reading
suggests. Every one of the 40 sampled changes is a genuine function word:

```
وعليه → و + على + ه      منها → من + ها        فيهن  → في + هن
فقد  → ف + قد            لأنه → ل + أن + ه      فيهما → في + هما
وليس → و + ليس           منهما → من + هما       وهذه  → و + هذه
```

`encode`/`decode` round-trip unchanged; `كنت` still needs mode 2, deliberately.

### Why mode 2 is not a default: the 60-word probe lied

The probe set is perfect in mode 2 — كنت→كون، قال→قول، دعا→دعو — and on real text it changes 15% of
roots with roughly a third of those wrong: `يشذ → شوذ` (should stay `شذذ`), `كذا → كوذ`,
`باعي → بعو`, `الفم → فوم`, `بري → برو`. The closed-class half was found by the same
eyeball-a-sample method and came out clean; this half did not, and it stays opt-in until the gold
set can bound its error rate.

### Seven guards, each from an observed failure

Every one came from a specific wrong output, not from theory:

| guard | failure it removes |
|---|---|
| **closed class first** — a word consumed by clitics + a particle has no root | `عليه → علو`, `منها → نها`, `فلا → فلو` |
| prefix must be a **true clitic** (no أ ي ن ت م س، مست، يست) | `ألف → ليف`, `أصل → صول`, `تكل → كول` |
| a **mudariʿa prefix belongs to the wazn** | `يشذ → ي + شذ → شوذ` |
| **bare preposition needs a pronoun** (ب+هل is not a word) | `كانت → ك + انت`, `أهل → هل` |
| و/ف are **conjunctions**, not prepositions — they may precede a bare particle | `فلم`, `فلا`, `فهذه`, `ومن` |
| **if al-ʿAyn already attests the greedy root, don't touch it** | `أصل → صول`, `بالعين → بلع`, `لتكون → وتك` |
| a **demonstrative family** in the closed inventory | `كذا`, `هؤلاء`, `هذان`, `الذين` |

The last two were found only by dropping an earlier over-broad gate (`greedy_root == the whole
word`, added to protect `كان`) — it was also blocking `مني = من + ي`. The `كان` case is now handled
precisely, by refusing to strip a bare preposition.

**Note for training:** mode 1 changes the token stream for 5.5% of words, so
`/workspace/nrmp_cache/*.pt` (built with the greedy analyzer) must be rebuilt before any metric
means anything.

## Files

| file | what |
|---|---|
| `build/khalil_attest_v4.py` | the extractor (idempotent, ~30 s) → `results/khalil_attest_v4.json` |
| `build/analyzer_segmentation_v4.py` | analyzer + v4 record, no rule change — **null result** |
| `build/analyzer_segmentation_v5.py` | + mudaʿʿaf pair gate |
| `build/analyzer_segmentation_v9.py` | + ranked evidence, root calculation, ajwaf/nāqiṣ elision, particle exclusivity |
| `build/validated_segmentation.py` | the same, as a release module with the closed-class gate and the six guards |
| `build/morphemic_tokenizer_v12_arabic_PATCHED.py` | the shipped tokenizer, +18 lines, flag-gated |
