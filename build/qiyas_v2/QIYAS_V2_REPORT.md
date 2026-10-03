# al-qiyās, re-run under the CORRECT condition set — the root as ʿilla

**Session:** delegated subagent of `session-a4f1047e-2a01-48e1-8224-820d03d1563c`
**Pod:** `g6exduq0bd17z8` @ `213.173.104.76:46758` · Python `/workspace/venvs/rootformer/bin/python`
**Date:** 2026-10-03 · **CPU-only.** Every run was `nice -n 19`; no GPU job was launched and no
other agent's process was signalled. Pod load stayed 5–8 of 48 cores (the GPU arm ladder was
running throughout). Nothing was edited in place: `qiyas_engine.py` md5
`44f51619dfda0f6d73d4b2460ddeb681`, byte-identical on pod and local, imported read-only. New
code lives in `/workspace/qiyas_v2/`, mirrored to `rootformer/build/qiyas_v2/`.

---

## 0. Verdict, in one paragraph

**Yes — the root itself is a valid ʿilla by the standard the two authors actually set.** Under
**munḍabiṭ + muṭṭarid only**, `identity` (ʿilla = the root) is **ACCEPTED**, and it is the *best*
ʿilla in the table on both required measures: cell purity **1.000000** (0 impure cells in 25,042)
and iṭṭirād **0.996278** — ahead of the shipped `strict7` (0.988684 / 0.988637). Its old rejection
rested on a fourth condition, *mutāʿaddī*, which al-Ghazālī and al-Rāzī both say is **not
required** («مسألة العلة القاصرة صحيحة»; «مذهب الشافعي أن يجوز التعليل بالعلة القاصرة»); that
rejection is **withdrawn**. **But acceptance does not make the root operative.** On the exact
3,588 held-out-root positions the headline is **flat at 84.48 %** — and it is flat because
`identity` **fires 0 times**: all 3,487 rulings are carried by the root's phonological class
(`strict7`). Isolated from the back-off lattice — the ʿilla being *only* the root, which is what
the question literally asks — it realises **0 / 3,588 = 0.00 %** and identifies **0 / 3,588**, with
**no analogue for all 3,588**, against the learned decoder's 0.00 %. The reason is structural and
is exactly the classical one: the ruling here (the root→surface alignment) is **wazn-specific**, so
a root carries no ruling to a form of a different wazn. **The root is the aṣl — the locus — and its
ʿilla is valid but *qāṣira* (non-extending); the operative cause of the 84.48 % is a determinate
property of the root, its phonological class, which is the *mutāʿaddiya* refinement.** In the
tradition's own terms: **العلة القاصرة صحيحة، والتعدية فرع الصحة** — so the root is a sound ʿilla
of the confined kind, and the claim "a shared root representation generalises to unseen
derivational forms" is true of the root's determinate class, not of the root *as such*.

---

## 1. The condition set, verified in the held texts (not re-derived)

Every locator below was **opened and read** before use. Editions named, because the project holds
these works in more than one edition and the line numbers move.

```
corpus/usul/Ghazali_Al_Mustasfa.txt   md5 a7866827987e08dca6a82097e62169c6  15,369 lines
corpus/usul/Razi_Al_Mahsul.txt        md5 ba4ae500744d579da59d75010cc43a31  19,330 lines
corpus/basran/Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt
                                      md5 663e493f1d2e7a3223dcc8c6b702e7d0   1,701 lines
```

The pod files `/workspace/scholastic_sanitized/{Ghazali_Al_Mustasfa,Razi_Al_Mahsul}.txt` and
`/workspace/heritage_foundations/Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt` have **identical md5s**
to the local copies, so the locators below resolve on both sides. (The **OpenITI** edition of
al-Īḍāḥ is a different file — 1,864 lines, shifted **+92 / +142 / +148**; per
`build/antigravity/idah_verify.md`, the `1203–1205` locator is on the sister edition, which is
what is quoted here. Naming the edition is load-bearing.)

**Verified line by line** (`awk` on the exact ranges, not a string search):

| condition | status | locator (edition named) | the line as it stands |
|---|---|---|---|
| muṇḍabiṭ | **REQUIRED** | Mustasfa:12851 | «الربا جار في الدقيق والعجين **فلم ينضبط باسم البر فلا بد من ضابط** ولا ضابط أولى من الطعم» |
| muṭṭarid | **REQUIRED** | Mustasfa:13254 | «فقال قوم: إنه ينقض العلة ويفسدها ويبين أنها لم تكن علة **إذ لو كانت لاطردت ووجد الحكم حيث وجدت**» |
| muṭṭarid | **REQUIRED** | Mahsul:14389–90 | «**لأن الطرد واجب في العلل والعكس غير واجب فيها**» |
| munʿakis | **NOT required** | Mustasfa:12599–600 | «**العكس ليس بشرط في العلل الشرعية** فلا أثر لوجوده وعدمه» |
| munʿakis | **NOT required** | Mahsul:15485–87 | «**وإما أن العكس غير واجب في العلل** فهو قولنا وقول المعتزلة وأما أصحابنا فإنهم أوجبوا العكس في العلل العقلية وما أوجبوا في العلل الشرعية» |
| mutaʿaddī | **NOT required** | Mustasfa:13507 | «**مسألة العلة القاصرة صحيحة** وذهب أبو حنيفة إلى إبطالها» |
| mutaʿaddī | **NOT required** | Mustasfa:13511 | «**فالتعدية فرع الصحة** فكيف يكون ما يتبع الشيء مصححا له؟» |
| mutaʿaddī | **NOT required** | Mahsul:15982–86 | «**مذهب الشافعي رضي الله عنه أن يجوز التعليل بالعلة القاصرة** وهو قول أكثر المتكلمين وقال أبو حنيفة وأصحابه لا يجوز … فلو توقفت صحتها في نفسها على صحة تعديتها إلى الفرع **لزم الدور**» |

**I agree with the settled set: required = muṇḍabiṭ + muṭṭarid, and only those.** mutaʿaddī is
**not** reintroduced as a gate anywhere in this report; it is reported in §3 as a *non-required
diagnostic*.

### 1.1 One correction to the old report's *reason*, stated because it was cited

`QIYAS_REPORT.md:200–206, 232–243` rejected `identity` by reading `Mahsul:15719–15721` as a
general requirement that the ʿilla be extendable. That is not what the passage says. Read in
full (Mahsul, md5 `ba4ae500…`):

```
15696  المسألة الأولى اختلفوا في جواز التعليل بمحل الحكم والحق أن العلة إما أن تكون قاصرة أو
15697  متعدية فإن كان الأول صح التعليل بمحل الحكم ...        <- al-Razi's OWN position: qasira OK
15700  فإن قلت لو كان محل الحكم علة للحكم لكان الشئ الواحد فاعلا وقابلا معا وهو محال لوجهين
                                                              <- the OBJECTION, introduced «فإن قلت»
15718  قلت قد بينا في كتبنا العقلية ما في هذين الوجهين من المغالطة وأما إن
                                                              <- al-Razi's REPLY to that objection
15719  كانت العلة متعدية لم يصح أن يكون محل الحكم علة للحكم لأن العلة المتعدية
15720  هى التي توجد في غير مورد النص وخصوصية مورد النص يستحيل حصولها في غيره
15721  لأن الشئ لا يكون نفس غيره المسألة الثانية الوصف الحقيقي إذا كان ظاهرا
15722  مضبوطا جاز التعليل به                                <- and immediately: the mundabit condition
```

Two things follow, and neither is the old report's reading. **(a)** `15719–21` is al-Rāzī's **own
voice**, continuing his reply after «قلت … المغالطة»; it is not a quoted opponent position being
refuted. (The task's framing — that this passage *is* the refuted position — is therefore also
not what the text shows.) **(b)** Its subject is a narrower question — *التعليل بمحل الحكم*,
whether the **locus of the ruling** may itself serve as the ʿilla — and it is conditioned on the
ʿilla being *mutaʿaddiya*; it presupposes, two lines earlier, that a *qāṣira* ʿilla is sound. It
is not a general "the ʿilla must be extendable". The munḍabiṭ condition begins in the very next
sentence (`15721–22`), which is likely why the neighbourhood was misread.

*Note:* `STATE.md:152–157` already records the correct reading («`:15719-21` is Rāzī's own qāṣira
argument»), while `STATE.md:101–105` still carries the earlier, superseded framing. The
correction here is independent — made by opening the file — and agrees with `:152–157`.

Also worth recording, because the previous report marked it "not verifiable": **Ibn ʿUsfūr is
held**, three works, and the dictum verifies verbatim at
`corpus/andalusian/13_IbnUsfur_Mumtic_fi_al_Tasrif.txt:161` —
«هذا النحو من الاشتقاق **غير مأخوذ به؛ لعدم اطراده**، ولما يلحق فيه من التكلف». The earlier
search looked only at the pod's `heritage_foundations/`, where he is absent.

---

## 2. Reproducibility — the two runs are directly comparable

Asserted in code, not assumed (`two_condition_rerun_v2.py`, 11 asserts; all passed):

| quantity | this run | stored run | |
|---|---|---|---|
| vocab | 9,490 roots / 142 awzan | 9,490 / 142 | ✓ |
| aṣl set | **106,664** / 5,551 roots | 106,664 / 5,551 | ✓ |
| held-out types | 188,722, default round-trip 108,682 = **57.5884 %** | same | ✓ |
| gold positions | **3,588** / 37 roots | 3,588 / 37 | ✓ |
| gold histogram vs `comp24k_eval.json` | `histogram_matches_recorded = True` | True | ✓ |
| aṣl excluding the 37 roots | **104,549** / 5,514 roots | 104,549 | ✓ |
| **comparator** `strict7` Task A | **3,023 / 3,588 = 84.2531 %** | 3,023 = 84.2531 % | ✓ |
| **comparator** `strict7` Task B top-1 | **3,031 / 3,588 = 84.4760 %** | 3,031 = 84.4760 % | ✓ |

The comparator was asserted to be reproduced *before* any conclusion was drawn, so the two runs
are the same experiment with one thing changed: the condition set.

---

## 3. The corrected condition table — `identity`'s verdict

Induction over the same **106,664 admissible aṣl / 5,551 roots**. Thresholds unchanged from
`qiyas_corpus.py` (declared before the numbers were seen): purity ≥ 0.98, iṭṭirād ≥ 0.98. The
required pair is judged on the ʿilla's **own** rule (no fall-back crutches), exactly as stored.
munʿakis and mutaʿaddī are shown as **non-required diagnostics** and do not gate.

| ʿilla | cells | muṇḍabiṭ purity | impure cells | muṭṭarid iṭṭirād (own) | exceptions | **2-cond verdict** | *[not required]* munʿakis lift | *[not required]* mutaʿaddī (H-pair firing) |
|---|---|---|---|---|---|---|---|---|
| `wazn_only` | 39 | 0.945671 ✗ | 32 | 0.945671 ✗ | 5,795 | **REJECT** | +0.0000 (by construction) | 0.0000 |
| `coarse_weak` | 135 | 0.988515 | 61 | 0.988506 | 1,226 | ACCEPT | +0.1209 | 0.9999 |
| `strict7` *(shipped)* | 231 | 0.988684 | 79 | 0.988637 | 1,212 | ACCEPT | +0.0752 | 0.9998 |
| `positional` | 462 | 0.990062 | 72 | 0.989837 | 1,084 | ACCEPT | +0.0639 | 0.9960 |
| **`identity` (ʿilla = THE ROOT)** | **25,042** | **1.000000** | **0** | **0.996278** | **397** | **ACCEPT** | **+0.1297** | **0.0000** |

```
ACCEPTED under munḍabiṭ + muṭṭarid : ['coarse_weak', 'strict7', 'positional', 'identity']
REJECTED                            : ['wazn_only']
identity (ʿilla = THE ROOT) verdict  : ACCEPTED   purity 1.000000 (>= 0.98), ittirad 0.996278 (>= 0.98)
```

**`identity` is ACCEPTED, and it is the highest-scoring ʿilla on both required measures.** This
supersedes the old table's `REJECTED by mutāʿaddī`. The only ʿilla the corrected set rejects is
the negative control `wazn_only` — and it is rejected on both required conditions independently
of any fourth.

*One honest note on the opposite flank:* identity's iṭṭirād (0.996278) is **not** better than its
old "with-back-off" figure suggested, and it is not vacuous-free. See §4.

---

## 4. What `identity`'s acceptance actually is — forensics

This matters, because "ACCEPT" here means something narrower than it looks.

| fact | value | consequence |
|---|---|---|
| identity cells | 25,042 | one cell per (root, wazn) |
| observations per cell | 4.26 mean | — |
| **singleton cells** | **10,066** | vanish under leave-one-out |
| cells with >1 distinct alignment (**genuine impurity**) | **0** | purity is **exactly** 1.000000 |
| iṭṭirād failures | **397** | **all 397 are singleton-cell collapses** |
| distinct cells containing a failure | 397 | **0 non-singleton exception cells** |

So identity's perfection is partly definitional. `illa_identity(root) = root`, so the cell key
`(label, wazn)` is `(root, wazn)` — **a cell contains observations of exactly one root**. Muṇḍabiṭ's
purity test therefore has nothing to discriminate: the modal alignment trivially covers the whole
cell. And every one of the 397 iṭṭirād failures is a singleton cell that empties under
leave-one-out and then falls to the context-free template — arithmetic, not irregularity. The
LOO-cell separation proves it: `identity` has **397 failing cells, 397 of them singleton-only,
and 0 non-singleton exception cells** (`exception_loo_cells.py`).

*For contrast, by the same separation:* `strict7` has 1,212 failures in 81 cells — 2 singleton-only
and **79 genuine exception cells**; `positional` 1,084 in 83 cells (11 singleton, **72 genuine**).

**Three holdout protocols** (same 106,664 aṣl), to see where identity does and does not fire:

| protocol | what is withheld | `wazn_only` | `strict7` | `positional` | **`identity`** |
|---|---|---|---|---|---|
| **H-pair** | 10 % of (root,wazn) **cells** — 2,504/25,042, 10,544 obs | firing 0.0000 | **0.9998** | 0.9960 | **0.0000** |
| | realisation | 0.9586 | 0.9955 | 0.9955 | 0.9955 *(carried by strict7)* |
| **H-obs** | 10 % of **observations**, cells stay populated — 10,666 | 0.0000 | 1.0000 | 0.9997 | **0.8987** |
| | realisation | 0.9551 | 0.9970 | 0.9975 | **0.9995 (best)** |
| **H-root** | the **root** — see §5 | 0.0000 | fires 3,487 | fires 3,487 | **0.0000** |

The pattern is unambiguous and is the whole answer: **identity fires only when its own (root,wazn)
cell is present.** Withhold the cell and it fires 0 / 10,544. Withhold the root and it fires
0 / 3,588. Leave the cell present and withhold a token, and it is the *best* predictor of all
(realisation 0.9995). It is a valid cause **of its own locus** — a *qāṣira* ʿilla — and it does not
travel.

---

## 5. The headline: does the root-as-ʿilla change 84.48 %?

Three configurations, each measured on the **exact 3,588** held-out-root positions (aṣl induced
from the 104,549 whose root is not among the 37). Reported against the **84.48 % comparator**
(`strict7`, the ʿilla previously used) and the **0.00 % learned decoder**.

| Task A — realisation | positions | rate |
|---|---|---|
| shipped hand-coded realiser | 2,933 / 3,588 | 81.7447 % |
| **`strict7`** — the previous ʿilla (comparator) | **3,023 / 3,588** | **84.2531 %** |
| `positional` | 3,023 / 3,588 | 84.2531 % |
| **`identity` + the engine's back-off lattice** | **3,023 / 3,588** | **84.2531 % — FLAT** |
| `identity` + only the context-free template below it | 2,695 / 3,588 | 75.1115 % |
| **`identity` ISOLATED (the root alone)** | **0 / 3,588** | **0.0000 %** |
| `wazn_only` (negative control) | 2,695 / 3,588 | 75.1115 % |

| Task B — inverse root identification | top-1 | top-5 |
|---|---|---|
| **neural compositional decoder** (same 3,588 gold roots) | **0 / 3,588 = 0.00 %** | 0.00 % |
| chance over 37 candidates | 2.7027 % | 13.5135 % |
| **`strict7`** — the previous ʿilla (comparator) | **3,031 / 3,588 = 84.4760 %** | 84.5039 % |
| `positional` | 3,032 / 3,588 = 84.5039 % | 84.5039 % |
| **`identity` + the engine's back-off lattice** | **3,031 / 3,588 = 84.4760 % — FLAT** | 84.5039 % |
| `identity` + only the context-free template below it | 2,694 / 3,588 = 75.0836 % | 75.1115 % |
| **`identity` ISOLATED (the root alone)** | **0 / 3,588 = 0.0000 %** | 0.0000 % |
| `wazn_only` (negative control) | 2,694 / 3,588 = 75.0836 % | 75.1115 % |

**Firing provenance answers the "flat" question.** Under the lattice, `identity` fires
**0 of 3,588** times; the provenance is `strict7` **3,487**, `NONE` 101 — byte-identical to the
comparator. So the 84.48 % is **not** attributable to the root: it is carried entirely by the
root's **phonological class**. Under isolation the root is the *only* ʿilla and every one of the
3,588 positions has **no analogue at all** (3,588 × `NONE`), giving 0.0000 %.

**So the plain answer: flat at 84.48 % under the lattice, and 0.00 % when the root is actually
the ʿilla. Up? No. Down? Yes — to zero — as soon as the class fall-back is removed. Flat? Only
in the sense that the root is firing zero times in a system where something else does all the
work.**

*Limits on the comparator, carried forward unchanged:* Task B with the **full 9,220-root
inventory** is **79.10 %** (2,838/3,588), not 84.48 % — the 37-root candidate set inflates the
headline by ~5.4 pp (`STATE.md:201–214`, `full_inventory_id.py`). And Task B is supplied the gold
prefix/suffix split. Neither limit is affected by this re-run.

---

## 6. Is the phonological-class ʿilla still better, or superseded?

**Both, in different registers — and the distinction is the finding.**

* As a **verdict under the two conditions**: `identity` is **better** — purity 1.000000 vs
  0.988684, iṭṭirād 0.996278 vs 0.988637, munʿakis lift +0.1297 vs +0.0752 (the last not
  required). On the standard the authors set, the root beats the class.
* As an **operative ʿilla for unseen derivational forms**: the class is **not superseded, and is
  the only thing that fires** — `identity` fires 0/3,588 on the held-out-root task and 0/10,544 on
  the withheld-cell task. Its 84.48 % is the class's 84.48 %.
* Where the root *is* better at prediction — H-obs, realisation 0.9995 vs 0.9970 — the cell is
  present, so it is predicting a token of an attested type, not extending to a new farʿ.
* Between the two class ʿillas: `positional` is marginally ahead of `strict7` on the headline
  (Task B 3,032 vs 3,031, +1 position), on purity (0.990062 vs 0.988684) and on iṭṭirād
  (0.989837 vs 0.988637), but its cells are twice as sparse (462 vs 231) and it carries **72
  genuine exception cells to strict7's 79**. Both are accepted; `strict7` remains the shipped one
  and the comparator here.

**One-line summary: the root is the valid-but-confined cause; the root's class is the effective
cause that extends.**

---

## 7. The exception cells, under al-Zajjājī's licence

The licence was **opened and verified** before use —
`corpus/basran/Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt:1203–1205`, md5 `663e493f1d2e7a3223dcc8c6b702e7d0`:

```
1203  الجواب في ذلك أن يقال: إن الشيء إذا اطرد عليه باب، فصح في القياس وقام
1204  ~~في المعقول، ثم اعترض عليه شيء شاذ نزر قليل، لعلة تلحقه، لم يكن ذلك مبطلا
1205  ~~للأصل، والمتفق عليه في القياس المطرد، ...
```

The licence is **conditional**: the exception must be «شاذ نزر قليل» **and** «لعلة تلحقه» — it
must have a determinate cause attaching to it. Both halves were tested, not assumed.

**Definition used:** for an ʿilla's own table, an **exception cell** is a cell whose modal
alignment does not reproduce every member (`purity < 1`); its non-modal members are the
exceptions. "Gets its own ʿilla" is tested as: the **finer accepted ʿilla** reproduces that
exception's surface from its own determinate cell — i.e. the exception is a determinate
sub-regularity, not an accident.

| ʿilla | cells | exception cells | exception obs | share of aṣl | **all exceptions explained by the finer ʿilla** | partly | none |
|---|---|---|---|---|---|---|---|
| `wazn_only` | 39 | 32 | 5,795 | 5.4329 % | 25 | 6 | 1 |
| `coarse_weak` | 135 | 61 | 1,225 | 1.1485 % | 52 | 6 | 3 |
| `strict7` *(shipped)* | 231 | **79** | 1,207 | 1.1316 % | **69** | 5 | 5 |
| `positional` | 462 | 72 | 1,060 | 0.9938 % | 62 | 3 | 7 |
| **`identity`** | 25,042 | **0** | **0** | **0.0000 %** | 0 | 0 | 0 |

The LOO-cell separation (§4) gives the same picture by the other route: `identity` **0 genuine
exception cells** (all 397 failures singleton collapses), `strict7` **79**, `positional` **72**,
`coarse_weak` **61**, `wazn_only` **32**.

**Reading.** The licence applies cleanly to the **class** ʿillas and not at all to the root. For
`strict7` — the ʿilla actually shipped — the exception set is genuinely «نزر قليل»: 1,207
observations = **1.13 % of 106,664**, in 79 cells, and of those 79 cells **69 have every
exception explained by the finer `positional` ʿilla** (5 partly, 5 not). That is the licence's
second half satisfied in the strong form: 953 of its 1,207 exceptions (79 %) have a determinate
ʿilla attaching to them. The residual — 254 observations, 5 cells — is the honest rump of
idiosyncratic data. For `identity`, by contrast, the exception set is **empty**, so the licence
has nothing to do: the root needs no excusing because it never extends far enough to have an
exception.

### 7.1 The requested "**36** exception cells" — not reproducible

No artifact in this project defines 36 exception cells. I searched the whole local tree and the
pod (`*.md`, `*.py`, `*.json`, `*.log`, including `STATE.md`, `NEXT.md`, `QIYAS_REPORT.md` and
every qiyās artifact) for `"36 exception"`, `"exception cells"`, `"exception_cells"` and
«شاذ نزر» — there is **no such figure**. Rather than accept a number I could not source, I
computed the exception cells directly under every natural definition. The measured counts are:

```
exception cells (purity < 1)          : wazn_only 32, coarse_weak 61, strict7 79, positional 72, identity 0
exception cells all-explained by finer: wazn_only 25, coarse_weak 52, strict7 69, positional 62, identity 0
LOO non-singleton exception cells     : wazn_only 32, coarse_weak 61, strict7 79, positional 72, identity 0
cells containing any LOO failure      : wazn_only 32, coarse_weak 61, strict7 81, positional 83, identity 397
```

None is 36. The nearest values are 32 (`wazn_only`) and 79 (`strict7`). **I report this as a
non-match rather than force the number**, and flag it back so a locator can be supplied if the
figure came from an artifact I cannot see. Everything the requested step was *for* — the exception
cells getting their own ʿilla under the Zajjājī licence — is done above, and it is done for the
ʿilla that actually has exception cells (`strict7`: 79 cells, 69 fully excused).

---

## 8. The verdict, in the tradition's own terms

The question the project has been circling is: *is the root a valid ʿilla?* The answer the two
authorities support, and the measurement confirms, is:

> **الجذر أصلٌ، وعلّته قاصرة صحيحة.** The root is the **aṣl** — the locus, the attested instance.
> Taken *as* the ʿilla it is **muṇḍabiṭ** (purity exactly 1.000000, 0 impure cells in 25,042) and
> **muṭṭarid** (iṭṭirād 0.996278, with all 397 failures being arithmetic singleton collapses and
> **no genuine exception cells**). Both required conditions are met. Nothing in al-Ghazālī's or
> al-Rāzī's two conditions forbids it. **It is a valid ʿilla.**

> **والعلة القاصرة صحيحة، والتعدية فرع الصحة.** Its limitation is **qaṣr** — it exists only in its
> own locus — and **qaṣr does not invalidate**. This is precisely why the old rejection was
> unsound: it rejected on *taʿdiya*, and «التعدية فرع الصحة», extension is downstream of validity
> (`Mustasfa:13511`), while «مسألة العلة القاصرة صحيحة» (`Mustasfa:13507`) and «مذهب الشافعي أن
> يجوز التعليل بالعلة القاصرة وهو قول أكثر المتكلمين» (`Mahsul:15982–83`) affirm the confined
> cause outright.

> **والعلة المتعدية في هذا الباب هي الصنف الصوتي للجذر، لا الجذر نفسه.** The cause that actually
> travels to a **farʿ** is a *determinate property of the root* — its phonological class
> (ṣaḥīḥ / mithāl / ajwaf / nāqiṣ / lafīf / muḍaʿʿaf / mahmūz), refined by `positional` — because
> the **ḥukm here is wazn-specific**: the root→surface alignment is fixed by the wazn, and only the
> irregular adjustments (iʿlāl, idghām, hamza) depend on the root. Hence a root carries no ruling
> to a form of a different wazn, and the class does. That is measured: under the lattice the class
> fires 3,487/3,588 and the root **0/3,588**; isolated, the root realises **0/3,588**.

So the project's claim — *a shared root representation generalises to unseen derivational forms* —
**stands, with its ʿilla named precisely.** It is not the root *as such* that does the carrying
(that is the aṣl, and its ʿilla is qāṣira); it is a **determinate, muṇḍabiṭ and muṭṭarid property
of the root** — its phonological class — which is the **mutāʿaddiya** ʿilla. The number 84.48 %
belongs to the class, not to the root; and the root's own status is not "rejected" but
**صحيحة قاصرة**.

---

## 9. Honest limits

1. **The headline does not move, and that is the finding — not a null result.** Flat at 84.48 %
   with `identity` firing 0×; 0.00 % isolated. Anyone reading "the root is now ACCEPTED" as "the
   root produces 84.48 %" has read it backwards: §5's provenance histogram is the guard.
2. **`identity`'s two mandated conditions are met partly by construction.** One root per cell
   makes purity 1.000000 unfalsifiable as a test, and its iṭṭirād failures are singleton
   artefacts. The conditions say **valid**, and valid it is — but "ACCEPT" should not be read as
   "as informative as `strict7`'s ACCEPT".
3. **The isolation is done in the harness, not the engine.** `qiyas_engine.py` was not edited
   (read-only, md5 recorded). `isolate()` resets `chain`, `tables` and `wazn_cells` after
   `induce()`, working around the two traps the project had already found
   (`qiyas_engine.py:322–323` appends `wazn_only` even with `backoff=False`; `:397` falls through
   to `wazn_cells` outside the chain loop). Because a check that passes on presence is worthless,
   the path is **executed and asserted**: positive control 800/800 pairs still return a rule,
   negative control **0/800** withheld pairs return one. Both asserts fire in the run.
4. **Task B's 84.48 % is the 37-root candidate set.** The full 9,220-root inverse search gives
   **79.10 %**; the gold prefix/suffix split is supplied. Unchanged by this work.
5. **"36 exception cells" could not be sourced** (§7.1). I computed the number rather than adopt
   it. If it came from an artifact outside this workspace, the locator would let me re-check.
6. **The four-condition column in §3 is history, not a live gate.** It is shown so the two runs
   are comparable; only the 2-condition column determines the verdict.
7. **The 1,207-vs-1,212 discrepancy is real and benign.** 1,212 is `ittirad_fail` (leave-one-out);
   1,207 is the in-sample sum of `cell_size − modal_support`. The 5-observation gap is one more
   consequence of singleton cells vanishing under LOO.

---

## 10. Artifacts

Pod `/workspace/qiyas_v2/` ⇄ local `rootformer/build/qiyas_v2/`:

| file | what it is |
|---|---|
| `two_condition_rerun_v2.py` | the re-run: P0 reproducibility, P1 corrected verdict, P2 forensics + 3 holdouts, P3 headline, P4 exception cells, P5 verdict. 11 asserts. |
| `two_condition_rerun_v2.log` | full run log (72 s incl. both harvests) |
| `two_condition_rerun_v2.json` | every number |
| `exception_cellwise.py` / `.log` / `.json` | per-cell Zajjājī test: all/partly/none of each exception cell's exceptions explained by the finer accepted ʿilla |
| `exception_loo_cells.py` / `.log` / `.json` | separates genuine exception cells from singleton LOO collapses |
| `two_condition_rerun.py`, `two_condition_rerun.json` | the **prior** agent's partial re-run, kept for provenance (it did the condition table only; no headline, no exception cells) |

Deterministic, CPU-only, `nice -n 19`. No shipped module touched; no other agent's process
signalled.
