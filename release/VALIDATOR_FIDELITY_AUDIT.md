# Validator fidelity audit

**The rule this file applies.** The Basran grammarians (al-Khalīl, Sībawayh, al-Mubarrad, al-Akhfash,
Ibn Jinnī, al-Zajjājī, al-Azharī) and the Andalusians (Ibn Mālik, Ibn ʿUsfūr, Abū Ḥayyān,
al-Shātibī, Ibn Maḍāʾ, Ibn Sayyidih) are not a hypothesis awaiting validation. The question for any
implementation is only whether it is **faithful to what they specified**. So every rule in
`validated_segmentation.py` must either cite a primary text or be marked as engineering.

## Inventory: what the validator asserts

### Rules with a classical basis to check

| # | rule as implemented | where in the code |
|---|---|---|
| 1 | ḥarf / ism / fiʿl division; a ḥarf has no root | `_closed_split`, the closed-class branch |
| 2 | proclitics: ال، و، ف، ب، ك، ل (+ وال، بال، كال، فال، لل) | `PREFIXES`, `PRE_CLOSED` |
| 3 | the muḍāriʿa letters أ ن ي ت belong to the wazn, not the clitic layer | `MUdARIA_LITERAL`, the prefix guard |
| 4 | pronominal enclitics ـه ـها ـهم ـك ـي ـنا … | `PRON_SUFFIXES`, `SUFFIXES` |
| 5 | alif dropped before a pronoun: على + ه → عليه | `_closed_split` (the `core[:-1] + 'ى'` branch) |
| 6 | a bare preposition cannot stand alone: ب + هل is not a word | `SINGLE_PREPS` |
| 7 | و/ف are conjunctions and may precede a bare particle: فلم، فلا، ومن | `SINGLE_PREPS` (excludes و، ف) |
| 8 | ajwaf deletes its medial weak radical; nāqiṣ retains its final one | `_generates` (the `pos != len(gen)-1` guard) |
| 9 | al-qalb: a waw after kasra → yāʾ; alif may stand for waw/yāʾ | `khalil_root_calculator`, `SUBST` |
| 10 | al-taqālīb: permute the radicals, then accept or reject | `_candidate_roots`, `TasrifEngine._align` |
| 11 | مستعمل / مهمل as al-ʿAyn's own verdict | `_real_root`, the `ATTESTED`/`MUHMAL` ranking |
| 12 | the muḍāʿʿaf root is the doubling of an ordered pair, judged by the 2-letter chapter | the muḍāʿʿaf pair gate |
| 13 | demonstratives and relatives are a closed class, not root-derived | `SUPPLEMENT_CLOSED` |
| 14 | لفظ الجلالة is a name, not a root-and-pattern word | `INDECLINABLE_NAMES`, `NAME_WORD` |

### Engineering, with no classical basis — marked as such

| item | value | status |
|---|---|---|
| round-trip via the generator as the standard of proof | `_generates` | engineering device |
| scoring weights | root round-trips `+10`, cannot generate `−6`, attested `+8`, lexicon `+3`, muhmal `−12`, each affix `−1`, particle `+8` | **invented numbers** |
| flag modes | `0`/`1`/`2` | engineering choice |
| the 46 category tokens and 65 bare particles in the blueprint | ids 117..228 | project-built, needs its own sourcing |

## Finding 1 — the closed inventory was duplicated by hand

`BASE_CLOSED` restated by hand what the blueprint already carries as its **Sībawayhian particle
partition**: 46 category tokens plus 65 bare particles at ids 117..228. The blueprint's list is the
project's own encoding of the source; a second hand-typed copy can only drift from it.

Measured: the blueprint partition contributed **65 items, 0 of them new** — my list was a strict
superset. The partition is now loaded as the primary inventory and my list is demoted to
`SUPPLEMENT_CLOSED`, labelled unsourced, with the count of items from each reported in the stats.

Behaviour was unchanged (1,103 root changes in 20,000 corpus words, probe set unchanged), which is
precisely the point: the duplication was invisible until it drifted.

## Finding 2 — the disambiguation is numeric, and the sources give principles

`_score()` chooses between competing analyses by adding invented integers. The tradition does not
work this way: it decides by named principles — **al-aṣl** (the underlying form), **al-ḥaml ʿalā
al-akthar** (prefer what is more attested), **al-istʿmāl** (actual usage), and al-ʿAyn's own
**mustaʿmal / muhmal**. A lexicographic priority over those principles would be faithful where a
weighted sum is not.

The four running audits are checking, against the corpus, which of these are stated outright and
which I have been assuming. Verdicts are recorded below.

## Audit results

### Audit 1 — ḥarf / ism / fiʿl, prefixes, the divine name

Every quotation below was retrieved from the corpus by search; line numbers are in the sources.

| claim | verdict | the text |
|---|---|---|
| the three classes | **SOURCED** | Sībawayh, *Al-Kitāb*, «فالكلم: اسم، وفعل، وحرف جاء لمعنى ليس باسم ولا فعل» — and he names the ḥarf: «فنحو: ثم، وسوف، وواو القسم ولام الإضافة» |
| a ḥarf has no root / no derivation | **SOURCED** | Ibn Jinnī, *Al-Khaṣāʾiṣ*: «الحروف يشتق منها ولا تشتق هي أبدا». Al-Azharī, *Tahdhīb*: «والحروف المبهمة: التي لا اشتقاق لها، ولا يعرف لها أصول، مثل الذي والذين وما ومن وعن» |
| the four muḍāriʿa letters | **SOURCED** | Sībawayh: «الزوائد الأربع: الهمزة، والتاء، والياء، والنون». The mnemonics أنيت / نأيت are in al-Shāṭibī's *Sharḥ Alfiyyah*, Abū Ḥayyān's *Irtishāf*, and *Lāmiyyat al-Afʿāl* |
| a preposition must govern a majrūr | **SOURCED in substance** | Al-Mubarrad, *Al-Muqtaḍab*: «كل ما دخل عليه حرف من حروف الجر فهو اسم». Sībawayh: «كما أن الجر لا يكون إلا في الأسماء» |
| **ال + و ف ب ك ل as a group of zoʾāʾid** | **NOT FOUND as stated** | Sībawayh's ten zāʾida letters are سألتمونيها — **ف، ب، ك are not among them**. Individual augments are attested (al-Zajjājī: «والألف واللام زائدان»), the grouped list is not |
| **الله is not root-analysed** | **CONTRADICTED by the sources** | Sībawayh: «وكأن الاسم والله أعلم إله، فلما أدخل فيه الألف واللام حذفوا الألف وصارت الألف واللام خلفا منها». Al-Mubarrad: «أصل هذا: إلاه. وأن الألف واللام بدل من همزة إله». Ibn Fāris gives the root ل-ا-ه |

#### The divine name — closed, by instruction

`INDECLINABLE_NAMES` / `NAME_WORD` leaves `الله` and its clitic-bearing forms untouched. I had
briefly withdrawn it on the strength of claim 6(b); the result was worse, not better, and it is
restored:

```
withdrawn:  الله  → ('ال', 'له', None, None)       — not Sibawayh's analysis either
            إله   → (None, None, None, None)       — no analysis at all
restored:   الله  → (None, 'اله', 'فَعَلَ', None)   — identical to the greedy reading
            بالله → (None, 'بلل', 'فَاعِل', 'ه')    — identical to the greedy reading
```

**Decision: leave the word alone.** The exemption is recorded as a *deliberate product decision* —
the word is not re-segmented — and **not** as fidelity to the sources, which do analyse it
(Sībawayh: «أصل هذا: إلاه»; Ibn Fāris: root ل-ا-ه). The code comment says exactly this, so that
nothing downstream reads the exemption as a grammatical claim. Effect on the record: none — 1,103
root changes in 20,000 words, the same as before the episode.

#### The unsourced prefix group

The rule stands in behaviour — particles do enter upon words, which Sībawayh states of `ال`
(«تدخلان لتعريف وتخرجان») and the ḥurūf al-maʿānī tradition elaborates — but my *justification*
was wrong: I had called `و ف ب ك ل` a set of zoʾāʾid, and that grouping is unattested. It is
recorded as a practical clitic list, not a cited catalogue.

### Audit 3 — clitics and pronouns

| claim | verdict | the text |
|---|---|---|
| the pronominal clitic set | **SOURCED** | Sībawayh, *Al-Kitāb*, «باب علامة المضمرين المنصوبين» enumerates it item by item: «الكاف التي في رأيتك، وكما التي في رأيتكما، وكم … وكن … والهاء التي في رأيته … ورأيتها … ورأيتهما … ورأيتهم … ورأيتهن … ونى … ونا». The ـي from «والياء في غلامي وبى» |
| alif before an attached pronoun | **SOURCED — as al-QALB, not al-ḥadhf** | Ibn ʿUsfūr: «إن العرب قد **تقلب** الألف ياء مع المضمر في نحو: عليه وإليه ولديه». Al-Zajjājī (*Ḥurūf al-Maʿānī*, لدى): «ومع المضمر **تنقلب** ياء تقول لدى زيد ولديك». Abū Ḥayyān: «وقلبت ألفه ياء لإضافته إلى المضمر» |
| a bare preposition is not self-sufficient | **SOURCED — and it gives the reason** | Ibn Mālik, *Tashīl*: «وخص الجر بالاسم **لأن عامله لا يستقل**». *Sharh al-Kāfiya*: «والجر مخصوص بالاسم … لامتناع دخول عامله عليه». Ibn ʿUsfūr: «وحروف الجر لا بد لها مما تتعلق به ظاهرا أو مضمرا» |
| the two kinds of tāʾ | **SOURCED** | Al-Zajjājī: «التاء تكون **اسما وحرفا** فَالاسم قولك قمت وخرجت والحرف قولك هند قامت». Al-Mubarrad names the pattern side: «ومن هذا **الوزن** فعلت … فالتاء **الزائدة** عوض» |
| a preposition + pronoun is one unit | **SOURCED** | Sībawayh: «لأن المجرور داخل في الجار، **فصارا كأنهما كلمة واحدة**»; and he ranks «الهاء التي في عليه» as علامة الإضمار that «لا تصرف ولا تذكر إلا فيما قبلها» |
| و/ف may precede a particle | **PARTIAL** | Stated for named particles — Sībawayh «والواو تدخل على هل»; al-Mubarrad «إنما الواو تدخل عليهن … وهل … وكيف … ومتى … وأين»; al-Zajjājī «فإن دخل عليها الواو أو الفاء». **No general rule found**, and none for و+من specifically; Ibn Jinnī's examples do give «فلم تجبني» and «فقد أجبتك» |
| relatives/demonstratives are a closed, **non-derived** class | **PARTIAL** | The cataloguing is real — Ibn Mālik: «وجملة المعارف سبعة: المضمر، والعلم، واسم الإشارة، والموصول…»; Sībawayh's «الأسماء المبهمة»; Ibn Jinnī: they «جارية مجرى الأسماء المضمرة». But **«not derived from roots» is not stated for them** — the explicit «غير مشتق» is only for pronouns, and Ibn Sayyidīh assigns الذي radicals: «الأصول من الذي ثلاثة أحرف لام وذال وياء» |

#### Corrections applied

| what I had written | what the sources say | action |
|---|---|---|
| "a particle ending in alif **drops** it before a pronoun", with the example ما + ه → مه | the alif is **turned into** yāʾ (قلب), and the example is **unattested** | comment rewritten, bogus example removed |
| the bare-preposition rule, justified loosely | Ibn Mālik supplies the actual reason: «عامله لا يستقل» | now cited in the code |
| و/ف "are conjunctions, they may precede a bare particle" — as a rule | stated only for named particles | recorded as PARTIAL, with Ibn Jinnī's two usage examples |
| the closed list implies these items have no root | only pronouns are «غير مشتق»; الذي is given three radicals by Ibn Sayyidīh | the list now asserts **closed-class membership only** — which is sourced — and nothing about roots |

The pronominal set, the alif→yāʾ rule, the bare-preposition rule, the two kinds of tāʾ and the
preposition+pronoun unit are all **confirmed**. That is the whole of mode 1's machinery.

*(audits 2 and 4 — weak letters, preference principles — still running)*

### Audit 2 — weak letters (7 claims)

| claim | verdict | the text |
|---|---|---|
| the four classes أجوف/ناقص/مثال/لفيف | **SOURCED — but only in the Andalusians** | Ibn Mālik, *Tashīl*: «يقال للمعتل الفاء مثال، وللمعتل العين أجوف، وللمعتل اللام ناقص…»; Abū Ḥayyān, *Irtishāf*: «فصحيح، ومهموز، ومثال، وأجوف، ولفيف، ومنقوص». **Absent** from Sībawayh, Ibn Jinnī, al-Mubarrad, al-Mumtiʿ (which says المعتل العين/اللام) |
| ajwaf deletes its medial radical | **SOURCED** | Ibn Jinnī, *al-Khaṣāʾiṣ*: «لما سكنت عين فعلت ولامه حذفوا العين البتة فقالوا: قلت وبعت وخفت»; Ibn ʿUsfūr: «وتحذف العين لالتقاء الساكنين… فتقول: خفت وكدت وطلت» |
| naqis restores its final radical | **SOURCED behaviourally** | Ibn ʿUsfūr: «رددت الألف إلى أصلها من الياء أو الواو، نحو: رميت وغزوت». Caveats: the contrast is stated for **this position only** (naqis *does* delete before tāʾ al-taʾnīth: «رمت هند»), and كنيت appears only in lexica |
| vowels are "parts" of و/ي/ا | **SOURCED, near-verbatim** | Ibn Jinnī, *Sirr*: «فالفتحة بعض الألف، والكسرة بعض الياء، والضمة بعض الواو»; Ibn ʿUsfūr 1851; Abū Ḥayyān; al-Shāṭibī |
| al-qalb: و after kasra → ي | **SOURCED, near-verbatim** | Ibn ʿUsfūr: «الواو المكسورة بمنزلة الياء والواو… فيقولون: طويت طيا والأصل طويا… وسيد والأصل سيود» |
| alif = منقلبة عن واو/ياء, fixed by the pattern | **PARTIAL** | «الألف لا تكون أبدا أصلا. بل… منقلبة عن ياء أو واو» ✓. But **"the wazn tells you waw-vs-yāʾ" is NOT stated** — that is settled by الاشتقاق/lexical origin, and Abū Ḥayyān records a live dispute about it |
| prefer the no-ellipsis / al-aṣl reading | **SOURCED as a principle** | Sībawayh: «فالذي من الأصل أولى»; Abū Ḥayyān: «إذا دار الأمر إلى حذف… كان الرد أولى من الحذف»; Ibn ʿUsfūr: «فينبغي ألا يعدل عنها»; al-Shāṭibī: «الحمل على الأصل أولى». **Not absolute** — Ibn Jinnī, *Sirr*: «هذا أصل وإن قامت الدلالة عليه فإنه مرفوض، كما أن أصل قام: قوم» |

Also: **كنت is never used as an ajwaf exemplar** in any of these texts. The analysis is Ibn Jinnī's
in *Sirr Ṣināʿat al-Iʿrāb*: «وكذا كان القياس أن تقول في كنت: كوني، تحذف التاء… فترد الواو التي هي
عين الفعل من كنت». And al-Shāṭibī notes «الإعلال بالحذف قليل، ولذلك لا تجده مطردا».

### Audit 4 — the decision procedure (this is the important one)

**The tradition does supply a qualitative procedure, and Ibn ʿUsfūr states it as a list:**

> «أما الأدلة التي يعرف بها الزائد من الأصلي فهي: **الاشتقاق، والتصريف، والكثرة، واللزوم**،
> ولزوم حرف الزيادة البناء وكون الزيادة لمعنى، **والنظير، والخروج عن النظير، والدخول في أوسع
> البابين عند لزوم الخروج عن النظير**.» — *al-Mumtiʿ fī al-Taṣrīf*

The last item is explicitly a tie-breaker among competing analyses. The supporting principles:

| principle | source |
|---|---|
| **الاشتقاق والتصريف** first | Ibn ʿUsfūr's list order |
| **مستعمل beats مهمل** | al-ʿAyn, introduction: «يُكتب مُستعملها ويُلغى مُهملها» |
| **الحمل على الأكثر** | al-Shāṭibī: «الثابت في الأصول أن **الكثرة دليل الأصالة**»; «والحمل على الأكثر واجب». Ibn Jinnī: «والحكم على الأكثر لا على الأقل». Sībawayh: «أحمله على الأكثر» |
| **الأصل عدم الزيادة** | al-Shāṭibī: «**والأصل عدم الزيادة، فمن ادعاها فعليه الدليل**». Ibn ʿUsfūr: «ولا يحكم عليه بالزيادة إلا بدليل» |
| **الحمل على الأصل** | Ibn Jinnī: «ومتى أمكن تناول الكلمة على ظاهرها لم يجز العدول عن ذلك بها» |

The invented weights are **gone** — mode 2 now compares lexicographically over those principles.

### The limit the reorder exposed

Ranking by al-taṣrīf (does the reading generate the surface?) above attestation fixed `يشذ→شوذ`,
`كذا→كوذ` and `فمهما→فوم`; ranking attestation first fixed `كنت→كون`. **They cannot both win with a
binary مستعمل/مهمل record:**

```
كنت   baseline كنن  →  generate(كنن, فَعَلَ) = كنّ  →  strip = كن = stem  →  0 operations
       candidate كون →  generate(كون, فَعَلَ) = كان →  elide ا           →  1 operation
يشذ   baseline شذذ  →  generate(شذذ, يَفْعُلُ) = يشذّ →  strip = يشذ     →  0 operations
```

Both baselines generate the surface at zero cost, because the undiacritized text cannot distinguish
كنتُ (kuntu, كان+ت) from كَنْتُ (kantu, كنّ+ت). The sources resolve exactly this by **الكثرة** —
«والحمل على الأكثر» — which is a *frequency* judgement, not a binary one. My record has مستعمل /
مهمل and nothing in between, so it cannot break the tie.

**What that implies:** the next step is not another guard, it is a **usage count**. «الكثرة دليل
الأصالة» is computable — count the generated forms in the corpus — and it would settle both cases
the way the grammarians do. Until then, mode 2 stays off by default, and mode 1 is unaffected
(5.51% of roots, 40/40 correct, and it does not touch this machinery at all).
