# FINDINGS — what this project currently knows

Consolidated 2026-10-03. Every number here is measured; every claim has a comparator.
Read `STATE.md` for the retracted errors, `NEXT.md` for what to do next.

---

## 1. The main result: early root attachment beats late, and the old "ceiling" was protection

**Matched trunk training, 20,000 steps, batch 32, LR 1e-3 — only the attachment layer differs.**

```
arm              pathway          layers   ALL      NOVEL    ridge val_all  ridge unseen
C EARLYROOT_C    residual+bias    0-23     18.99 %  19.22 %  90.64 %        80.16 %
E RESIDUAL_R     residual only    0-23     18.94 %  19.17 %  90.44 %        79.53 %
X LATE_X         residual+bias    20-23    15.43 %  15.53 %  89.51 %        78.07 %
A FLOOR_A_V2     none             —         8.04 %   7.87 %  87.37 %        74.73 %
D SCOREBIAS_D    bias only        0-23      7.83 %   7.71 %  87.36 %        74.96 %
```

**Early beats late: +3.56 pp ALL, +3.69 pp NOVEL.**

### The reversal

```
19.53 %   late (20-23), 0.01x LR, trunk FROZEN for most of training   <- previously read as a ceiling
15.43 %   late (20-23), 1.0x  LR, trunk TRAINED                       <- X
18.99 %   early (0-23), 1.0x  LR, trunk TRAINED                       <- C
```

**The same late schedule scores 4.10 pp worse when the trunk is actually trained.** So the frozen
trunk was never a handicap — **it was protection.**

Full-rate training moves the trunk away from the representation a late pathway needs, and layers
20–23 cannot recover it, because by then the root signal is gone. Early attachment can, because the
root signal is still present at the input stage and the pathway holds it on the way up.

> **The 19.53 % was never about early-vs-late placement. It was about what full-rate training does
> to the representation. Attach early and you reach ~19 % with no freezing at all.**

### A normally-trained trunk is a floor — and worse than the trunk it started from

```
                      ridge val_all    ridge unseen
released trunk           92.34 %         83.92 %
A (no pathway, 20k)      87.37 %         74.73 %    -4.97 / -9.19 pp
C (pathway off at probe) 90.02 %         78.87 %
C (pathway active)       90.64 %         80.16 %    -1.70 / -3.76 pp
```

**Full-rate training damages root decodability**, badly on unseen words. **The early root pathway
roughly halves the damage**, and it helps even with the pathway switched **off** at probe time —
so the benefit is **in the weights, not the readout**.

> **The pathway is not a readout. It is a scaffold the representation is built around.**

This also explains the ablation result: `RCA_OFF` = 6.38 % is *below* arm A's 8.04 %. A trunk
trained **with** the pathway and then ablated is worse than a trunk trained without one. The
representation came to **depend** on it. Load-bearing, not additive.

---

## 2. The score bias is inert — five measurements, four from separate runs

```
D (bias alone) 7.83 %  vs  A (nothing) 8.04 %         -0.21 pp
D ridge vs A ridge                                     within 0.23 pp
C with the bias ablated in place:  18.99 -> 18.81 %    -0.18 pp
C (both) vs E (residual only):     18.99 vs 18.94 %    -0.05 pp
C with the RESIDUAL ablated:       18.99 ->  6.38 %    -12.61 pp
X with the RESIDUAL ablated:       15.43 ->  7.18 %    -8.25 pp
```

The native score-bias path was `stream_mix[1].grad == 0.0` **exactly, in all 24 layers**, on the old
trunk (element 0 live 24/24). This session gave it gradients **for the first time in the project's
history** — and it still changes nothing measurable.

**The entire early-root effect is the 57.86 M residual injector.** The 11.01 M score-bias path is
dead: demonstrated, not assumed. (`RESIDUAL_R_30K` is queued to confirm inertness holds at 30k
rather than the bias merely being slow.)

---

## 3. The morphosyntactic premise: the transmutation is what creates the root signal

```
root identity linearly decodable from h_t
  original Qwen2.5-0.5B           29.30 %   unseen 23.34 %    (marginal 0.40 %)
  the transmuted trunk            92.34 %   unseen 83.92 %
  local isolated probe:  layer 4  21.83 %  ->  layer 23  13.00 %   DECAYS with depth
  the transmuted trunk is MONOTONE UPWARD to layer 23
```

**A raw base carries root identity at 29.30 % — real, 73× its marginal, but 3.15× short — and it
*loses* root information with depth** as it organises for next-token prediction over a 152k BPE
vocabulary. **The morphemic adaptation is what created a root-preserving representation.**

A cross-attention whose keys/values *are* the root table can only **re-select** among root
embeddings using query weights from `h`; it cannot create root information `h` lacks. So the
transmutation is not preprocessing — it is the step that makes the pathway able to work at all.

---

## 4. al-qiyās: a computed rule beats training at the project's own claim

```
neural compositional decoder, held-out root, top-1/top-5    0 / 3,588   (0.00 %)
computed qiyas, inverse root ID from the surface            3,031 / 3,588 = 84.48 %
```

### And its ʿilla is now correctly named

The required condition set is **two**, not four — both authors, verbatim:

```
mundabit   REQUIRED       [Mustasfa:12851]  «فلم ينضبط باسم البر فلا بد من ضابط...»
muttarid   REQUIRED       [Mustasfa:13254]  «إذ لو كانت لاطردت ووجد الحكم حيث وجدت...»
                          [Mahsul:14389]    «لأن الطرد واجب في العلل والعكس غير واجب فيها»
mun'akis   NOT REQUIRED   [Mustasfa:12599]  «العكس ليس بشرط في العلل الشرعية»
muta'addi  NOT REQUIRED   [Mustasfa:13507]  «مسألة العلة القاصرة صحيحة... فالتعدية فرع الصحة»
                          [Mahsul:15982]    «مذهب الشافعي أن يجوز التعليل بالعلة القاصرة... لزم الدور»
```

Under the corrected set, **`identity` (ʿilla = the root) is ACCEPTED** and is the best on **both**
required measures (purity 1.000000, iṭṭirād 0.996278). The earlier rejection used *mutaʿaddī*,
which is not required.

**But accepting it changes nothing, and the reason is the finding:**

```
strict7 (the root's CLASS)     Task A 84.2531 %   Task B 84.4760 %
identity + engine lattice      IDENTICAL -- because identity FIRES 0x
identity ISOLATED (root alone) 0 / 3,588 = 0.0000 %
```

**Structural cause: the ḥukm (root → surface alignment) is wazn-specific.** A root carries no ruling
to a different pattern. Its purity of 1.000000 is one-root-per-cell memorisation — 25,042 cells,
10,066 singletons, **0 cells with more than one distinct alignment** — and its 397 iṭṭirād failures
are all singleton-cell leave-one-out collapses, with **0 genuine exception cells**.

> **الجذر أصلٌ، وعلّته قاصرة صحيحة** — the root is a **valid but confined** (*qāṣira*) cause. The
> cause that **extends** is a determinate property of it: its **phonological class**.

**The 84.48 % belongs to the root's class, not to the root as such.** That is the correct statement
of the project's claim, in the tradition's own terms.

---

## 5. Grammar grounding — verified, and one class of problem that is *not* fabrication

284 citation claims audited: 74.3 % exact-or-near, **VERBATIM pass 60/60**, locators 56/63 exact,
**0 fabricated locators**. Rules: **FAITHFUL 6 · OVERREACH 1 · UNSOURCED 0 · CONTRADICTS 0 ·
UNCHECKABLE 2**.

**A second class found later, and it is not dishonesty: edition ambiguity.** When a text is held in
more than one edition, a locator that names only a line number **manufactures false positives**.
al-Īḍāḥ alone differs by **+92 / +142 / +148** between the project's sister edition (1,701 lines)
and OpenITI's (1,864 lines). Three "citation defect" flags were raised this session and **all three
were retracted** — the searchers had looked in the wrong file. **Locators must name the edition.**

---

## 6. What was eliminated — negative results, stated plainly

```
cardinality     9,490 -> 500 root classes: lift stayed 1.86x (no change)
depth           4 -> 8 RCA layers:        +0.13 pp
width           896 -> 1792:              +0.22 pp (best acc@5 and CE_z of any arm)
discrete n-gram pathway:                  6.06 %, declining
clamp            dead code: 0 / 456 attention calls carry root_ids
logit gauge      GATE2 vs CONTROL t = 0.21
conditioning branch: harmful when live
prefix stream    carried zero information; fix +0.43 pp / 0.70 pp causal
score bias       five measurements, inert (see §2)
Fisher split     corr(F_ret, F_syn) = 0.9959 -- the essential/accidental split does not exist
EWC on text Fisher:  DODGEABLE -- Arm P's own number looked healthy (16.34 %) while its trunk
                     was destroyed (head-independent 2.676 %)
EWC on CAPABILITY Fisher: the trunk held (6.672 % vs 6.641 % untouched) at the 0.1x LR that
                     otherwise destroys it -- preservation by REDIRECTING the update
                     (rho 8.434 -> 0.279 at unchanged displacement), not by braking it
```

---

## 7. The thesis — STILL NOT TESTED

**The claim:** one root embedding shared across كتب / كاتب / مكتوب / كتابة generalises to **unseen
derivational forms**.

```
attempt 1   structurally impossible -- a softmax over root types cannot emit an unseen root
            (proved at 4,117 positions)
attempt 2   ran at 0.141 epochs -- 0 / 3,588
qiyas       demonstrates it as a COMPUTED CAPACITY (84.48 % vs 0.00 %)
```

**It has never been demonstrated as a *learned* capacity.** This needs a **root-disjoint** split and
retraining with held-out roots **removed** from the training set (~6 h). Everything else in this
document is scaffolding around that question.

---

## 8. Architecture fixes — all verified

```
constants         num_roots 9015 -> 9490, num_awzan 128 -> 142, vocab 10052
                  one root-id space, injective, 48 stale-SPACE tables dropped
root source       stale (9015,64) -> morphemic_embed.root_embed (9490,448)
pathway           layers 20-23 post-layer hook -> ALL 24 layers, input stage upward
init-equivalence  gate 0 bit_equal to no pathway, max|d| = 0.0 (torch.equal)
isolation         gate 1 with roots rolled, trunk input FIXED -> output changes (reads roots)
checkpoint load   RootformerNRMT vs CausalLM key spaces: naive load = 555/520 -> LOADS NOTHING
152 M orphans     live hooks on layers 1/11/14, no checkpoint, no model.parameters();
                  perturbation-proved detachable
attention         eager [B,14,T,T] -> SDPA, fused kernel 58 MB vs 224 MB
                  (QK^T was running fp32 while V was bf16; bf16 shipped, fp64 oracle)
in-place bug      root_cross_attn.py:136 clamp_ mutates the CALLER's tensor -- harmless while
                  morphemic_embed is frozen, fatal once the input stage trains
disk              checkpoints to /tmp (pod 50 G overlay); /workspace is MooseFS (NETWORK) and
                  its "466 T free" is the CLUSTER's. ENSOSPC killed two arms at a write
liveness          201 parameters causally inert; 193 with .grad is None
```
