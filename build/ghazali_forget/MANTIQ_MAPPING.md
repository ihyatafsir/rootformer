# al-Ghazālī's manṭiq → numerical proxies: what is operational, what I dropped

The rule applied here is the brief's own: **a classical term earns its place only if it can be
given a number that changes a decision in the code.** Everything below is stated next to the
measurement that either supports or kills it. Numbers are from this study's own artefacts on the
pod (`/workspace/ghazali_forget/`).

## 0. What I verified about the sources, and what I did not

Verified (URLs checked):

- al-Ghazālī, *al-Mustaṣfā min ʿilm al-uṣūl*, carries the chapter conventionally titled
  **al-taʿāruḍ wa-l-tarjīḥ** — the printed pages consulted are 374–375 of the Abū ʿAbd al-Salām
  ʿAbd al-Shāfī printing ([p. 374](https://ablibrary.net/book_content/b/4388/374?lang=fa),
  [p. 375](https://ablibrary.net/book_content/b/4388/375)); a study devoted to exactly this topic in
  that work is ["Gazzâlî'ye Göre Delillerin Teâruzu ve Tercih Sebepleri (el-Mustasfâ Bağlamında)"](http://yerbilimleri.cumhuriyet.edu.tr/en/pub/ulum/issue/79192/1287580#2).
- al-Ghazālī's definition of the conflict itself — «التعارض هو التناقض» — is quoted from him in an
  uṣūl source ([ekb.troweb.app](https://ekb.troweb.app/api/v1/blob/63cdfd23478a0a0007848407?download=true#20#4)).
- The dependency rule of the pair: **al-tarjīḥ has no subject-matter except where al-taʿāruḍ
  exists** ( «والترجيح لا يكون إلا عند وجود التعارض» ) — stated as the standard position in an uṣūl
  textbook ([fshs.uit.ac.ma](https://fshs.uit.ac.ma/images/cours%20en%20ligne/S2S4S62020/EISLAMIQUE/S4/%d8%a3%d8%b5%d9%88%d9%84%20%d8%a7%d9%84%d9%81%d9%82%d9%87%2012.pdf)).
  **This one has a direct consequence for the method and is kept.**
- *Miʿyār al-ʿIlm* is Ghazālī's logic handbook ([PDF](https://rissc.jo/wp-content/uploads/2024/09/Miyar_Ghazali_28.08.24_WebLR.pdf)).
- The fourfold vocabulary **jins / nawʿ / faṣl / ʿaraḍ ʿāmm / khāṣṣa** in the Avicennian tradition
  Ghazālī transmits, with the canonical worked example ("colour is a *nawʿ* for the dense, a *jins*
  for black and white, a *khāṣṣa* for body, an *ʿaraḍ ʿāmm* for man"), appears verbatim in the
  commentary tradition ([Sharḥ Ḥikmat al-ʿAyn](https://archive.org/download/sharhHikmaRazi/sharhHikmaRazi_text.pdf#95#11)).

**Not claimed:** I found no place where Ghazālī assigns numerical weights, Fisher information, or
anything resembling a curvature criterion. The transfer from predication-in-a-definition to
parameter importance is **mine, and it is analogical**. Where the analogy fails to yield a number,
I have dropped it rather than dress it in a citation — that is the point of this file.

## 1. Operational (kept — each has a number that changes a decision)

| manṭiq term | numerical proxy actually used | measured | decision it drives |
|---|---|---|---|
| **al-dhātiyyāt** (essential) | `F_head,i`: diagonal empirical Fisher of the retained **capability** loss (FIX head root CE on the aligned held-out cache) | sum F = 6.431 over 62.25 M params; acc@1 6.52 % at the anchor | enters the penalty as `F_i` |
| **al-ʿaraḍiyyāt** (accidental) | `F_text,i`: diagonal Fisher of the *text* distributions (heritage LM and/or synthesis corpus) | sum F = 3917.06 (retained) / 4134.91 (synthesis) | enters the penalty as `F_i` in arm P |
| **al-taʿāruḍ** (conflict) | (i) sign test on the *same* trunk displacement; (ii) displacement ratio `‖ΔW‖/‖W‖ / d*` with `d* = 0.035` measured | (i) Run A's displacement raised the capability root CE by **+3.5321** while **lowering** the heritage LM CE by **−5.4621**; (ii) 3.4585–3.7765 % / 3.5 % ≈ 1.0 | sets `d*`; makes the whole penalty non-vacuous |
| **al-tarjīḥ** (preference) | the pre-registered rule `λ* = κ·L_task / (d*²·Σ F_i θ0_i²)`, κ = 1 | λ* = **2421.79** (`F_ret`), **953215** (`F_head`) | the actual penalty coefficient |
| **al-jins / al-faṣl** (genus / differentia) | the measured *intersection* and *difference* of profiles: top-decile overlap `F_ret ∩ F_syn`; top-decile overlap `F_head ∩ F_text` | 0.889 (intersection: the shared text-competence structure is most of the mass) vs 0.556 (differentia: the capability-specific tensors are a minority) | tells you *how much* of the trunk is capability-specific, hence how much room a selective penalty has |

The last row matters: it is the only part of the classical apparatus that produced a **new fact**
here rather than a label — namely that the "differentia" (tensors that the measured capability
needs and the text distributions do not) is a minority, ~half the top decile, not the bulk.

## 2. Decorative — dropped, with the reason

1. **The five predicables as a five-way partition** (jins / nawʿ / faṣl / khāṣṣa / ʿaraḍ ʿāmm).
   Dropped: nothing in a parameter vector distinguishes *nawʿ* from *jins*, or *khāṣṣa* from
   *ʿaraḍ ʿāmm*. Only the binary dhātī/ʿaraḍī survives, and only as a continuous score `F_i`.
2. **al-lāzim (the concomitant) as a category distinct from ʿaraḍ ʿāmm.** Dropped: no proxy.
3. **"Fisher is the māhiyya (quiddity)".** Dropped **by measurement**, not by taste:
   - against the retained-LM loss the second-order term `½ Σ F_ret ΔW²` = **+0.0626** while the
     measured change is **−5.4621** — *wrong sign*;
   - against the capability loss `½ Σ F_head ΔW²` = **8.3 × 10⁻⁵** while the measured change is
     **+3.5321** — off by ~4 × 10⁴.
   So the Fisher is at best a **ranking** heuristic here (and its ranking is the only one of the
   three that tracks the damage: tensor-level Spearman +0.5907 vs +0.1450 and +0.1423). Calling it
   the essence of the capability would be exactly the unearned citation the project already caught
   once. It is local curvature of a chosen loss, and it is measured to be a poor local model at the
   displacement scale that actually matters.
4. **al-taṣawwur / al-taṣdīq (conceiving / assenting).** Dropped: no proxy.
5. **al-taʿāruḍ as logical contradiction between two propositions.** Dropped in that form (my two
   objectives are not propositions and are not contradictory — they are jointly satisfiable in
   principle). Only the engineering proxy survives: gradient interference and the displacement
   ratio. Note the one classical rule that *does* transfer: **tarjīḥ has no subject-matter without
   taʿāruḍ** — and correspondingly, if the new objective's update had no component along
   high-`F_i` directions, λ would be irrelevant. The measured conflict is what makes it relevant.

## 3. The headline honesty item

The brief's own proposal — ESSENTIAL = high Fisher on the *retained linguistic* distribution,
ACCIDENTAL = low Fisher there — is **falsified as stated**, because the retained-linguistic and
synthesis profiles are the same profile:

    F_ret vs F_syn:  Pearson +0.9959, Spearman +0.9951 (per parameter)
                     tensor-level Spearman +0.9961, top-decile overlap 0.889

and because the retained-linguistic Fisher is *anti-correlated with the capability damage*
(corr(ΔW², F_ret) = −0.0137; the damaging displacement improved the LM readout it was supposed to
protect). The distinction that **is** measurable is a different one: the distribution we *train on*
versus the loss we *measure the capability with* (`F_head` vs `F_text`: Pearson +0.19,
Spearman +0.60, top-decile overlap 0.556). That substitution is a change to the brief's criterion,
and it is reported as such rather than presented as if it had been the plan.

## 4. The single measurement that earns the mapping

If one number justifies keeping the classical vocabulary at all, it is this one (representative CPU
protocol, §5a of `RESULTS.md`; anchor FIX-head acc@1 6.572 %, retained behaviour measured after
fine-tuning the trunk at its real LR):

| arm | rel. displacement | ρ = F-alignment of the update | retained capability |
|---|---|---|---|
| λ = 0 | 0.6055 % | **8.434** | **65.3 %** |
| λ = 2421.79 (pre-registered) | 0.5718 % | **0.279** | **90.7 %** |

The two updates are of **almost identical size**. What the penalty changes is not how far the trunk
moves but **which subspace it moves in**: it pushes the update out of the Fisher-heavy directions
(ρ falls 30×), and that alone converts a 35 % loss of the retained capability into a 9 % loss. That
is precisely the claim the dhātī/ʿaraḍī distinction was recruited to make — essential structure
resists, accidental structure is surrendered — expressed as two numbers that can be recomputed from
the artefacts. The mapping is kept on this evidence, not on the authority of the text, and it would
be dropped again if this number came out at ρ ≈ 1 in both rows.

## 5. Refinement, same session — **and a correction to §1's jins/faṣl row**

§1 reported the `F_head ∩ F_ret` top-decile overlap as 0.556 and read it as "the capability-specific
differentia is a minority". That reading is too strong. Reading out the actual tensors:

| top-k of each ranking | overlap |
|---|---|
| 5 | **80 %** (4 of 5) |
| 9 | 56 % |
| 10 | 50 % |
| 18 | 67 % |
| 46 | 76 % |

Both profiles' top 5 is the same structure — the four `stream_mix` scalars (layers 20–23) plus
`L23.v_proj.weight`. Spearman *within* `F_head`'s top-20 tensors is **+0.677**, essentially the same
as over all 92 (+0.682). What actually differs is narrower: `F_ret`'s top-10 wrongly promotes three
tensors the capability does not care about (`L20/L21.self_attn.coverage_threshold`, at `F_head` ranks
60 and 68, and one at rank 86), and it under-ranks `L23.stream_mix` (its rank 35 in `F_ret` versus 1
in `F_head`).

So the honest statement of the genus/differentia split is: **the two profiles agree strongly at the
very top and diverge in the middle of the ranking.** That is exactly why `F_ret`'s penalty still
worked in §5a of `RESULTS.md` — it does capture the top of the essential structure — and it is why
the brief's *retained-vs-synthesis* split being empty (ρ = 0.996) does not make `F_ret` useless. It
is a good proxy at the top, a poor one in the middle, and it is not the metric the brief claimed it
was.

### What the essential structure actually is (measurable, named)

Per-parameter, the most sensitive knobs in the trunk are **2-element `stream_mix` scalars** in the
IshtiqaqAttention of layers 20–23. In aggregate mass (F × numel) it is the big projections:

| tensor | F_head (mean) | aggregate share |
|---|---|---|
| `layers.{20,21,22,23}.self_attn.stream_mix` (2 params each) | 1.1e−05 … 4.7e−05 | ~0 % |
| `layers.{20..23}.self_attn.v_proj.weight` (114 688 each) | 3.0e−07 … 9.9e−07 | largest cluster |
| `layers.{20..23}.self_attn.o_proj.weight`, `mlp.down_proj.weight`, `mlp.gate_proj.weight` | 1.7e−07 … 4.3e−07 | remainder |
| top 18 of 92 tensors combined | – | **56.8 % of the capability Fisher mass** |

So "essential structure" here is not a vague property: it is 18 named tensors holding 56.8 % of the
capability's Fisher mass, led by a handful of scalar stream-mixing gates and the value/output
projections of the top four layers. A targeted protection of those 18 tensors would cover more than
half the measured importance for a small fraction of the parameter count — which is the practical
form the al-dhātiyyāt / al-ʿaraḍiyyāt distinction takes in this model.
