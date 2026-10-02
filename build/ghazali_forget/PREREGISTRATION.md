# Pre-registration — al-tarjīḥ (the preference rule), fixed BEFORE the 0.1× run

Author: ghazali_forget agent · pod `g6exduq0bd17z8` (213.173.104.76:46758) · written 2026-10-02,
before any GPU training arm of this study was launched. Every number below comes from an artefact
that existed before the run.

## 1. What is being decided

Two claims are on the table, and they are separable.

**C1 (the brief's claim).** *Retained-structure importance* (diagonal Fisher on the retained
linguistic distribution) and *abandoned-objective importance* (diagonal Fisher on the v19.2
synthesis objective) are different profiles, so a Fisher-weighted penalty can protect the first
while letting the second be forgotten selectively.

**C2 (the operational claim).** Whichever importance profile is used, a penalty of the form
`λ · Σ_i F_i (θ_i − θ0_i)²` at the 0.1× trunk LR that previously destroyed the trunk keeps the
retained capability alive.

## 2. The rule (stated in advance, not chosen after seeing results)

> The new objective may move the trunk freely until its displacement reaches the scale that was
> **measured** to destroy the retained capability; at that scale the retained-structure penalty is
> worth one full unit of task loss. Beyond it, retained structure wins.

Arithmetic, with no free parameter other than the measured inputs:

    S1 = Σ_i F_i · θ0_i²                       per unit relative displacement
    S  = d*² · S1                              penalty per unit λ at displacement d*
    λ* = κ · L_task / S                        κ = 1 (primary, pre-registered)
    d*    = 0.035     MEASURED: ||ΔW||/||W|| = 3.4585/3.6328/3.7765/3.5856 % on layers 20-23
                      of the arm that collapsed (reproduced here from its saved payload)
    L_task= 11.093619 MEASURED: step-1 task loss of that same arm, from its own trace
    F, θ0 = the profiles below and the pretrained trunk in the penalised scope (layers 20-23)

Measured values of λ* by profile (identical rule, different F):

| profile | what it is | S1 | λ* (κ=1) |
|---|---|---|---|
| `F_ret`  | next-token CE, held-out heritage Arabic | 3.73939 | **2421.79** |
| `F_syn`  | next-token CE, v19.2 synthesis corpus  | 3.95235 | 2291.30 |
| `F_head` | FIX head root CE, aligned held-out cache (the measured capability) | 0.0095005 | **953215** |

κ = 0.1 and κ = 10 are a pre-stated *sensitivity band* only; they are not selection candidates.

## 3. Arms, fixed in advance

Both at the trunk LR that previously collapsed (`--trunk-lr-scale 0.1`), both 20 000 steps,
batch 32, `--eval-every 1000`, on the shipped aligned cache.

| arm | config | penalty | why |
|---|---|---|---|
| **P** `EWC_A_RUNA_LRET` | Run A verbatim (`--root-cross-attn top4 --unfreeze-last 4 --rca-lr-scale 1.0`) | `F_ret`, λ = 2421.79 | the exact one-variable replication of the measured collapse |
| **Q** `EWC_B_A2_LHEAD` | A2's stabilised config (`--rca-lr-scale 0.3 --rca-out-norm --rca-dropout 0.1`) at 0.1× trunk LR | `F_head`, λ = 953215 | the best available shot at the stated success bar |

Arm P is the pre-registered decisive test. Arm Q is labelled exploratory: it changes two things
relative to A2 (LR and penalty), so it can support but not by itself establish C2.

## 4. Endpoints against which the arms are scored (all measured earlier, same evaluator)

| reference | ALL_val acc@1 | provenance |
|---|---|---|
| frozen baseline, CTL converged | 16.54 % at step 15 000 (SIGTERMed there; flat 16.5 % from step 10 000) | `/workspace/root_attn_ctl/` |
| healthy 0.01× (A2, complete) | **19.53 %** | `results_RCA_UNFREEZE_A2.json` |
| collapsed 0.1× (Run A, FIX head on its trunk, RCA off) | **2.411 %** | `REPORT_root_attn_ctl.md` |
| FIX head on the *released* trunk (same measure) | **6.635 %** | same |
| Run A's own arm accuracy at step 5 000 | 9.900 % | same |

Additional head-independent measures reproduced on CPU by this study before the run, to prove the
scoring pipeline matches the published numbers:

| measure | this study (CPU fp32, live) | published |
|---|---|---|
| FIX head + released trunk | 6.641 % / acc@5 12.550 % / ce_z 8.0998 | 6.635 / 12.603 / 8.0997 |
| FIX head + Run A trunk, RCA off | 2.433 % / 5.257 % / ce_z 9.5878 | 2.411 / 5.241 / 9.5876 |
| ‖ΔW‖/‖W‖ layers 20-23 | 3.4585/3.6328/3.7765/3.5856 % | 3.46/3.63/3.78/3.59 |

## 5. Pre-stated quantitative expectation

From the measured task-gradient scale (`grad_norm_trunk` = 1.46 at step 1 over 62.25M params) and
the calibrated λ, the penalty balances the task gradient at a *relative* trunk displacement of

    d_eq ≈ grad_task_rms / (2 λ F̄ θ_rms) ≈ 1.85e-4 / (0.0596) ≈ 0.3 %   (F_ret)
    d_eq ≈ 1.85e-4 / (0.0385) ≈ 0.5 %                                   (F_head)

i.e. roughly 10× BELOW the measured 3.5 % collapse scale. **Pre-stated prediction: the trunk will
be effectively frozen, so both arms should land near the frozen-trunk regime (CTL, 16.5 %) rather
than collapse; the stated success bar of ≥ 19.5 % requires the root cross-attention pathway to
carry the difference.** If an arm lands at 16.5 % ± 1, the penalty worked but did not close the
cliff; that is a partial result and will be reported as such.

## 6. Falsifiers, fixed in advance

- **F1 — the criterion itself.** If `corr(F_ret, F_syn) ≈ 1` (say Spearman > 0.95 and top-decile
  overlap > 0.9 at both parameter and tensor level) then C1 is falsified: there is no selective
  distinction to exploit, and any benefit is ordinary anchoring, not essential/accidental
  separation. **Already measured before the run: parameter-level Pearson +0.9959, Spearman +0.9951,
  tensor-level Spearman +0.9961, top-decile overlap 0.889 → C1 IS FALSIFIED.**
  Corollary, also already measured: the retained-LM Fisher does not model the measured damage
  (corr(ΔW², F_ret) = −0.0137; the Taylor term predicts +0.0626 against a *measured* heritage-CE
  change of **−5.4621**, wrong sign) because the released trunk's tied-embedding LM readout is
  miscalibrated (mean CE 15.99 vs ln(10052) = 9.22 uniform) and the damaging displacement
  *improved* it. So the brief's literal proxy is not merely redundant, it is anti-correlated with
  the capability.
- **F2 — the mechanism.** If at λ* the head-independent measure (FIX head on the EWC trunk, RCA
  off) is ≤ 3 % — i.e. no better than Run A's 2.411 %/2.433 % — then the weighted anchoring did not
  prevent the trunk damage, whatever the arm's own accuracy did.
- **F3 — the utility.** If the arm's own ALL_val acc@1 stays ≤ Run A's 9.900 % at step 5 000, the
  penalty bought protection at the price of the new capability: the cliff moved, it did not close.
- **Success (pre-registered).** Arm ALL_val acc@1 ≥ 19.5 % at any eval ≥ step 15 000 **and**
  head-independent FIX-head acc@1 ≥ 6.0 %. Anything in between is reported as a partial result,
  not rounded up.

## 7. What would make me withdraw a conclusion

If the arms' own numbers disagree with the head-independent measure in direction (e.g. a healthy
own-accuracy with a destroyed trunk), the head-independent measure wins, because it is the only one
in this project that is independent of the head being trained — that is the lesson of the
evaluator bug recorded in `REPORT_root_attn_ctl.md` §3.

---

## AMENDMENT 1 — 2026-10-02T18:05Z, **before any GPU arm of this study was launched**

The CPU endpoint test specified in §5 was run (60 steps, trunk lr 3e-3 on the synthesis-corpus LM
loss, 120 val windows, anchor FIX-head acc@1 6.572 %, FIX-head-on-trunk measured after training):

| profile | λ | rel. disp | FIX head acc@1 after | Δ vs anchor |
|---|---|---|---|---|
| (anchor, no training) | – | – | 6.572 % | – |
| `F_ret` | 0 | 8.851 % | 0.215 % | −6.357 |
| `F_ret` | 2421.79 (κ=1, **pre-registered**) | 4.113 % | 0.151 % | −6.421 |
| `F_ret` | 24217.9 (κ=10) | 2.248 % | 0.146 % | −6.426 |
| `F_ret` | 242179 (κ=100) | 1.421 % | 0.398 % | −6.174 |
| `F_ret` | 2.42179e6 (κ=1000) | 0.729 % | **3.644 %** | −2.928 |
| `F_head` | 953215 (κ=1) | 1.599 % | 0.636 % | −5.936 |
| `F_head` | 9.53215e6 (κ=10) | 0.941 % | **5.366 %** | −1.206 |

Two conclusions, both taken from CPU evidence only:

1. **The pre-registered κ=1 does not reproduce the "λ large preserves" endpoint.** The rule's λ is
   too weak by 1–3 orders of magnitude in this protocol. The brief's requirement ("a method that
   cannot reproduce both endpoints is not yet a method") is therefore **not met at κ=1**, and this
   will be reported as a negative result rather than repaired by re-tuning κ after seeing a GPU
   number.
2. The *capability* profile `F_head` reaches near-preservation at κ=10, whereas the *text* profile
   `F_ret` needs κ=1000 for a weaker result — i.e. **the profile matters, but not the profile the
   brief proposed.** Per unit of λ-rule, the capability Fisher buys ~100× more protection.

**Decision, declared here before the GPU result exists:**

- **Arm P stays at the pre-registered κ=1 on `F_ret`** (λ = 2421.79). It is the test of the rule as
  written, and its expected failure is itself the answer to the brief's decisive question.
- **Arm Q is amended to κ=10 on `F_head`** (λ = 9 532 150), because the CPU scan shows κ=1 on that
  profile does not preserve while κ=10 does.

This is a declared deviation, not a post-hoc choice: the arm's outcome is unknown at the time of
writing, and the amendment is reasoned from a different, already-complete experiment run at a 30×
harsher trunk LR. The falsifier is unchanged: if arm Q at κ=10 still lands at or below Run A's
9.900 % own-arm accuracy, the cliff is not closed.
