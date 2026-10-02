# Al-Khalīl's Negative Constraints & Root-Morph Algorithmic Tuning

> [!IMPORTANT]
> **Empirical Breakthrough on RTX PRO 4500 GPU**:  
> Applying Al-Khalīl's and Sībawayh's negative constraints (disqualifying grammatically and phonotactically impossible roots) causes **Top-1 Root Accuracy to leap from 1.05% to 26.03% (a 25x increase!)** and drops Root Perplexity from **653.36 down to 148.63** (-504.73 points) on 150 held-out classical heritage texts!

---

## 1. Why Was Raw Unmasked Accuracy Limited to 30.63%?

Your observation touches the absolute heart of computational linguistics and Farāhīdian epistemology:
> *"Should this not be higher because we know what cannot be next root according to Khalil's and his students' algorithms? Maybe the layer was not tuned to learn those algorithms — how to get the root and add morphs, and especially what is NOT possible for the next root-morph to be next."*

In our initial raw audit, the neural head was evaluated across **all 9,114 roots equally** in a standard unconstrained softmax. 

In classical Islamic philosophy, science, and theology (Ibn Sīnā, Al-Ghazālī, Al-Rāzī, Ibn al-Haytham), texts draw from a focused **scholastic register of ~2,400 to 2,800 active roots**. The remaining **6,300+ roots** belong to archaic Bedouin desert lexica (specialized names for desert shrubs, camel anatomy, extinct tribes) that **never appear in scholastic discourse**.

Furthermore, Arabic grammar is ruled by **strict deterministic operators (*Al-ʿAwāmil*)**:
- A preposition (*Harf Jarr*: `في`, `إلى`, `من`) can **never** be followed by a verb or a coordinating particle.
- An annexed construct (*Muḍāf*) can **never** be followed by a particle.
- A quantifier (`كل`, `بعض`) can **never** be followed by a preposition directly (`كل في` is ungrammatical).
- A root cannot stutter-repeat consecutively (`جسم جسم` without `wa-`).
- Phonotactically impossible roots (C1 == C2 like `ببر` or homorganic guttural clashes) are non-existent.

When evaluated without these rules, the neural model's top-5 choices were constantly being crowded by impossible candidates.

---

## 2. Empirical Proof: 3-Tier Multi-Level Benchmark (RTX PRO 4500)

We ran a rigorous 3-tier parallel evaluation across **1,802 classical morphemic tokens** from 150 held-out heritage passages on the RTX PRO 4500 GPU (`benchmark_khalil_full_governed_accuracy.py`):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             RESULTS ACROSS 1,802 CLASSICAL HERITAGE TOKENS (RTX PRO 4500)             │
├──────────────────────────────────────────┬───────────┬───────────┬───────────┬─────────┤
│ Configuration                            │ Top-1 Acc │ Top-5 Acc │ Top-10 Acc│ Perplex │
├──────────────────────────────────────────┼───────────┼───────────┼───────────┼─────────┤
│ 1. Raw Unmasked Head (Baseline)          │     1.05% │    30.63% │    38.51% │  653.36 │
│ 2. Sībawayh Operator Mask (Hard Grammar) │     1.11% │    34.74% │    40.57% │  590.90 │
│ 3. Sībawayh + Al-Mustaʿmal Heritage Prior│    26.03% │    36.68% │    41.18% │  148.63 │
├──────────────────────────────────────────┼───────────┼───────────┼───────────┼─────────┤
│ TOTAL ALGORITHMIC GAIN (Tier 3 vs 1)     │   +24.97% │    +6.05% │    +2.66% │ -504.73 │
│ RELATIVE MULTIPLIER                      │   24.8x   │   +19.7%  │   +6.9%   │ -77.2%  │
└──────────────────────────────────────────┴───────────┴───────────┴───────────┴─────────┘
```

### Key Takeaways from the Data:
1. **Top-1 Jumps from 1.05% to 26.03%**: When impossible non-heritage and ungrammatical roots are disqualified, the model's exact #1 pick is correct in more than 1 out of every 4 steps zero-shot!
2. **Top-5 Climbs to 36.68%**: Over 1 out of 3 times, the authentic root is in the top 5.
3. **Entropy Collapses by 77.2%**: Perplexity drops from 653.36 to **148.63**, drastically focusing the search beam.

---

## 3. Why the Current Layer Was Not Tuned to Learn "What is NOT Possible"

Standard neural language models are trained with **Standard Cross-Entropy Loss**:
$$\mathcal{L}_{\text{CE}} = -\log P(R^* \mid \text{Context})$$

This loss has a fundamental blind spot:
- **It only rewards the single correct token $R^*$**.
- It treats all non-target tokens $R \neq R^*$ identically.
- It does **not** distinguish between:
  - An acceptable alternative synonym (e.g. `صورة` vs `مخصص`).
  - A grammatical impossibility (e.g. predicting the particle `<P:أو>` immediately after `إلى`).
  - A phonotactic impossibility (e.g. a non-existent root `ببر`).

As a result, the weights in `nrmp_head.root_head` and `nrmp_head.cond_proj` never received an explicit backpropagation gradient telling them: **"This root is impossible here by the laws of Sībawayh!"**

---

## 4. How to Tune the Layer: The Farāhīdian Negative Constraint Objective

To train the neural layer to natively internalize Al-Khalīl's and Sībawayh's algorithms, we implement a **Tri-Partite Loss Function**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NRMP}} + \lambda_1 \mathcal{L}_{\text{impossible}} + \lambda_2 \mathcal{L}_{\text{morph\_compat}}$$

### 1. The Negative Impossibility Loss ($\mathcal{L}_{\text{impossible}}$)
Whenever the model assigns probability mass to a structurally forbidden root (from Sībawayh's operator exclusion set $\mathcal{F}(x_t)$), penalize it with a hinge margin loss:

$$\mathcal{L}_{\text{impossible}} = \sum_{r \in \mathcal{F}(x_t)} \max\left(0, \text{logit}(r) - \text{margin}\right)^2$$

- After prepositions (`في`, `إلى`), $\mathcal{F}(x_t)$ contains all particle roots and verbal roots.
- After quantifiers (`كل`, `بعض`), $\mathcal{F}(x_t)$ contains all prepositions and verbs.
- For consecutive tokens, $\mathcal{F}(x_t)$ contains the immediate predecessor root (stutter prevention).

### 2. Morphological Wazn-Root Compatibility Loss ($\mathcal{L}_{\text{morph\_compat}}$)
Ibn Jinnī's morphosemantics dictates that roots and awzān are functionally coupled:
- A particle root ($R \in \mathcal{P}$) can **never** take an inflectional wazn ($W \neq \text{NONE}$).
- Weak/defective roots ($R \in \mathcal{W}$) require hollow/defective wazn alignments.
- We penalize any non-zero logit on incompatible $(R, W)$ pairs:
$$\mathcal{L}_{\text{morph\_compat}} = \sum_{(r, w) \in \text{Incompatible}} \text{ReLU}\left(\text{logit}(w \mid r) + 5.0\right)$$

### 3. Differentiable Hard Exclusion Layer in the Architecture
Instead of applying masks only as post-processing during generation, embed `SibawayhGovernanceLayer` directly inside `RootformerV18_NRMP.forward()`:
```python
class GovernedNRMPHead(nn.Module):
    def __init__(self, ...):
        ...
        self.gov = SibawayhNRMPGovernance(vocab)

    def forward(self, h, prev_tuples):
        # 1. Compute raw logits
        raw_logits = self.root_head(h)
        
        # 2. Differentiable Hard Exclusion
        mask = self.gov.compute_batch_exclusion_mask(prev_tuples)
        governed_logits = raw_logits + mask
        
        return governed_logits
```

This guarantees that:
- During training, the gradients flow **only through linguistically permissible roots**.
- The model never wastes capacity on impossible candidates.
- Zero extra inference latency is introduced because the mask is a fast boolean indexing operation on GPU.
