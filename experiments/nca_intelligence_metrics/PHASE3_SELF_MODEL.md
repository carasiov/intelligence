# Phase 3: Self-Model and Coherence Maintenance

## Overview

This phase tests the **consciousness-related** hypotheses: whether trained NCAs develop internal representations that function as self-models, and whether these representations participate causally in coherence maintenance.

**Prerequisites:** Phase 1.5 and Phase 2 complete.

**Epistemic status:** This phase tests a *conjectural* extension of the framework. Negative results here do not invalidate the core intelligence metrics—they constrain the consciousness story.

---

## 1. The Three Conditions for Self-Model ($M$)

From the theoretical framework, a macro-agent has a functional self-model $M$ when:

1. **Representation:** $M$ encodes the agent's boundary, goals, and approximate light cone
2. **Causal involvement:** The policy factors through $M$; interventions on $M$ produce structured behavior changes
3. **Coherence maintenance:** There is an ongoing process that detects prediction-reality mismatches, updates $M$, and uses the updated $M$ to steer behavior

This phase operationalizes and tests each condition.

---

## 2. Condition 1: Representation Tests

### 2.1 What Should Be Represented?

For an NCA to have a self-model, its hidden state should encode:

| Property | Description | Probe Target |
|----------|-------------|--------------|
| Boundary | Where is the pattern's edge? | Binary mask of "body" cells |
| Damage state | Has damage occurred? Where? How severe? | Damage location and size |
| Global error | How far is current state from target? | MSE or similar |
| Completeness | What fraction of the target is achieved? | Fraction of cells at target |

### 2.2 Probe Architecture

**Linear probes (primary):**
- Input: flattened hidden state $h_t \in \mathbb{R}^{H \times W \times C_{\text{hidden}}}$
- Output: target property (binary mask, scalar, etc.)
- Why linear: tests whether information is *explicitly* represented, not just computable

**Nonlinear probes (secondary):**
- Small MLP (2-3 layers)
- Tests whether information is present but in a more complex format

### 2.3 Protocol

1. **Collect data:**
   - Run trained NCA with and without damage
   - Record $(h_t, \text{ground truth property})$ pairs at multiple timesteps

2. **Train probes:**
   - Separate train/test split
   - Report: accuracy (for classification), $R^2$ (for regression)

3. **Baselines:**
   - Random-weight NCA (untrained): does representation emerge from training?
   - Probe on visible state only: is hidden state necessary?

### 2.4 Expected Results

| Comparison | Prediction |
|------------|------------|
| Trained vs untrained NCA | Trained has higher probe accuracy |
| Linear vs nonlinear probe | If linear works well, representation is explicit |
| Hidden vs visible state | Hidden state has *more* information about damage/error |
| Early vs late training | Probe accuracy increases with training |

**Positive evidence for representation:** Trained NCA has significantly higher probe accuracy than untrained, especially for high-level properties (global error, boundary).

---

## 3. Condition 2: Causal Involvement

### 3.1 The Core Question

Representation is necessary but not sufficient. The self-model must **participate in control**.

**Test:** Interventions along "self-model" directions produce *structured, goal-relevant* behavior changes, while random interventions just add noise.

### 3.2 Identifying Self-Model Directions

From the probes in Section 2, identify latent directions:

| Direction | Meaning |
|-----------|---------|
| $\Delta z_{\text{damage}}$ | Direction that increases "damage detected" |
| $\Delta z_{\text{error}}$ | Direction that increases "global error" |
| $\Delta z_{\text{boundary}}$ | Direction that shifts perceived boundary |

**Method:** 
- Use the weights of the linear probe (if linear probe works)
- Or: use gradient of nonlinear probe output w.r.t. hidden state

### 3.3 Intervention Protocol

1. **Snapshot:** Run NCA to time $t$; record hidden state $h_t$
2. **Perturb:** Apply perturbation $\Delta h$ to all cells (or a subset)
3. **Continue:** Run NCA from perturbed state; do not touch visible state at $t$
4. **Measure:** Compare behavior to unperturbed trajectory

**Perturbation types:**

| Type | Definition |
|------|------------|
| Self-model direction ($\Delta z_{\text{damage}}$, etc.) | Perturbation along identified latent direction |
| Random direction | Random vector with same L2 norm as self-model perturbation |
| Zero perturbation | Control (no change) |

### 3.4 Metrics

| Metric | Definition |
|--------|------------|
| $\Delta E_{\text{target}}$ | Change in final MSE vs target |
| $\Delta B$ | Change in behavioral metric (e.g., amount of regrowth, boundary shift) |
| Structured change score | Are behavior changes interpretable? (qualitative + quantitative) |

### 3.5 Predictions

**Positive evidence for causal involvement:**

| Perturbation | Predicted Effect |
|--------------|-----------------|
| +$\Delta z_{\text{damage}}$ (inject "you are damaged") | Triggers repair behavior; extra growth/regrowth |
| -$\Delta z_{\text{damage}}$ (inject "you are fine") | Suppresses repair; less activity |
| +$\Delta z_{\text{error}}$ (inject "high error") | Triggers corrective dynamics |
| Random direction | Mostly noise; no structured effect; may degrade performance |

**Quantitative test:**
- Variance of $\Delta B$ for self-model perturbations > variance for random perturbations
- Mean effect of self-model perturbations is *directional* (positive for +, negative for -)

### 3.6 Control Experiments

| Control | Purpose |
|---------|---------|
| Perturb visible state instead of hidden | Is hidden state special? |
| Perturb with very small $\|\Delta h\|$ | Is effect continuous in perturbation size? |
| Perturb untrained NCA | Does causal structure require training? |

---

## 4. Condition 3: Coherence Maintenance

### 4.1 The Core Question

Does the NCA maintain a self-model over time, updating it in response to reality?

This is the hardest condition to test, but we can look for:

1. **Error tracking:** Internal "error latent" correlates with actual error over time
2. **Temporal ordering:** Error signal precedes corrective action
3. **Error-driven updates:** Perturbing the error latent triggers self-model updates

### 4.2 Error Latent Dynamics

**Protocol:**
1. Identify error latent direction $e$ from probes (Section 3.2)
2. Define error signal: $E_t = \langle h_t, e \rangle$ (projection onto error direction)
3. Track $E_t$ over time during:
   - Normal growth (seed → target)
   - Damage and recovery

**Expected dynamics:**
- $E_t$ spikes after damage
- $E_t$ decreases as recovery proceeds
- Corrections in visible pattern follow *after* increases in $E_t$

**Analysis:**
- Cross-correlation between $E_t$ and $\frac{d}{dt}\text{MSE}(s_t, \text{target})$
- Granger causality: does $E_t$ predict future corrections?

### 4.3 Coherence-Violation Interventions

**The critical test:** Inject inconsistent error signals and see if the system corrects them.

**Protocol:**

| Intervention | Setup | Predicted Response |
|--------------|-------|-------------------|
| False alarm | Pattern is complete; inject high $E_t$ | Unnecessary repair starts, then subsides as reality signal overwrites |
| Missed damage | Damage just occurred; inject low $E_t$ | Repair is sluggish initially; system updates $E_t$ as mismatch accumulates |

**What we're looking for:**
- After injection, $E_t$ drifts back toward the "correct" value
- This drift is *faster* than passive decay (there's active correction)
- Behavior initially follows the injected $E_t$, then adjusts

**Positive evidence for coherence maintenance:**
- Error latent is not just a passive encoding; it participates in a feedback loop
- Injecting wrong values causes temporary misbehavior, followed by correction

**Negative evidence (still valuable):**
- Error latent is passively updated from visible state with no autonomous dynamics
- Injecting wrong values simply wrecks behavior with no correction

---

## 5. Deliverables

### 5.1 Figures

1. **Fig 1:** Probe accuracy (trained vs untrained) for each property
2. **Fig 2:** Probe accuracy vs training time
3. **Fig 3:** Perturbation effect sizes: self-model directions vs random
4. **Fig 4:** Example trajectories: +damage perturbation triggers repair
5. **Fig 5:** Error latent $E_t$ over time during damage/recovery
6. **Fig 6:** Cross-correlation of $E_t$ with MSE dynamics
7. **Fig 7:** Coherence-violation intervention: false alarm scenario
8. **Fig 8:** Coherence-violation intervention: missed damage scenario

### 5.2 Claims to Test

| Claim | Test | Pass Criterion |
|-------|------|----------------|
| Trained NCAs represent self-properties | Probe accuracy > untrained baseline | Significant difference ($p < 0.01$) |
| Hidden state contains more than visible | Probe on hidden > probe on visible | For at least damage/error properties |
| Self-model is causally involved | Self-model perturbations > random perturbations | Effect size difference significant |
| Coherence is actively maintained | Error latent corrects after violation | Drift-back faster than passive decay |

---

## 6. Interpretation Matrix

| Representation | Causal | Coherence | Interpretation |
|----------------|--------|-----------|----------------|
| ✓ | ✓ | ✓ | Full self-model; supports consciousness hypothesis |
| ✓ | ✓ | ✗ | Representation + control, but no active maintenance; "reactive" self-model |
| ✓ | ✗ | ✗ | Representation only; self-information present but not used |
| ✗ | ✗ | ✗ | No self-model; NCA works without explicit self-representation |

Each outcome is informative. The consciousness hypothesis requires all three; partial results constrain the theory.

---

## 7. Relation to Bach's Framework

This phase connects to Joscha Bach's "consciousness as coherence maximizer" view:

| Bach Concept | Our Operationalization |
|--------------|------------------------|
| Consciousness increases coherence | Coherence maintenance loop (Condition 3) |
| Self as a model in the mind | Self-model $M$ (probed representation) |
| Attention to incoherence | Error latent $E_t$ tracking mismatches |
| Cortical conductor resolving conflict | Causal involvement of error signal in control |

If Conditions 1-3 hold, we have evidence that the NCA implements something like Bach's "coherence-maximizing operator."

---

## 8. Risk Register

| Risk | Mitigation |
|------|------------|
| Probes overfit | Use separate train/test; report generalization gap |
| Self-model directions are ill-defined | Use multiple methods (linear, gradient); check consistency |
| Perturbation effects are too noisy | Increase sample size; use matched controls |
| Coherence maintenance is too subtle | Use longer interventions; track over more timesteps |
| Negative results for all conditions | Report honestly; constrains consciousness hypothesis |

---

## 9. Go/No-Go Summary

**Strong positive result:** All three conditions show positive evidence → self-model hypothesis supported for NCAs

**Mixed result:** Some conditions pass, others fail → refine the hypothesis; identify what NCAs lack

**Strong negative result:** No conditions pass → NCAs achieve morphogenesis without self-models; the consciousness story doesn't apply here (but intelligence metrics from Phase 1-2 may still hold)

Any of these outcomes is scientifically valuable.
