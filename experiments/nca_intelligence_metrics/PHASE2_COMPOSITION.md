# Phase 2: Composition and Macro-Agency

## Overview

This phase tests the $\mathsf{IsAgent}$ predicate and composition laws. The core question: **when does a collection of sub-agents behave as a single macro-agent, and does composition improve intelligence?**

**Prerequisites:** Phase 1.5 complete with validated $K$, $I_{\text{local}}$, and cone metrics.

---

## 1. Experimental Setup

### 1.1 The Composition Testbed

**Architecture:** A 64×64 NCA grid conceptually divided into left half (A) and right half (B).

**Coupling parameter $\lambda \in [0, 1]$:**
- $\lambda = 1$: Full coupling (standard NCA, all neighbors communicate)
- $\lambda = 0$: No coupling (left and right halves are independent NCAs)
- Intermediate $\lambda$: Cross-midline kernel weights multiplied by $\lambda$

**Implementation:**
```python
def apply_coupling(kernel_weights, lambda_val):
    """Mask cross-midline connections by lambda."""
    # For a 3x3 kernel at the midline:
    # - Same-side connections: weight unchanged
    # - Cross-midline connections: weight *= lambda
    ...
```

### 1.2 Tasks

| Task | Description | Why |
|------|-------------|-----|
| `independent` | Left half → pattern A, Right half → pattern B (no boundary constraint) | Tests baseline separation |
| `joint` | Single pattern spanning both halves with boundary alignment required | Tests whether coordination helps |
| `boundary-critical` | Pattern with fine structure exactly at the midline | Maximizes need for cross-boundary information |

### 1.3 Models to Compare

| Model | Description |
|-------|-------------|
| $\pi_A$ | NCA trained only on left half, pattern A |
| $\pi_B$ | NCA trained only on right half, pattern B |
| $\pi_{\text{sep}}$ | $\pi_A$ and $\pi_B$ run independently (no communication) |
| $\pi_{\text{fused}}(\lambda)$ | Single NCA on full grid with coupling $\lambda$ |

---

## 2. The Three Conditions for $\mathsf{IsAgent}$

### 2.1 Condition 1: Structural Connectivity

**For NCAs:** Always satisfied when $\lambda > 0$ (there exist paths across the midline).

**Measurement:** Binary check on the communication graph.

### 2.2 Condition 2: Sufficient Coupling (Information Flow)

**Operational measure:** Mutual information across the midline.

$$\text{MI}(\lambda) = I(L_t; R_{t+1})$$

where:
- $L_t$ = internal states of cells in a band near the left side of midline at time $t$
- $R_{t+1}$ = internal states of cells in a band near the right side at time $t+1$

**Estimation protocol:**
1. Run trained NCA for many trajectories (with and without damage)
2. Collect $(L_t, R_{t+1})$ pairs
3. Estimate MI using:
   - Binning estimator (coarse but robust)
   - Or: MINE / neural MI estimator (more precise but noisier)
4. Report MI as function of $\lambda$

**Expected behavior:**
- MI$(\lambda = 0) \approx 0$
- MI$(\lambda)$ increases with $\lambda$
- MI$(\lambda = 1)$ reflects natural information flow in fused NCA

### 2.3 Condition 3: External Indistinguishability

**Question:** Can an external observer distinguish the fused NCA from a "macro-agent" formed by coupling two separate NCAs?

**Classifier-based test:**

1. **Generate outputs:**
   - Run $\pi_{\text{fused}}(\lambda_{\text{high}})$ on joint task, collect final patterns
   - Run $\pi_{\text{sep}}$ (coupled via explicit message passing with same $\lambda$) on joint task
   - Run $\pi_{\text{sep}}$ with $\lambda = 0$ (broken coupling)

2. **Train discriminator:**
   - Binary classifier $D$: predict which model produced the pattern
   - Same architecture and training budget for all comparisons
   - Report accuracy on held-out test set

3. **Power calibration (CRITICAL):**
   - Verify $D$ can distinguish fused vs broken ($\lambda = 0$) with accuracy > 0.7
   - If $D$ fails this, the test has no power → increase $D$ capacity or sample size

4. **Decision rule:**
   
   | $D$ accuracy on fused vs broken | $D$ accuracy on fused vs coupled | Interpretation |
   |--------------------------------|----------------------------------|----------------|
   | > 0.7 | < 0.55 | Indistinguishable → $\mathsf{IsAgent}$ = true |
   | > 0.7 | > 0.65 | Distinguishable → $\mathsf{IsAgent}$ = false |
   | < 0.6 | any | Test underpowered → increase samples/capacity |

---

## 3. Composition Laws

### 3.1 Quantities to Measure

For the joint task:

| Quantity | Definition |
|----------|------------|
| $K_A$ | Search compression of $\pi_A$ on pattern A alone |
| $K_B$ | Search compression of $\pi_B$ on pattern B alone |
| $K_{\text{sep}}$ | Search compression of $\pi_A + \pi_B$ on joint task (no coupling) |
| $K_{\text{fused}}(\lambda)$ | Search compression of fused NCA on joint task |
| $|L^{\text{goal}}_T(\lambda)|$ | Size of competence cone for joint task |

### 3.2 Theoretical Predictions

From the corridor theorem and general framework:

| Prediction | Expected Behavior |
|------------|-------------------|
| Low coupling limit | $K_{\text{fused}}(\lambda \to 0) \to K_{\text{sep}} \approx \max(K_A, K_B)$ |
| High coupling | $K_{\text{fused}}(\lambda_{\text{high}}) \geq \max(K_A, K_B)$ |
| Synergy (hoped) | For boundary-critical tasks: $K_{\text{fused}} > \max(K_A, K_B)$ |
| Cone correlation | $|L^{\text{goal}}_T(\lambda)|$ increases with $\lambda$ |
| MI correlation | Higher MI$(\lambda)$ correlates with higher $K_{\text{fused}}(\lambda)$ |

### 3.3 Composition Transition

Plot $K_{\text{fused}}(\lambda)$, MI$(\lambda)$, and $|L^{\text{goal}}_T(\lambda)|$ vs $\lambda$.

**Look for:**
- A transition point $\lambda^*$ where macro-agency "kicks in"
- Correspondence between MI threshold and K improvement
- Cone connectivity changes at similar $\lambda^*$

### 3.4 Falsification Scenarios

**Scenario 1: No synergy anywhere**
- For all $\lambda$: $K_{\text{fused}}(\lambda) \approx \max(K_A, K_B)$
- Interpretation: Composition provides parallel coverage but no emergent capability
- Action: Report honestly; note that corridor theorem is a special case

**Scenario 2: Interference**
- For high $\lambda$: $K_{\text{fused}}(\lambda) < \max(K_A, K_B)$
- Interpretation: Coupling harms performance; macro-agent is worse than best sub-agent
- Action: Investigate why (conflicting gradients? interference in latent space?)

**Scenario 3: Threshold behavior**
- Sharp transition at some $\lambda^*$: below it, $K \approx K_{\text{sep}}$; above it, $K$ jumps
- Interpretation: $\mathsf{IsAgent}$ is a genuine phase transition, not a gradient
- Action: Characterize the transition; relate to MI threshold

---

## 4. Cone Composition

### 4.1 Definition

For the fused NCA on the joint task:

$$L^{\text{goal}}_T(\lambda) = \{(x, t_d) : C(x, t_d, \delta; \lambda) > \theta\}$$

### 4.2 Expected Behaviors

| Regime | Cone Structure |
|--------|---------------|
| $\lambda = 0$ | Two separate cones, one per half |
| $\lambda$ intermediate | Cones begin to merge at boundary |
| $\lambda = 1$ | Single unified cone |

### 4.3 Measurement

1. Estimate $C(x, t_d, \delta)$ as in Phase 1.5
2. Visualize cones for $\lambda \in \{0, 0.25, 0.5, 0.75, 1.0\}$
3. Track cone "connectivity" (is the cone a single connected region or two separate ones?)

---

## 5. Deliverables

### 5.1 Figures

1. **Fig 1:** MI($\lambda$) vs $\lambda$
2. **Fig 2:** $K_{\text{fused}}(\lambda)$ vs $\lambda$, with $K_{\text{sep}}$ and $\max(K_A, K_B)$ as reference lines
3. **Fig 3:** Classifier accuracy vs $\lambda$ (fused vs coupled comparison)
4. **Fig 4:** Cone visualizations at 5 values of $\lambda$
5. **Fig 5:** Cone size $|L^{\text{goal}}_T(\lambda)|$ vs $\lambda$
6. **Fig 6:** Scatter plot of MI vs $K_{\text{fused}}$ across $\lambda$ values

### 5.2 Claims to Test

| Claim | Test | Pass Criterion |
|-------|------|----------------|
| $\mathsf{IsAgent}$ is operational | Classifier + MI + K all show consistent transition | All three change together |
| Composition can improve K | $K_{\text{fused}}(\lambda_{\text{high}}) > K_{\text{sep}}$ | Statistically significant ($p < 0.05$) |
| Cones expand under composition | $|L^{\text{goal}}_T(\lambda_{\text{high}})| > |L^{\text{goal}}_T(0)|$ | Visible in visualization + quantified |
| Coupling failure → cone collapse | At $\lambda = 0$, cone splits into two | Clear separation in visualization |

---

## 6. Go/No-Go for Phase 3

**Proceed if:**
- [ ] $\mathsf{IsAgent}$ conditions are consistently measurable
- [ ] At least one of: synergy in $K$, or clear composition transition, is observed
- [ ] Cone behavior matches predictions (shrink/expand/split)

**Pause and reconsider if:**
- [ ] MI and $K$ are uncorrelated
- [ ] Classifier always distinguishes fused from coupled (no "macro-agent" regime)
- [ ] Cones show no dependence on $\lambda$
