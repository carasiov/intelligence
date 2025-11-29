# NCA Intelligence Metrics: Experimental Specification

## Overview

This document specifies the first empirical instantiation of the Resource-Bounded Intelligence framework using Neural Cellular Automata (NCAs). The goal is to validate that the theoretical constructs ($K$, $I_{\text{local}}$, cognitive light cones, $H_{\text{eff}}$) are operationally meaningful and behave as predicted.

**Status:** Phase 1.5 — Core metric validation and cone pilot

---

## 1. System Under Test

### 1.1 Architecture

**Base NCA specification:**

| Parameter | Default | Ablation |
|-----------|---------|----------|
| Grid size | 64×64 | — |
| Channels (C) | 16 | 8 (half) |
| Hidden channels | 128 | 64 (half) |
| Kernel size | 3×3 | — |
| Update rule | Conv → ReLU → Conv | — |
| Stochastic mask | p=0.5 per cell per step | — |

The NCA state is $z_t \in \mathbb{R}^{H \times W \times C}$, with visible channels $s_t = z_t[:,:,:4]$ (RGBA) and hidden channels $h_t = z_t[:,:,4:]$.

### 1.2 Tasks

| Task ID | Description | Difficulty |
|---------|-------------|------------|
| `trivial` | Fill grid with constant color | Low |
| `simple` | Small centered circle or square | Medium |
| `complex` | Structured pattern with bilateral symmetry and fine detail (e.g., emoji, organism shape) | High |

**Target patterns** are fixed RGBA images. The goal functional is:

$$g(\text{Traj}) = -\text{MSE}(\text{final visible state}, \text{target})$$

### 1.3 Resource Vector $R$

For this experiment, resources include:

| Resource | Symbol | Values |
|----------|--------|--------|
| Time horizon (NCA steps) | $T$ | 64, 128, 256 |
| Channels | $C$ | 8, 16 |
| Training compute | $B$ | 10k, 50k, 100k, 200k steps |
| Damage size | $\delta$ | 0.05, 0.1, 0.2, 0.4 (fraction of grid) |

---

## 2. Baseline Policies

### 2.1 Random-Weight Baseline ($\pi_{\text{blind}}^{\text{rw}}$)

**Definition:**
- Same architecture as trained NCA (channels, kernel, update rule structure)
- Weights sampled i.i.d. from initialization distribution (e.g., Kaiming normal)
- Deterministic forward dynamics (same stochastic mask protocol)
- No training on any task

**Estimation protocol:**
1. Sample $N = 50$ independent weight initializations
2. For each, run forward dynamics for $T$ steps from the standard seed
3. Compute final MSE vs target
4. Report: $J(\pi_{\text{blind}}^{\text{rw}}) = \frac{1}{N} \sum_{i=1}^N \text{MSE}_i$ with standard error

### 2.2 Trivial Dynamics Baseline ($\pi_{\text{blind}}^{\text{triv}}$)

**Definition:**
- Output = initial seed state (no dynamics applied)
- Or: output = zero everywhere

**Purpose:** Tests whether *any* NCA dynamics help, vs. just outputting the seed.

**Estimation:** Single deterministic evaluation.

### 2.3 Approximate Optimal Policy ($\pi^*_B$)

**Definition:**
- Best model found under training budget $B$
- Budget includes: random hyperparameter samples × training steps

**Protocol:**
1. Fix a search budget schedule:
   - Level 1: 10 configs × 10k steps = 100k total
   - Level 2: 20 configs × 50k steps = 1M total
   - Level 3: 30 configs × 100k steps = 3M total
2. For each level, define $\pi^*_B = \arg\min_{\pi \in \text{found}} J(\pi)$
3. Track $K_{\text{opt}}(B) = \log_{10} \frac{J(\pi_{\text{blind}}^{\text{rw}})}{J(\pi^*_B)}$
4. Look for plateau in $K_{\text{opt}}(B)$ as $B$ increases

**Plateau criterion:** $|K_{\text{opt}}(B_{i+1}) - K_{\text{opt}}(B_i)| < 0.1$ for two consecutive budget levels.

---

## 3. Core Metrics

### 3.1 Search Compression ($K$)

**Definition:**

$$K(\pi) = \log_{10} \frac{J(\pi_{\text{blind}})}{J(\pi)}$$

**Variants:**
- $K_{\text{vs-rw}}$: relative to random-weight baseline
- $K_{\text{vs-triv}}$: relative to trivial baseline
- $K_{\text{opt}}(B)$: achieved by $\pi^*_B$ vs blind

**Interpretation:**
- $K = 1$ means policy is 10× better than blind
- $K = 2$ means 100× better
- Negative $K$ means worse than blind

### 3.2 Normalized Local Intelligence ($I_{\text{local}}$)

**Definition:**

$$I_{\text{local}}(\pi; B) = \frac{J(\pi_{\text{blind}}) - J(\pi)}{J(\pi_{\text{blind}}) - J(\pi^*_B)}$$

**Interpretation:**
- $I_{\text{local}} = 0$: policy equals blind baseline
- $I_{\text{local}} = 1$: policy equals best-known optimum
- $I_{\text{local}} > 1$: policy beats current $\pi^*_B$ (update $\pi^*_B$!)
- $I_{\text{local}} < 0$: policy is worse than blind

**Reporting:** Always specify the budget $B$ used to define $\pi^*_B$.

### 3.3 Metric Validation Checks

For the metrics to be considered "sane":

1. **Monotonicity with training:** $K$ and $I_{\text{local}}$ should increase (on average) as training progresses
2. **Correlation with loss:** Strong negative correlation between loss and $K$
3. **Sensitivity to resources:** 
   - Higher $C$ → higher achievable $K_{\text{opt}}$
   - Higher $T$ → higher achievable $K_{\text{opt}}$ (up to saturation)
4. **Task scaling:** Harder tasks have higher $K_{\text{opt}}$ (more "room for intelligence")

---

## 4. Cognitive Light Cones

### 4.1 Goal-Based Competence Function

**Definition:**

For a trained NCA, define the competence at location $x$, damage time $t_d$, and damage size $\delta$:

$$C(x, t_d, \delta) = P(\text{successful recovery} \mid \text{damage of size } \delta \text{ at } x \text{ applied at time } t_d)$$

**Success criterion:** Final MSE within $\epsilon$ of undamaged trajectory's final MSE.

**Damage model:**
- Shape: square of side $s = \sqrt{\delta \cdot H \cdot W}$ centered at $x$
- Effect: set all channels in damage region to zero
- Applied at step $t_d$

### 4.2 Estimation Protocol

1. **Grid of probe points:** Coarse grid over $(x, t_d)$ space
   - $x$: 5×5 spatial grid (25 locations)
   - $t_d$: 5 values in $[0, T]$
   - $\delta$: 4 values (0.05, 0.1, 0.2, 0.4)
   
2. **Trials per condition:** $N = 20$ (different stochastic mask seeds)

3. **Output:** 4D array $C[x, t_d, \delta, \text{trial}]$

### 4.3 Cone Visualization

For fixed $\delta$:
- Plot $C(x, t_d)$ as a heatmap over space × time
- Threshold at $\theta = 0.5$ (or 0.8) to define the "cone boundary"

### 4.4 Predicted Behaviors

| Manipulation | Predicted Effect on Cone |
|--------------|-------------------------|
| Increase $\delta$ | Cone shrinks |
| Increase $T$ | Cone expands (more time to recover) |
| Increase $C$ (channels) | Cone expands (more capacity) |
| Late $t_d$ (damage near end) | Cone shrinks (less time to recover) |
| Damage at center vs edge | Center may have larger cone (more neighbors to draw from) |

---

## 5. Effective Prediction Horizon ($H_{\text{eff}}$)

### 5.1 Definition

$$H_{\text{eff}} = \max\{k : E_k < \theta \cdot \text{Var}(s_{t+k})\}$$

where $E_k$ is the prediction error of a probe trained to predict $s_{t+k}$ from $z_t$.

### 5.2 Estimation Protocol

1. **Collect trajectories:** Run NCA for $T$ steps, record $(z_t, s_t)$ for all $t$
2. **Train probes:** For each $k \in \{1, 2, 4, 8, 16, 32, 64\}$:
   - Train a small MLP: $\hat{s}_{t+k} = f_k(z_t)$
   - Use held-out trajectories for evaluation
   - Record $E_k = \mathbb{E}[\|s_{t+k} - \hat{s}_{t+k}\|^2]$
3. **Baseline variance:** $\text{Var}(s_{t+k})$ = variance of future states
4. **Threshold:** $\theta = 0.5$ (prediction error < 50% of variance)

### 5.3 Required Comparisons

| Condition | Expected $H_{\text{eff}}$ |
|-----------|--------------------------|
| Untrained (random weights) | Low (baseline) |
| Trained on trivial task | ≈ Untrained |
| Trained on complex task | > Untrained |

**Hypothesis:** $H_{\text{eff}}(\text{trained, complex}) > H_{\text{eff}}(\text{untrained})$

**Falsification:** If all three are approximately equal, $H_{\text{eff}}$ is an architectural constant, not a learned property.

---

## 6. Phase 1.5 Deliverables

### 6.1 Minimum Viable Experiment

For **one grid size (64×64), one base architecture (16 channels), two tasks (simple, complex)**:

| Deliverable | Success Criterion |
|-------------|-------------------|
| Baseline estimates | $J(\pi_{\text{blind}}^{\text{rw}})$ with SE < 10% of mean |
| $K_{\text{opt}}(B)$ curve | Shows plateau or diminishing returns |
| Training curves | $K$, $I_{\text{local}}$, loss all tracked |
| Resource ablation | $K_{\text{opt}}$ responds to channel count |
| Task ablation | $K_{\text{opt}}(\text{complex}) > K_{\text{opt}}(\text{simple})$ |
| Cone pilot | Heatmaps for 2 values of $\delta$ showing shrinkage |
| $H_{\text{eff}}$ pilot | Comparison across 3 conditions |

### 6.2 Expected Figures

1. **Fig 1:** Loss, $K$, $I_{\text{local}}$ vs training steps (3 panels)
2. **Fig 2:** $K_{\text{opt}}(B)$ vs search budget, showing plateau
3. **Fig 3:** $K_{\text{opt}}$ vs channels (resource sensitivity)
4. **Fig 4:** $K_{\text{opt}}$ by task difficulty (simple vs complex)
5. **Fig 5:** Competence heatmaps $C(x, t_d)$ for $\delta \in \{0.1, 0.3\}$
6. **Fig 6:** Cone size $|L^{\text{goal}}_T(\delta)|$ vs $\delta$
7. **Fig 7:** Prediction error $E_k$ vs horizon $k$ for 3 conditions
8. **Fig 8:** $H_{\text{eff}}$ comparison bar chart

### 6.3 Go/No-Go Criteria

**Proceed to Phase 2 if:**
- [ ] $K$ and $I_{\text{local}}$ increase with training (monotonicity)
- [ ] $K_{\text{opt}}$ responds to resources as predicted
- [ ] Cones shrink with $\delta$ and expand with $T$
- [ ] At least one of the above effects is statistically significant ($p < 0.05$)

**Revise framework if:**
- [ ] $K$ and loss are uncorrelated or inversely correlated
- [ ] Cones show no spatial or temporal structure
- [ ] $H_{\text{eff}}$ is identical across all conditions

---

## 7. Future Phases (Preview)

### Phase 2: Composition and $\mathsf{IsAgent}$

- Implement coupling parameter $\lambda$ between grid halves
- Measure $K_{\text{macro}}(\lambda)$, MI($\lambda$), cone connectivity
- Test external indistinguishability with classifier
- Look for composition transition

### Phase 3: Self-Model ($M$) and Coherence

- Train probes for boundary, damage, error representations
- Causal intervention experiments along "self" latent directions
- Coherence maintenance tests (error latent → corrective action → updated error)

### Phase 4: Cross-System Comparison

- Apply same metrics to:
  - RL agents in spatial tasks
  - Biological data (if available from Levin lab)
- Test whether $K$, cones, $H_{\text{eff}}$ generalize across substrates

---

## 8. Code Organization

```
experiments/nca_intelligence_metrics/
├── EXPERIMENT_SPEC.md          # This document
├── src/
│   ├── nca.py                  # NCA architecture
│   ├── baselines.py            # π_blind implementations
│   ├── metrics.py              # K, I_local, cone estimation
│   ├── probes.py               # H_eff predictors
│   └── training.py             # Training loop with metric logging
├── configs/
│   ├── base.yaml               # Default hyperparameters
│   ├── ablations/              # Resource and task ablations
│   └── search_budget.yaml      # Budget schedule for π*_B
├── scripts/
│   ├── run_phase1.py           # Main experiment runner
│   ├── estimate_baselines.py   # Baseline estimation
│   └── analyze_results.py      # Figure generation
├── results/
│   ├── baselines/              # Cached baseline estimates
│   ├── training_runs/          # Training logs and checkpoints
│   ├── cones/                  # Cone estimation results
│   └── figures/                # Generated figures
└── tests/
    ├── test_metrics.py         # Unit tests for metric computation
    └── test_nca.py             # NCA forward pass tests
```

---

## 9. Risk Register

| Risk | Mitigation |
|------|------------|
| Baseline variance too high | Increase N, use variance reduction |
| $K_{\text{opt}}$ doesn't plateau | Report as open question; use best-at-budget |
| Cones show no structure | Check damage model; try different thresholds |
| $H_{\text{eff}}$ probe too weak | Use deeper probes; report probe architecture |
| Compute budget exceeded | Prioritize: baselines → K curves → cones → H |

---

## 10. Timeline (Suggested)

| Week | Milestone |
|------|-----------|
| 1 | NCA implementation + training loop working |
| 2 | Baseline estimation complete |
| 3 | Training curves with K, I_local logged |
| 4 | Resource and task ablations |
| 5 | Cone estimation pilot |
| 6 | H_eff estimation pilot |
| 7 | Analysis and figure generation |
| 8 | Write-up of Phase 1.5 results |

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\pi$ | Policy (trained NCA) |
| $\pi_{\text{blind}}$ | Baseline policy (random weights or trivial) |
| $\pi^*_B$ | Best policy found under budget $B$ |
| $J(\pi)$ | Cost (MSE) of policy |
| $K$ | Search compression: $\log_{10} \frac{J(\pi_{\text{blind}})}{J(\pi)}$ |
| $I_{\text{local}}$ | Normalized performance: $\frac{J(\pi_{\text{blind}}) - J(\pi)}{J(\pi_{\text{blind}}) - J(\pi^*_B)}$ |
| $C(x, t_d, \delta)$ | Competence at location $x$, damage time $t_d$, damage size $\delta$ |
| $L^{\text{goal}}_T$ | Goal-based cognitive light cone (region where $C > \theta$) |
| $H_{\text{eff}}$ | Effective prediction horizon |
| $R$ | Resource vector $(T, C, B, ...)$ |
| $T$ | Time horizon (NCA steps) |
| $C$ | Number of channels |
| $B$ | Training/search budget |
| $\delta$ | Damage size (fraction of grid) |

---

## Appendix B: Theoretical Grounding

This experiment instantiates the following theoretical constructs from the framework:

| Framework Concept | NCA Instantiation |
|-------------------|-------------------|
| Environment class $\mathcal{E}$ | Grid + initial seed + damage distribution |
| Goal $g$ | Negative MSE to target pattern |
| Resource vector $R$ | $(T, C, B)$ |
| Policy $\pi$ | Trained NCA weights |
| $\mathrm{Perf}(\pi; e, g, R)$ | $-J(\pi) = -\text{MSE}$ |
| $\mathrm{Perf}^*(e, g, R)$ | $-J(\pi^*_B)$ |
| Cognitive light cone | Region where $C(x, t_d, \delta) > \theta$ |
| Prediction horizon $H$ | $H_{\text{eff}}$ from probe decay |

The experiment tests whether these instantiations behave consistently with the theoretical predictions (monotonicity, resource sensitivity, cone geometry).
