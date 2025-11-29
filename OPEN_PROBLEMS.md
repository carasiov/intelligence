# Open Problems in the Resource-Bounded Theory of Intelligence

This document collects conceptual and technical questions that are not
yet resolved in the core framework.

**Status key:**
- 🔴 **Open:** No concrete approach yet
- 🟡 **In progress:** Experimental approach designed, not yet executed
- 🟢 **Addressed:** Experimental results available (see linked documents)

---

## 1. Null models for meta-level problem-space search

🔴 **Open**

We can define a meta-space \(\mathcal{P}^{(2)}\) whose states are problem
spaces \(P = \langle S,O,C,E,H\rangle\) and whose operators are edits to
\((S,O,C,E,H)\). Searching well in \(\mathcal{P}^{(2)}\) corresponds to
"representational creativity" or meta-intelligence.

**Open questions:**

- What should count as a "blind baseline" \(\pi^{(2)}_{\mathrm{blind}}\)
  in \(\mathcal{P}^{(2)}\)?
- How should we define \(\tau^{(2)}_{\mathrm{blind}}\) and
  \(K^{(2)}\) in a way that is not arbitrary?
- Are there natural priors over edits to \((S,O,C,E,H)\) that make sense
  biologically or computationally?

---

## 2. Operationalizing self-models \(M\)

🟡 **In progress** — See \`experiments/nca_intelligence_metrics/PHASE3_SELF_MODEL.md\`

The consciousness story relies on a self-model \(M\) that encodes a
macro-agent's boundary, goals, and light cone, and that participates in
control.

**Original questions:**

- How can we empirically identify \(M\) in biological systems or
  artificial agents?
- What minimal dimensionality / expressiveness should \(M\) have to
  count as a genuine self-model, rather than a fixed parameter (like a
  thermostat setpoint)?
- How do we distinguish "mere internal state" from a structure that
  supports flexible, counterfactual control over the agent's own
  boundary and goals?

**Proposed experimental approach:**

1. **Representation tests:** Train probes to decode boundary, damage,
   and global error from hidden states. Compare trained vs untrained
   systems.

2. **Causal involvement tests:** Identify latent directions
   corresponding to self-properties (e.g., "damage detected"). Perturb
   along these directions and compare behavioral effects to random
   perturbations of equal magnitude. Self-model perturbations should
   produce structured, goal-relevant changes.

3. **Coherence maintenance tests:** Track "error latent" over time;
   verify it spikes after damage and precedes corrective action. Inject
   inconsistent error signals (false alarms, missed damage) and observe
   whether the system corrects the self-model back toward reality.

**Success criteria:** All three conditions (representation, causal
involvement, coherence maintenance) show positive evidence in at least
one trained system.

---

## 3. Empirical discriminability vs IIT, GWT, etc.

🔴 **Open**

Our framework predicts patterns in \(K\), \(I_{\mathrm{local}}\),
\(\mathsf{IsAgent}\), and light-cone structure; other theories (e.g.
Integrated Information Theory, Global Workspace Theory) make predictions
about integrated information, broadcast patterns, and accessibility.

**Open questions:**

- Can we construct systems with high \(\Phi\) but low \(K\), and vice
  versa, and test which measures better track conscious reports or
  flexible control?
- In anesthetic or fragmentation paradigms, do changes in
  \(\mathsf{IsAgent}\) and self-model participation predict loss of
  consciousness better than changes in \(\Phi\) or global broadcast?
- What experimental designs are feasible given current measurement
  tools?

---

## 4. Measuring prediction horizon \(H\)

🟡 **In progress** — See \`experiments/nca_intelligence_metrics/EXPERIMENT_SPEC.md\` §5

The prediction horizon \(H\) is a distinct resource from time horizon
\(T\), but it is not straightforward to measure.

**Original questions:**

- How can we empirically estimate \(H\) for cells, tissues, organisms,
  or collectives?
- How does \(H\) scale across levels?
- Are there generic scaling laws relating \(H\) to other resources?

**Proposed operational definition:**

\[
H_{\mathrm{eff}} = \max\{k : E_k < \theta \cdot \mathrm{Var}(s_{t+k})\}
\]

where \(E_k\) is the prediction error of a probe trained to predict
future states \(s_{t+k}\) from current internal state \(z_t\).

**Required comparisons:**
- Untrained (random-weight) system: baseline \(H_{\mathrm{eff}}\)
- Trained on trivial task: ≈ baseline
- Trained on complex task: > baseline (hypothesis)

**Falsification:** If \(H_{\mathrm{eff}}\) is identical across all
conditions, it's an architectural constant rather than a learned
property.

---

## 5. Measuring \(K\) and \(K_{\mathrm{opt}}\) in AI systems

🟡 **In progress** — See \`experiments/nca_intelligence_metrics/EXPERIMENT_SPEC.md\` §2-3

The search-efficiency metric \(K\) is attractive conceptually but can be
difficult to estimate for modern AI systems.

**Original questions:**

- What should count as a "blind" baseline for large models and RL agents?
- How can we approximate costs without running prohibitive baselines?
- How can capacity bounds be used to estimate \(K_{\mathrm{opt}}\)?

**Proposed solutions:**

1. **Baseline specification:** For a given architecture, \(\pi_{\mathrm{blind}}\)
   is defined as same architecture with random (untrained) weights,
   averaged over \(N \geq 30\) weight samples. This makes \(K\) relative
   to that declared baseline.

2. **Approximate optimal:** Define \(\pi^*_B\) as the best policy found
   under a fixed search budget \(B\). Track \(K_{\mathrm{opt}}(B)\) vs
   budget and look for a plateau; freeze \(\pi^* = \pi^*_B\) once
   diminishing returns are observed.

3. **Sensitivity reporting:** Always report the budget \(B\) and show
   the \(K_{\mathrm{opt}}(B)\) curve so readers can assess stability.

**Remaining question:** How to extend this to large foundation models
where even random-weight runs are expensive?

---

## 6. Composition laws beyond simple domains

🟡 **In progress** — See \`experiments/nca_intelligence_metrics/PHASE2_COMPOSITION.md\`

The 1D corridor theorem shows linear capacity scaling until saturation.
Does this extend to more complex domains?

**Open questions:**

- Under what conditions does macro-\(K\) exceed max sub-\(K\) (synergy)?
- When does composition cause interference (macro-\(K\) < max sub-\(K\))?
- How do coupling strength and information flow relate to composition gains?

**Proposed experimental approach:**

1. **Coupling parameter \(\lambda\):** Vary cross-boundary communication
   in a divided NCA grid from 0 (independent) to 1 (fully coupled).

2. **Track three quantities:** Mutual information across boundary,
   \(K_{\mathrm{fused}}(\lambda)\), and classifier distinguishability.

3. **Look for transition:** Does macro-agency "kick in" at some
   threshold \(\lambda^*\)?

**Predicted outcomes:**
- At \(\lambda \to 0\): \(K_{\mathrm{fused}} \to K_{\mathrm{sep}}\)
- At high \(\lambda\): \(K_{\mathrm{fused}} \geq \max(K_A, K_B)\)
- For boundary-critical tasks: possibly \(K_{\mathrm{fused}} > \max\)

---

## 7. Learnability and developmental trajectories

🔴 **Open**

The framework treats \(\pi\) as given. Real intelligence involves
*becoming* intelligent over time.

**Open questions:**

- How should we model \(I_t = I(\pi_t; \cdot)\) as a function of
  training/development time?
- Can we define "learning efficiency" as a resource-bounded quantity?
- How does consciousness (or coherence maintenance) relate to learning
  speed?

**Possible directions:**

- Define developmental resources (lifetime steps, plasticity budget) as
  part of \(R\).
- Compare agents not just by final \(I\) but by learning curves
  \(I(t)\) under matched resource budgets.
- Test whether coherence-maintenance (Phase 3) correlates with faster
  learning.

---

## 8. Cross-substrate validation

🔴 **Open**

The framework claims substrate-neutrality. This must be tested by
applying the same metrics across different systems.

**Open questions:**

- Do \(K\), \(I_{\mathrm{local}}\), cones, and \(H_{\mathrm{eff}}\)
  behave similarly in NCAs, RL agents, and biological systems?
- Can we obtain biological data (e.g., from Levin lab) to test
  predictions about tissue-level intelligence?
- What normalization is needed to compare \(K\) across substrates?

**Proposed approach:**

After NCA experiments (Phases 1-3), apply the same metric protocols to:
1. RL agents in spatial navigation tasks
2. Biological data on regeneration or morphogenesis (if available)

**Success criterion:** Qualitatively similar patterns (e.g., cones
shrink with damage size, composition improves \(K\)) across substrates.
