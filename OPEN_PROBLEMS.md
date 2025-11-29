# Open Problems in the Resource-Bounded Theory of Intelligence

This document collects conceptual and technical questions that are not
yet resolved in the core framework.

---

## 1. Null models for meta-level problem-space search

We can define a meta-space \(\mathcal{P}^{(2)}\) whose states are problem
spaces \(P = \langle S,O,C,E,H\rangle\) and whose operators are edits to
\((S,O,C,E,H)\). Searching well in \(\mathcal{P}^{(2)}\) corresponds to
“representational creativity” or meta-intelligence.

**Open questions:**

- What should count as a “blind baseline” \(\pi^{(2)}_{\mathrm{blind}}\)
  in \(\mathcal{P}^{(2)}\)?
- How should we define \(\tau^{(2)}_{\mathrm{blind}}\) and
  \(K^{(2)}\) in a way that is not arbitrary?
- Are there natural priors over edits to \((S,O,C,E,H)\) that make sense
  biologically or computationally?

---

## 2. Operationalizing self-models \(M\)

The consciousness story relies on a self-model \(M\) that encodes a
macro-agent’s boundary, goals, and light cone, and that participates in
control.

**Open questions:**

- How can we empirically identify \(M\) in biological systems or
  artificial agents?
- What minimal dimensionality / expressiveness should \(M\) have to
  count as a genuine self-model, rather than a fixed parameter (like a
  thermostat setpoint)?
- How do we distinguish “mere internal state” from a structure that
  supports flexible, counterfactual control over the agent’s own
  boundary and goals?

---

## 3. Empirical discriminability vs IIT, GWT, etc.

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

The prediction horizon \(H\) is a distinct resource from time horizon
\(T\), but it is not straightforward to measure.

**Open questions:**

- How can we empirically estimate \(H\) for cells, tissues, organisms,
  or collectives? (E.g. through control-based tasks, predictive coding
  paradigms, or intervention experiments.)
- How does \(H\) scale across levels (molecules → cells → tissues →
  organisms → collectives)?
- Are there generic scaling laws relating \(H\) to other resources in
  \(R\) (compute, memory, communication)?

---

## 5. Measuring \(K\) and \(K_{\mathrm{opt}}\) in AI systems

The search-efficiency metric \(K\) is attractive conceptually but can be
difficult to estimate for modern AI systems.

**Open questions:**

- What should count as a “blind” baseline for large models and RL
  agents?
- How can we approximate \(\tau_{\mathrm{blind}}\) and \(\tau_{\mathrm{agent}}\)
  without running prohibitive numbers of random policies?
- How can corridor-style capacity bounds and other theorems be used to
  bound or estimate \(K_{\mathrm{opt}}\) in realistic benchmarks?

