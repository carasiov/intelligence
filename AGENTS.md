# High-Level Summary of the Resource-Bounded Intelligence Framework

### Core Idea

At the center is a **resource-bounded, substrate-neutral definition of intelligence**.

An agent is a policy (\pi) interacting with an environment class (\mathcal{E}), pursuing goals (G) under resource constraints (R). For each environment–goal pair ((e,g)) we define:

* a **performance functional** (\mathrm{Perf}(\pi; e,g,R)),
* the **optimal performance** (\mathrm{Perf}^*(e,g,R)) over all policies satisfying (R),
* a **cost functional** (J(\pi; e,g,R)), enabling comparison to a **blind baseline** (\pi_{\mathrm{blind}}) and an **optimal policy** (\pi^*).

This yields two complementary local measures:

1. **Search compression (vs. blind):**
   [
   K = \log_{10}\frac{J(\pi_{\mathrm{blind}})}{J(\pi)}.
   ]

2. **Normalized performance (vs. optimal):**
   [
   I_{\mathrm{local}} = \frac{J(\pi_{\mathrm{blind}})-J(\pi)}{J(\pi_{\mathrm{blind}})-J(\pi^*)}.
   ]

A global intelligence functional
[
I(\pi;\mathcal{E},G,R,w,\mu)
]
aggregates normalized performance across environment and goal distributions, analogous to Shannon’s aggregation over channels.

This defines intelligence not philosophically but as a **quantitative functional over (environments, goals, resources, policies)**.

---

### Context and Regimes

Intelligence depends entirely on the specification:
[
(\mathcal{E}, G, R, w, \mu).
]

Different choices describe:

* **Local ecological intelligence**
* **Task-specific / benchmark intelligence**
* **Broad / general intelligence**

Capability decomposes along natural axes:

* **Generality** (breadth of (\mathcal{E}))
* **Adaptivity** (how performance changes under (\mu)-shift)
* **Robustness** (behavior under worst-case or adversarial conditions)
* **Compositionality / multi-scale structure**
* **Prediction depth** (prediction horizon (H) as a distinct resource)

These are all **derived views** of the same underlying mathematical structure.

---

### Formal Core (CORE.md)

`CORE.md` provides the mathematical backbone:

* **Environment classes** with states, actions, observations, and dynamics.
* **Policies** (\pi) mapping histories to actions.
* **Goals** as trajectory-level functionals.
* **Resource vectors** (R) including time (T), prediction horizon (H), computation, memory, energy, communication.
* **Performance** (\mathrm{Perf}), **optimal performance** (\mathrm{Perf}^*), and the global **intelligence functional** (I).

A baseline-sensitive normalization links blind baselines, optimal policies, and the metrics (K), (K_{\mathrm{opt}}), and (I_{\mathrm{local}}) via:
[
K = K_{\mathrm{opt}} + \log_{10} I_{\mathrm{local}}.
]

A mapping from **replicators** (r) (genomes, code) to policies (\pi_r) connects **evolutionary fitness** and **operational intelligence**.

---

### Geometry: Light Cones and Local Competence

`Light Cones and Composition.md` introduces a geometric layer with:

1. **Physical light cones** — regions the agent can observe or influence.
2. **Goal-based cognitive light cones** — regions where competence exceeds a threshold.

Cognitive cones lie within physical cones. The global intelligence functional can be interpreted as an **average brightness** of local competence over spacetime.

A simple 1D corridor model yields explicit **capacity bounds** and **scaling laws**:

* single-agent detection/reach sets shrink with local sensing and finite (H),
* multi-agent capacity grows linearly before saturating once cones overlap.

This demonstrates that light cones are **not metaphors** but calculable geometric constraints.

---

### Multi-Agent Structure, Self-Boundaries, and Composition

A **communication graph** (B) and a predicate (\mathsf{IsAgent}(\mathcal{A})) define when a collection of agents behaves as one macro-agent:

* strong internal coupling,
* coherent external behavior,
* stable goal-based competence.

A **composition operator** (S^{\mathrm{agent}}) yields a macro-policy with its own light cone and intelligence.

Capacity laws follow:

* before cone overlap: **additive capability**,
* after overlap: **saturation**,
* under coupling failure: **cone collapse** and loss of macro-intelligence.

This formalizes **self-boundaries** and how new agents emerge at higher scales.

---

### Multi-Scale Agency, Symbiogenesis, and Self-Models

Cells, tissues, organisms, and collectives each have their own:

* problem spaces (P = \langle S,O,C,E,H \rangle),
* intelligence metrics,
* light cones.

**Symbiogenesis** and fusion events correspond to forming new macro-agents satisfying (\mathsf{IsAgent}). Breakdowns in coupling (e.g. cancer, social fragmentation) correspond to macro-agent collapse.

A macro-agent is **functionally conscious** when it also carries a **self-model** (M) that:

1. represents its own boundary, goals, and approximate light cone,
2. participates causally in control,
3. is maintained by coherence processes correcting mismatches between prediction and reality.

This is a functional account of consciousness as **maintenance of macro-agency**, not a claim about subjective experience.

---

### Overall Story

Taken together, the documents define:

* a **general intelligence functional** (I(\pi;\mathcal{E},G,R,w,\mu)),
* a complementary **search-efficiency axis** (K),
* a **geometric interpretation** via light cones,
* a **multi-scale composition theory** via (\mathsf{IsAgent}) and (S^{\mathrm{agent}}),
* a **functional self-model architecture** for conscious macro-agency.

The goal is analogous to Shannon’s programme:

> **Not to define intelligence philosophically, but to define a clean mathematical object that supports capacity bounds, theorems, and cross-domain comparisons.**

This enables:

* **capacity bounds** on controllability in space/time,
* **composition laws** for scaling up agents,
* **robust trade-off analysis** (efficiency vs robustness vs prediction depth),
* and **cross-domain comparison** of RL agents, neural cellular automata, cells, tissues, organisms, and collectives.

Intelligence becomes a **resource-bounded, multi-scale, geometric, and measurable quantity**.
