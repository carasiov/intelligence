### Core idea

At the center is a **resource-bounded, substrate-neutral definition of intelligence**.

An agent is a policy (\pi) interacting with an environment class (\mathcal{E}), pursuing a family of goals (G) under resource constraints (R). For any such setup you define:

* a performance functional (\mathrm{Perf}(\pi; e,g,R)) on trajectories,
* the optimal achievable performance (\mathrm{Perf}^*(e,g,R)) over all resource-feasible policies (\Pi(R)),
* and an **intelligence functional**
  [
  I(\pi;\mathcal{E},G,R,w,\mu)
  ]
  that aggregates normalized performance (\mathrm{Perf}/\mathrm{Perf}^*) over environments (e\sim\mu) and goals (g\sim w). 

This is deliberately analogous to Shannon’s move in information theory: instead of defining “intelligence” philosophically, you define a quantitative functional over **(environments, goals, resources, policies)** and then study its structure.

---

### Context and regimes

In this view, “intelligence” is **not a single scalar in the void**. Its meaning and value depend on the choice of:

[
(\mathcal{E}, G, R, w, \mu).
]

Different choices describe different notions of intelligence:

* **Local ecological intelligence**: an organism (or controller) evaluated on a narrow class of environments and goals (its niche).
* **Task-specific / benchmark intelligence**: ML systems evaluated on particular task families or benchmarks.
* **Broad / general intelligence**: policies evaluated on heterogeneous environment classes with diverse goals.

Capability naturally decomposes along several axes:

* **Generality** – how broad (\mathcal{E}) is.
* **Adaptivity** – how performance behaves under shifts in (\mu).
* **Robustness** – how performance behaves in tails, adversarial conditions, and worst cases.
* **Compositionality / multi-scale structure** – what happens to (I) when you build larger agents out of smaller ones.

All of these are **derived views** of the same core object (I) under different ways of varying ((\mathcal{E},G,R,\mu)). 

---

### Formal core (CORE.md)

`CORE.md` provides the mathematical backbone: 

* **Environment classes** (\mathcal{E}) with state, action, and observation spaces and dynamics.
* **Policies** (\pi) as mappings from interaction histories to action distributions.
* **Goals** as functionals on trajectories, not just scalar reward functions on states.
* A **resource space** (R) and resource-feasible policy sets (\Pi(R)) that encode limits (time, computation, memory, communication, energy, etc.).
* A performance functional (\mathrm{Perf}(\pi;e,g,R)), optimal performance (\mathrm{Perf}^*(e,g,R)), and then the normalized intelligence functional (I).

The core also introduces a bridge from **replicators** (e.g. genomes, code) to agents:

* Each replicator (r) is mapped via an interpretation (D(r)) to an agent policy (\pi_r).
* This lets you talk about **evolutionary fitness** (F(r;\cdot)) and **operational intelligence** (I(\pi_r;\cdot)) in the same framework, and to reason about how **combining replicators** (symbiogenesis, code composition) induces new composite agents.

So the core is a **general language** for talking about any policy-bearing system under constraints, biological or artificial.

---

### Geometry: light cones and local competence

`Light Cones and Composition.md` adds a **geometric layer** by embedding agents in spacetime and distinguishing two kinds of light cone: 

1. **Physical light cones** – regions of spacetime the agent can in principle observe or influence given its physical speed, sensing reach, actuation range, and horizon.
2. **Goal-based cognitive light cones** – the subset of spacetime where the agent can **competently** achieve localized goals (i.e. where its normalized performance exceeds a threshold).

Under a locality assumption, you can prove that **goal-based cognitive cones always lie inside physical cones**: you cannot reliably control regions you cannot even physically reach or sense.

In this picture, the global intelligence functional (I) can be reinterpreted as an **average of local competencies** over regions of spacetime, weighted by (\mu) and (w). Roughly: “how bright is the agent’s goal-based light cone, on average, for the environments and goals we care about?”

A concrete 1D “corridor” model shows this is not just metaphor:

* Agents move along a line, must detect and reach targets within a time horizon, with bounded speed and sensing/actuation radii.
* One can derive explicit **capacity bounds** on the set of target locations that are reliably reachable.
* For multiple agents, there is a **team capacity** that scales approximately linearly with the number of agents until their cones overlap and the environment’s finite size forces saturation.

In that toy world, you get a **clean capacity law** and a clear “phase diagram” of regimes (detection-limited, coupled, reach-limited).

---

### Multi-agent structure, self-boundaries, and composition

The same geometric tools let you talk about **when a group of agents behaves like one agent**. 

* A **communication / interaction graph** (B) encodes who can talk to or influence whom.
* A predicate (\mathsf{IsAgent}(\mathcal{A})) characterizes sets of agents (\mathcal{A}) that are:

  * internally strongly connected,
  * sufficiently coupled (they exchange enough information / resources),
  * externally indistinguishable (to the rest of the world) from some macro-policy over the combined interface.

When (\mathsf{IsAgent}(\mathcal{A})) holds, a **composition operator** (S^{\text{agent}}) maps the team into a single **macro-agent policy** with its own light cone and its own intelligence functional (I). In corridor-like settings, this recovers the intuitive result that:

* team capacity ≈ “sum of individual capacities” while cones don’t overlap,
* then saturates at an environment-determined ceiling once the team effectively covers the whole space.

This provides a principled way to talk about **self-boundaries** (what counts as “one agent”) and how to move up and down levels of description.

---

### Multi-scale agency and symbiogenesis

With these pieces, you can view **cells, tissues, organs, organisms, and collectives** as agents at different scales:

* each with its own environment class (\mathcal{E}), goals (G), resources (R),
* each with its own intelligence functional (I) and light cone.

**Symbiogenesis** and fusion events (at the replicator level) correspond to forming new composite agents whose self-boundaries include multiple previous units. When conditions for (\mathsf{IsAgent}) are met, this enlarges the goal-based cognitive light cone and enables pursuit of **larger-scale goals**. Conversely, breakdowns in coupling (e.g. cancer, social fragmentation) correspond to **cone collapse** and a drop in higher-level (I).

This story is compatible with “everything is cognitive” perspectives (like the Free Energy Principle), but with a different emphasis:

* not “everything is Bayesian,” but “to what extent, and at what scales, does a system **actually achieve goals under resource constraints?**”

---

### Overall story

Taken together, the documents define:

* a **general, resource-bounded intelligence functional** (I(\pi;\mathcal{E},G,R,w,\mu)),
* a **geometric interpretation** via cognitive light cones and local competence in spacetime,
* a **multi-agent and multi-scale architecture** via self-boundaries and composition.

The aim is **not** just a slogan like “intelligence is goal-achievement.” It is to build a reusable, quantitative framework in which you can:

* prove **capacity bounds** (what regions of state/space/time can be reliably controlled under given resources?),
* derive **composition laws** (how does capability scale with more agents or more structure?),
* explore **trade-offs** (robustness vs efficiency, local vs global goals),
* and relate **very different systems**—ML models, single cells, tissues, organisms, collectives—within one coherent theory.

In that sense, the project is meant to play for intelligence the role Shannon’s theory played for communication: not to settle the philosophy, but to provide the mathematical stage on which concrete theorems and cross-domain comparisons can be made.
