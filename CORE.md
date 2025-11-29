
# 01_INTELLIGENCE: Formal Core

This document is the **formal core** of the project. It defines:

- environments \(\mathcal{E}\),
- goals \(G\),
- resources \(R\),
- policies/agents \(\pi\),
- performance \(\mathrm{Perf}\) and optimal performance \(\mathrm{Perf}^*\),
- the resource-bounded intelligence functional \(I(\pi; \mathcal{E}, G, R, w, \mu)\),
- a minimal replicator–agent link via an interpretation map \(D\).

All narrative motivation, examples, and detailed literature context are
collected in a separate `CONTEXT_AND_LITERATURE.md`. This file should
remain short, precise, and stable, serving as the shared mathematical
backbone for all other modules.

---

## 1. Core objects

We briefly define the basic ingredients; for detailed POMDP mappings see
`framework_to_pomdp_mapping.md`.

### 1.1 Environment class

We fix an **environment class** \(\mathcal{E}\).

Each environment \(e \in \mathcal{E}\) is a dynamical system that, when
coupled to an agent, generates trajectories of states, observations, and
actions over time.

For each \(e \in \mathcal{E}\) we assume:

- a state space \(S_e\),
- an observation space \(O_e\),
- an action space \(A_e\),
- a stochastic transition kernel governing state evolution,
- an observation kernel relating states to observations.

We write \(\mathrm{Traj}(e,\pi,R)\) for the (random) trajectory induced
when policy \(\pi\) interacts with environment \(e\) under resource
constraint \(R\).

### 1.2 Policies and agents

A **policy** \(\pi\) is a (possibly stochastic) mapping from interaction
histories to actions. For environment \(e\):

\[
\pi: (O_e \times A_e)^* \to \Delta(A_e),
\]

where \(\Delta(A_e)\) is the set of probability distributions over
\(A_e\). The policy may be non-stationary and history-dependent.

In this framework, **agents** are identified with policies together with
their resource constraints. When the context is clear, we freely call a
policy \(\pi\) an “agent”.

We denote by \(\Pi(R)\) the set of policies that respect a given
resource constraint \(R\).

### 1.3 Goals

A **goal functional** \(g\) for environment \(e\) is a functional

\[
g : \mathrm{Traj}(e) \to \mathbb{R},
\]

mapping full trajectories to a real-valued score. The score is
interpreted as a total return, utility, or degree of goal satisfaction.

We write \(G\) for a family of such goals, and assume that for each
\(g \in G\) and \(e \in \mathcal{E}\) it makes sense to evaluate
\(g(\mathrm{Traj}(e,\pi,R))\) for any resource-feasible policy \(\pi\).

### 1.4 Resources

A **resource vector** \(R\) encodes the constraints under which a policy
operates. It may contain components such as:

- time budget (horizon) \(T\),
- computational budget (e.g. number of steps of internal computation),
- energy or actuation budget,
- memory capacity,
- communication bandwidth (for multi-agent settings),
- data budget (for learning).

We treat \(R\) as an element of an abstract resource space
\(\mathcal{R}\). For each \(R\) we assume there is a well-defined set
\(\Pi(R)\) of policies that respect \(R\). We do not formalize internal
computation; \(R\) is an abstract knob for “what the policy is allowed
to do”.

---

## 2. Performance and optimal performance

Fix \(e \in \mathcal{E}\), \(g \in G\), and a resource vector \(R\).

### 2.1 Performance

The **performance** of policy \(\pi\) in environment \(e\) on goal
\(g\) under resources \(R\) is

\[
\mathrm{Perf}(\pi; e, g, R)
:= \mathbb{E}\big[g(\mathrm{Traj}(e,\pi,R))\big],
\]

where the expectation is over the randomness of the environment and the
policy (if stochastic).

The unit and scale of \(\mathrm{Perf}\) depend on \(g\); we normalize by
an optimal benchmark below.

### 2.2 Resource-bounded optimal performance

The **optimal performance** for environment \(e\) and goal \(g\) under
resources \(R\) is

\[
\mathrm{Perf}^*(e, g, R)
:= \sup_{\pi' \in \Pi(R)} \mathrm{Perf}(\pi'; e, g, R).
\]

We assume that for nontrivial goals \(\mathrm{Perf}^*(e,g,R) > 0\).

In general, \(\mathrm{Perf}^*\) depends on both the environment dynamics
and the resource constraint \(R\); changing \(R\) may change what
performance is achievable.

---

## 3. The resource-bounded intelligence functional

Let \(w\) be a probability measure over environments \(\mathcal{E}\),
and let \(\mu(g \mid e)\) be a conditional probability measure over
goals given environment \(e\).

### 3.1 Definition

The **resource-bounded intelligence** of policy \(\pi\) with respect to
environment class \(\mathcal{E}\), goal family \(G\), resource vector
\(R\), environment distribution \(w\), and goal distribution \(\mu\) is

\[
I(\pi; \mathcal{E}, G, R, w, \mu)
:= \int_{\mathcal{E}} \int_G
w(e)\,\mu(g \mid e)\,
\frac{\mathrm{Perf}(\pi; e, g, R)}
     {\mathrm{Perf}^*(e, g, R)}\,
dg\,de.
\]

The integrand is the **normalized competence** of \(\pi\) on goal \(g\)
in environment \(e\), under resources \(R\). The functional \(I\) is the
average of this normalized competence over the chosen environment and
goal distributions.

By construction and the assumption \(\mathrm{Perf}^* > 0\),
\(I(\pi; \cdot) \in [0,1]\) whenever the integrals exist.

### 3.2 Interpretation (formal)

Informally:

- \(I \approx 1\): the policy is near-optimal across the environment and
  goal distributions, given resources \(R\).
- \(I \approx 0\): the policy is near-worst across those distributions.
- Intermediate values: the policy attains some fraction of optimal
  performance on average.

Different choices of \((\mathcal{E},G,R,w,\mu)\) induce different
intelligence measures. All contextual interpretation (e.g. local niche
vs. broad generality) is handled in `CONTEXT_AND_LITERATURE.md`.

---

## 4. Replicators and agents

The framework above treats \(\pi\) as the primitive. For biological and
evolutionary applications it is useful to distinguish:

- **replicators**: heritable structures (e.g. genomes, programs) that
  persist and reproduce across episodes,
- **agents**: phenotypes or policies that interact with environments
  within episodes.

This section gives a minimal formal link.

### 4.1 Replicators

We denote replicators by \(r \in \mathcal{R}_{\mathrm{rep}}\), a set of
heritable structures (e.g. genotypes, code strings, program graphs).

For evolutionary analysis we can define a **fitness functional**

\[
F(r; \mathcal{E}^{\mathrm{rep}}, R^{\mathrm{rep}}),
\]

which measures the long-run reproductive success of replicator \(r\)
under an evolutionary environment class \(\mathcal{E}^{\mathrm{rep}}\)
and evolutionary resources \(R^{\mathrm{rep}}\).

The exact form of \(F\) is application-specific; it is typically defined
over distributions of reproductive outcomes rather than per-episode
trajectories.

### 4.2 Interpretation map

To connect replicators to agents, we posit an **interpretation map**

\[
D : \mathcal{R}_{\mathrm{rep}} \to \Pi,
\]

where \(D(r) = \pi_r\) is the policy induced by replicator \(r\) when
developed or instantiated in the relevant environment (e.g. an
organism’s phenotype, an evolved controller, or a compiled program).

Thus, each replicator \(r\) induces an agent \(\pi_r\) that can be
evaluated via the intelligence functional \(I(\cdot)\).

### 4.3 Fitness and intelligence

Given \(D\), we can compare:

- \(F(r; \mathcal{E}^{\mathrm{rep}},R^{\mathrm{rep}})\): how well \(r\)
  replicates over evolutionary time,
- \(I(\pi_r; \mathcal{E}, G, R, w, \mu)\): how well the induced agent
  \(\pi_r\) achieves goals in operational environments.

In general these need not coincide. However, in many settings we expect
a relationship of the form

\[
F(r; \mathcal{E}^{\mathrm{rep}},R^{\mathrm{rep}})
\text{ depends on }
I(\pi_r; \mathcal{E}, G, R, w, \mu)
\]

plus additional constraints (e.g. costs of complexity, mutational
robustness).

Symbiogenesis and multi-scale composition can then be treated as
operations on replicators and agents:

- at the replicator level via operators like \(S^{\mathrm{rep}}(r_1,\dots,r_k)\),
- at the agent level via composition operators \(S^{\mathrm{agent}}(\pi_1,\dots,\pi_k)\).

The latter are defined and analyzed in
`cognitive_light_cones_and_compositional_intelligence.md`.

---

## 5. Relation to other documents

This core file is intentionally compact. All of the following are
developed elsewhere:

- **Context and literature.**  
  Narrative motivation, comparisons to universal intelligence, bounded
  optimality, resource rationality, deep-learning scaling, morphogenesis,
  and concrete examples are collected in `CONTEXT_AND_LITERATURE.md`
  with explicit citations of the form (author, year)(url).

- **Cognitive light cones and composition.**  
  Physical and goal-based cognitive light cones, self-boundary
  predicates, composition operators, and a first composition theorem are
  developed in `cognitive_light_cones_and_compositional_intelligence.md`.

- **POMDP/Dec-POMDP mapping.**  
  The mapping between the objects \((\mathcal{E},G,R,\pi)\) and standard
  POMDP / Dec-POMDP constructs is given in
  `framework_to_pomdp_mapping.md`.

This separation allows us to keep `01_INTELLIGENCE.md` as a stable
mathematical kernel while refining context, examples, and literature
independently.
