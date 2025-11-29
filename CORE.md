
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

- time budget (interaction horizon) \(T\),
- prediction horizon \(H\) (how many steps into the future the agent can
  effectively predict or evaluate when choosing actions),
- computational budget (e.g. number of steps of internal computation),
- energy or actuation budget,
- memory capacity,
- communication bandwidth (for multi-agent settings),
- data budget (for learning).

The distinction between \(T\) and \(H\) is important: \(T\) is how long
the episode lasts (how much time the agent has), whereas \(H\) is how far
ahead the agent can “see” in its internal evaluation of policies or
trajectories. Simple agents often have \(H \approx 1\) even when \(T\) is
large; more sophisticated agents can have \(H \gg 1\) but still be limited
by a finite \(T\).

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

We assume that for nontrivial goals \(\mathrm{Perf}^*(e,g,R) > 0\), and
that goals are nonnegative in the sense that
\(g(\mathrm{Traj}(e,\pi,R)) \ge 0\) for all resource-feasible policies
\(\pi \in \Pi(R)\). Hence
\(\mathrm{Perf}(\pi; e,g,R) \in [0,\mathrm{Perf}^*(e,g,R)]\) for all
\(\pi \in \Pi(R)\).

In general, \(\mathrm{Perf}^*\) depends on both the environment dynamics
and the resource constraint \(R\); changing \(R\) may change what
performance is achievable.

### 2.3 Cost functionals and blind baselines

In many applications it is convenient to work with a **cost functional**
\(J(\pi; e,g,R)\) instead of performance, for example expected time or
energy to achieve a goal, or the negative of a reward. Formally, we may
take \(J\) to be any monotone transform of \(\mathrm{Perf}\) such that
larger values of \(J\) correspond to worse outcomes.

For a fixed \((e,g,R)\) we also distinguish two special policies:

- a **blind baseline** \(\pi_{\mathrm{blind}}\), typically a max-entropy
  or otherwise unstructured policy that ignores task-specific structure
  while respecting the same resource constraint \(R\);
- an **optimal** policy \(\pi^* \in \Pi(R)\) that attains
  \(\mathrm{Perf}^*(e,g,R)\) (when it exists), or approaches it in the
  sense of a maximizing sequence.

We then write \(J(\pi_{\mathrm{blind}}; e,g,R)\) and
\(J(\pi^*; e,g,R)\) for the corresponding costs. These quantities are
used below to define normalized local intelligence and search-efficiency
metrics; they do not affect the definition of the global intelligence
functional itself.

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

### 3.3 Local search efficiency, baselines, and their relation

For a fixed \((e,g,R)\) we often want to compare three objects:

- the **blind baseline** \(\pi_{\mathrm{blind}}\),
- the **agent of interest** \(\pi\),
- and an **optimal** policy \(\pi^* \in \Pi(R)\) (or a maximizing sequence),

all evaluated with respect to a cost functional
\(J(\cdot; e,g,R)\) as in §2.3. This leads to two complementary
local quantities.

1. The **blind-relative search efficiency**

   \[
   K(\pi; e,g,R)
   := \log_{10} \frac{J(\pi_{\mathrm{blind}}; e,g,R)}{J(\pi; e,g,R)},
   \]

   which measures how many orders of magnitude of cost \(\pi\) saves
   relative to the blind baseline on that particular problem.

2. The **baseline-anchored local intelligence**

   \[
   I_{\mathrm{local}}(\pi; e,g,R)
   := \frac{J(\pi_{\mathrm{blind}}; e,g,R) - J(\pi; e,g,R)}
            {J(\pi_{\mathrm{blind}}; e,g,R) - J(\pi^*; e,g,R)}.
   \]

   Here \(I_{\mathrm{local}} = 0\) for the blind policy,
   \(I_{\mathrm{local}} = 1\) for an optimal policy, and
   \(I_{\mathrm{local}}\) can be negative (worse than blind) or greater
   than 1 (better than our model of the optimum) when the modeling
   assumptions are violated.

The optimal blind-relative efficiency in \((e,g,R)\) is

\[
K_{\mathrm{opt}}(e,g,R)
:= \log_{10} \frac{J(\pi_{\mathrm{blind}}; e,g,R)}{J(\pi^*; e,g,R)}.
\]

These quantities are linked by the identity

\[
K(\pi; e,g,R)
= K_{\mathrm{opt}}(e,g,R)
  + \log_{10} I_{\mathrm{local}}(\pi; e,g,R),
\]

whenever the costs are finite and the logarithm is defined. Thus:

- \(K_{\mathrm{opt}}\) characterizes the **opportunity for
  intelligence** in a given problem (how much better than blind optimal
  performance could in principle be),
- \(I_{\mathrm{local}}\) measures the **fraction of that opportunity**
  that \(\pi\) actually realizes,
- \(K(\pi)\) records the **overall compression** of search cost
  achieved by \(\pi\) relative to blind.

The global functional \(I(\pi;\mathcal{E},G,R,w,\mu)\) can be viewed
as an aggregation of such local competences across \((e,g)\) drawn
from \((w,\mu)\). In contexts where a blind baseline is available it
is often natural to use \(I_{\mathrm{local}}\) as the normalized
performance factor inside that aggregation; in other contexts the
simpler ratio \(\mathrm{Perf} / \mathrm{Perf}^*\) may be used.

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
