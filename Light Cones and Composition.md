
# Cognitive Light Cones and Compositional Intelligence

This document develops the core geometric and compositional structures
used in the project:

- **cognitive light cones** (physical and goal-based);
- **self-boundary predicates** \(\mathsf{IsAgent}\);
- **composition operators** \(S^{\mathrm{agent}}\) that fuse agents into
  macro-agents;
- a first **composition theorem** in a simple 1D detect-and-reach
  environment;
- an interpretation of **symbiogenesis and multi-scale agency** in these
  terms.

The underlying objects (environments, goals, resources, policies,
performance, intelligence functional) are defined in
`01 INTELLIGENCE.md`. Here we assume that background and focus on
derived structures.

---

## 1. Setup and notation

We briefly restate key objects and notation from `01 INTELLIGENCE.md`.

- \(\mathcal{E}\): environment class. Each \(e \in \mathcal{E}\) has a
  state space \(S_e\), observation space \(O_e\), and action space
  \(A_e\). Interactions of a policy with \(e\) induce trajectories
  \(\mathrm{Traj}(e,\pi,R)\).
- \(G\): family of goal functionals
  \(g : \mathrm{Traj}(e) \to \mathbb{R}\).
- \(R \in \mathcal{R}\): resource vector specifying constraints
  (time, computation, energy, communication, …).
- \(\Pi(R)\): set of policies respecting resource constraint \(R\).
- \(\mathrm{Perf}(\pi; e, g, R)\): expected performance of \(\pi\) on
  goal \(g\) in environment \(e\) under \(R\).
- \(\mathrm{Perf}^*(e,g,R)\): resource-bounded optimal performance.
- \(I(\pi; \mathcal{E}, G, R, w, \mu)\): resource-bounded intelligence,
  defined as an average normalized performance over environment and goal
  distributions.

We treat policies \(\pi\) as **agents** and will write “agent \(\pi\)”
interchangeably with “policy \(\pi\)”.

When needed, we restrict attention to a single environment \(e\) or a
single environment class \(\mathcal{E}_0 \subseteq \mathcal{E}\).

---

## 2. Physical cognitive light cones

We now formalize **cognitive light cones** in physical spacetime as
regions where an agent can in principle obtain information or exert
influence under resource constraints.

### 2.1 Spacetime domain and events

For each environment \(e\), we assume there is an associated physical
spacetime domain

\[
\mathcal{X}_e \times [0,\infty),
\]

where \(\mathcal{X}_e\) is a spatial domain (e.g. a subset of
\(\mathbb{R}^d\)) and \(t \in [0,\infty)\) is time.

We will speak informally of “events” at spacetime points \((x,t)\).
Formally, an event is a property of the environment state, agent
configuration, or both, that is localized around \((x,t)\). The exact
formalism is model-dependent; for this document it suffices that:

- we can talk about how agent trajectories depend on what happens at
  \((x,t)\);
- we can talk about how agent actions affect what happens at \((x,t)\).

### 2.2 Observability cone

Intuitively, a spacetime point is in the **observability cone** if an
agent with a given embodiment and resources can, in principle, obtain
some information about what happens there.

Fix an environment \(e\), resource vector \(R\), and time horizon
\(T > 0\).

**Definition 2.1 (Observability cone).**  
The **observability cone** of agent \(\pi\) in environment \(e\) up to
time \(T\) under resources \(R\) is

\[
O_T(\pi; e, R)
:= \Big\{(x,t) \in \mathcal{X}_e \times [0,T] \;\Big|\;
\exists \pi' \in \Pi(R)
\text{ such that the observation distribution induced by }
\pi' \text{ depends nontrivially on the event at }(x,t)
\Big\}.
\]

Remarks:

- We quantify over all resource-feasible policies \(\pi' \in \Pi(R)\) to
  capture **potential observability** given embodiment and resources,
  rather than what a specific \(\pi\) currently does.
- “Depends nontrivially” can be formalized in information-theoretic
  terms (e.g. nonzero mutual information) in concrete models.

### 2.3 Reachability cone

Similarly, a point is in the **reachability cone** if the agent can, in
principle, influence what happens there.

**Definition 2.2 (Reachability cone).**  
The **reachability cone** of agent \(\pi\) in environment \(e\) up to
time \(T\) under resources \(R\) is

\[
R_T(\pi; e, R)
:= \Big\{(x,t) \in \mathcal{X}_e \times [0,T] \;\Big|\;
\exists \pi' \in \Pi(R)
\text{ such that running }\pi'
\text{ can change, with nonzero probability, the event at }(x,t)
\Big\}.
\]

Again, “can change” can be formalized via counterfactuals or
interventional distributions in specific models.

### 2.4 Physical cognitive light cone

**Definition 2.3 (Physical cognitive light cone).**  
The **cognitive light cone** of agent \(\pi\) in environment \(e\) up to
time \(T\) under resources \(R\) is

\[
L_T(\pi; e, R)
:= O_T(\pi; e, R) \cup R_T(\pi; e, R)
\subseteq \mathcal{X}_e \times [0,T].
\]

We may consider its size (e.g. Lebesgue measure) \(|L_T(\pi; e, R)|\) as
a coarse measure of the spacetime “extent” of the agent’s potential
cognitive reach under \(R\).

In later sections we consider **goal-based** refinements of this object.

---

## 3. Goal-based cognitive light cones (Levin-style)

Levin’s notion of a cognitive light cone emphasizes not just where an
agent can sense or act, but the **spatiotemporal scale of goals it can
competently pursue**. We formalize this via localized goals and a
competence threshold.

### 3.1 Localized goals

Fix environment \(e\), horizon \(T>0\), and resource vector \(R\).

We assume that for each spacetime point \((x,t) \in \mathcal{X}_e \times [0,T]\)
there exists a localized goal functional \(g_{x,t}\) that “targets”
\((x,t)\).

**Assumption 3.1 (Localized goal family).**  
For each \((x,t) \in \mathcal{X}_e \times [0,T]\) there is a goal
\(g_{x,t} \in G\) such that:

- \(g_{x,t}\) depends primarily on what happens in a small neighbourhood
  of \((x,t)\);
- the resource-bounded optimum satisfies
  \(\mathrm{Perf}^*(e,g_{x,t},R) > 0\).

The collection of all such goals is

\[
G^{\mathrm{loc}}_{e,T} := \{ g_{x,t} : (x,t) \in \mathcal{X}_e \times [0,T] \}.
\]

Examples:

- Search-and-capture: \(g_{x,t}\) rewards detecting and neutralizing a
  target at \(x\) by time \(t\).
- Morphogenesis: \(g_{x,t}\) rewards achieving a desired local
  anatomical configuration around \(x\) at time \(t\).
- Sensing: \(g_{x,t}\) rewards reducing uncertainty about a local
  variable at \((x,t)\).

### 3.2 Normalized competence

For each localized goal \(g_{x,t}\), define the **normalized competence**
of \(\pi\) as

\[
\kappa(\pi; e, g_{x,t}, R)
:= \frac{\mathrm{Perf}(\pi; e, g_{x,t}, R)}
        {\mathrm{Perf}^*(e, g_{x,t}, R)} \in [0,1].
\]

This is exactly the local contribution that appears in the intelligence
functional \(I(\pi;\cdot)\).

### 3.3 Goal-based cognitive light cone

Fix a competence threshold \(\theta \in (0,1]\).

**Definition 3.2 (Goal-based cognitive light cone).**  
The **goal-based cognitive light cone** of \(\pi\) in environment \(e\)
up to time \(T\) under resources \(R\) at threshold \(\theta\) is

\[
L^{\mathrm{goal}}_T(\pi; e, R, \theta)
:= \Big\{ (x,t) \in \mathcal{X}_e \times [0,T]
\;\Big|\;
\kappa(\pi; e, g_{x,t}, R) \ge \theta \Big\}.
\]

Interpretation: \(L^{\mathrm{goal}}_T\) is the region of spacetime where
\(\pi\) can **competently** pursue localized goals involving \((x,t)\)
with normalized performance at least \(\theta\).

### 3.4 Relation to physical light cones

We expect that to control or achieve a goal localized at \((x,t)\), the
agent must be able to sense and/or act near \((x,t)\).

**Assumption 3.2 (Local causality).**  
For each localized goal \(g_{x,t}\), any change in performance induced
by a policy \(\pi'\) relative to a baseline must arise via changes in
observations and/or effects in a neighbourhood of \((x,t)\) under the
resource constraint \(R\).

Under this assumption we have:

**Proposition 3.3 (Goal-cone contained in physical cone).**  
For any agent \(\pi\), environment \(e\), resources \(R\), horizon \(T\),
and threshold \(\theta \in (0,1]\),

\[
L^{\mathrm{goal}}_T(\pi; e, R, \theta)
\subseteq L_T(\pi; e, R).
\]

*Sketch.* If \((x,t)\) is outside the physical cone, then no
resource-feasible policy can make observations or environment outcomes
depend nontrivially on what happens at \((x,t)\). By local causality,
no policy can significantly affect the performance on \(g_{x,t}\), so
\(\kappa(\pi; e, g_{x,t}, R)\) cannot exceed a nontrivial threshold
\(\theta\). Hence \((x,t)\) cannot belong to \(L^{\mathrm{goal}}_T\).

Thus the physical cone \(L_T\) sets an outer bound on where meaningful
goal pursuit is possible; the goal-based cone \(L^{\mathrm{goal}}_T\)
picks out the subregion where the agent actually attains sufficient
competence.

### 3.5 Connection to the intelligence functional

Consider a fixed environment \(e\) and horizon \(T\). Let \(D\) be a
measurable subset of \(\mathcal{X}_e \times [0,T]\). Define a goal
distribution \(\mu^{\mathrm{loc}}\) as follows:

- sample \((x,t)\) uniformly from \(D\);
- set \(g = g_{x,t}\).

The intelligence functional restricted to localized goals in \(D\) is

\[
I^{\mathrm{loc}}(\pi; e, G^{\mathrm{loc}}_{e,T}, R, \mu^{\mathrm{loc}})
= \frac{1}{|D|} \int_{D}
\kappa(\pi; e, g_{x,t}, R)\,d(x,t).
\]

So over localized goals, intelligence is simply the **average normalized
competence over spacetime region \(D\)**.

The goal-based light cone determines, for each \(\theta\), the region
where competence exceeds \(\theta\); the intelligence functional is the
integrated view of these competencies over a chosen distribution of
localized goals.

---

## 4. Multi-agent systems and self-boundaries

We now formalize when a collection of agents can be regarded as a single
macro-agent. This captures fusion of agents (e.g. symbiogenesis, tightly
coordinated robot teams) and provides a notion of **self-boundary**.

### 4.1 Multi-agent systems and communication

Consider a fixed environment \(e\) with global observation space
\(O_e\) and action space \(A_e\).

A **multi-agent system** consists of agents
\(\{\pi_i\}_{i=1}^k\), each with their own local observation and action
channels \((O_{e,i}, A_{e,i})\), interacting with the shared
environment. The environment’s state evolution depends on the joint
actions \((a_1,\dots,a_k)\).

We summarize communication and coupling via a **communication matrix**
\(B \in \mathbb{R}^{k \times k}\), where \(B_{ij}\) encodes the effective
capacity (bandwidth, reliability, shared state) from agent \(i\) to
agent \(j\).

We do not fix the detailed semantics of \(B\); we only assume that:

- a larger \(B_{ij}\) means more reliable/fast/expressive communication;
- the graph induced by nonzero entries in \(B\) describes who can
  influence whose internal state.

### 4.2 Macro-policies and external indistinguishability

From the environment’s point of view, a multi-agent system induces a
distribution over global trajectories. This motivates:

**Definition 4.1 (Macro-policy induced by a multi-agent system).**  
A **macro-policy** \(\Pi\) for environment \(e\) is a policy in the
sense of `01 INTELLIGENCE.md`, mapping global observation histories to
actions in \(A_e\).

We say that a multi-agent system \((\{\pi_i\}, B)\) **induces** a
macro-policy \(\Pi\) if the trajectory distribution of the environment
under the team equals that under \(\Pi\):

\[
\mathrm{Traj}(e, \{\pi_i\}, B) \equiv \mathrm{Traj}(e, \Pi)
\]

as distributions (up to null sets).

### 4.3 Self-boundary predicate \(\mathsf{IsAgent}\)

We now define a predicate that says when a collection of agents behaves,
for relevant purposes, as a single agent.

**Definition 4.2 (Self-boundary predicate).**  
Given agents \(\{\pi_i\}_{i=1}^k\) with communication matrix \(B\) in
environment class \(\mathcal{E}_0 \subseteq \mathcal{E}\), a resource
regime \(R\), and a goal family \(G_0 \subseteq G\), we define

\[
\mathsf{IsAgent}(\{\pi_i\}, B; \mathcal{E}_0, G_0, R)
\]

to be **true** if the following hold:

1. **Connectivity.** The communication graph of \(B\) is strongly
   connected.
2. **Sufficient coupling.** The entries of \(B\) exceed some threshold
   \(b_{\min}\) (depending on \(\mathcal{E}_0, G_0, R\)) sufficient to
   coordinate actions for the goals of interest.
3. **External indistinguishability.** There exists a macro-policy
   \(\Pi\) such that for all \(e \in \mathcal{E}_0\) and goals
   \(g \in G_0\), the performance and normalized competence of the team
   equals that of \(\Pi\) under resources \(R\):
   \[
   \mathrm{Perf}(\{\pi_i\}, B; e, g, R)
   = \mathrm{Perf}(\Pi; e, g, R),
   \]
   and hence
   \[
   I(\{\pi_i\}, B; \mathcal{E}_0, G_0, R, w, \mu)
   = I(\Pi; \mathcal{E}_0, G_0, R, w, \mu)
   \]
   for any \(w,\mu\) supported on \(\mathcal{E}_0, G_0\).

If any of these conditions fail, we take
\(\mathsf{IsAgent}(\{\pi_i\}, B; \mathcal{E}_0, G_0, R) := \mathrm{false}\).

The precise form of the “sufficient coupling” condition (2) is
model-dependent; in concrete settings it can be derived from bandwidth,
latency, noise, and task requirements. The key point is that there is a
regime in which the group behaves, for the goals and environments of
interest, as if it were a single agent with policy \(\Pi\).

---

## 5. Composition operator \(S^{\mathrm{agent}}\)

Given a group of agents that satisfies the self-boundary predicate, we
define a composition operator that produces the corresponding
macro-agent.

**Definition 5.1 (Composition operator \(S^{\mathrm{agent}}\)).**  
Let \(\{\pi_i\}_{i=1}^k\) be agents with communication matrix \(B\),
considered over \((\mathcal{E}_0, G_0, R)\) as above. Suppose
\(\mathsf{IsAgent}(\{\pi_i\}, B; \mathcal{E}_0, G_0, R)\) is true, and
let \(\Pi\) be a macro-policy that witnesses the external
indistinguishability condition.

We define the **composition operator**

\[
S^{\mathrm{agent}}(\{\pi_i\}, B; \mathcal{E}_0, G_0, R)
:= \Pi.
\]

The macro-agent \(\Pi\) is then an ordinary agent (policy) that can be
evaluated via the intelligence functional \(I\) and equipped with
cognitive light cones \(L_T(\Pi;\cdot)\) and
\(L^{\mathrm{goal}}_T(\Pi;\cdot)\).

When \(\mathsf{IsAgent}\) is false, \(S^{\mathrm{agent}}\) is undefined
(or we say that no macro-agent exists at the chosen scale).

We can similarly define macro-resources \(R_{\mathrm{macro}}\) and
possibly a macro-environment class \(\mathcal{E}_{\mathrm{macro}}\) and
macro-goal family \(G_{\mathrm{macro}}\), but for the purposes of this
document it suffices to treat \(\Pi\) as a policy over the same
environment class \(\mathcal{E}_0\).

---

## 6. Example: 1D corridor capacity and composition

We now instantiate the above formalism in a simple 1D
detect-and-reach environment. This provides a concrete example of how
composition expands cognitive light cones and capacity.

### 6.1 Environment and single-agent capacity

**Assumption 6.1 (1D corridor environment).**  

- The environment is a 1D line segment \([0,N]\).
- There is a single static target at unknown position \(x^* \in [0,N]\).
- An agent starts from a known initial position (e.g. one endpoint or a
  specific interior point).
- The agent has:
  - maximum speed \(v\);
  - sensing radius \(r\) (it can detect the target if within distance
    \(r\));
  - actuation radius \(a\) (it can neutralize the target if within
    distance \(a\));
  - time horizon \(T\).

A **policy succeeds** if it detects and neutralizes the target by time
\(T\).

**Definition 6.2 (Single-agent capacity).**  
Let \(\pi\) be an agent with parameters \((v,r,a,T)\) in the corridor
environment. The **capacity** \(C_1\) of \(\pi\) is the maximal length
\(|C|\) of a subset \(C \subseteq [0,N]\) such that there exists a
policy (possibly \(\pi\) itself) that guarantees success for any target
position \(x^* \in C\) within time \(T\).

In the underlying spatial capacity work (see `paper.pdf`) one can
derive explicit formulas for \(C_1\) as a function of \((v,r,a,T)\). In
this document we only assume:

**Assumption 6.3.** For fixed \((v,r,a,T)\) there exists a well-defined
capacity \(C_1 \in [0,N]\).

### 6.2 Multi-agent team and team capacity

Now consider \(k\) identical agents with the same parameters
\((v,r,a,T)\), interacting with the same corridor environment.

We assume **full communication**:

**Assumption 6.4 (Full communication).**  

- The communication matrix \(B_{\mathrm{full}}\) has all entries above a
  high threshold \(b_{\min}\);
- communication is effectively instantaneous and reliable;
- the agents can share observations and coordinate their motion and
  actions.

The team capacity \(C_k\) is defined analogously to \(C_1\): maximal
length of a subset \(C \subseteq [0,N]\) such that the team can
guarantee success for any target position in \(C\) within time \(T\).

From the spatial capacity analysis (in the detection-limited regime), we
have the **team capacity law**

\[
C_k = \min(N, k\,C_1).
\]

We take this as a given result for the present document.

### 6.3 Localized capture goals and goal-based cones

We now define localized goals as in Section 3, specialized to the
corridor.

For each \(x \in [0,N]\), consider the environment instantiation with a
target at \(x\) and define

\[
g_x(\mathrm{Traj}) = 
\begin{cases}
1 & \text{if the trajectory detects and neutralizes the target by time } T,\\
0 & \text{otherwise.}
\end{cases}
\]

We treat \(g_x\) as a localized goal; here the time index is effectively
“by time \(T\)”, so the relevant domain is \([0,N]\) rather than full
spacetime.

- For each \(x\), the resource-bounded optimum satisfies
  \(\mathrm{Perf}^*(e,g_x,R) = 1\) (there exists a policy that guarantees
  success).
- For any deterministic guarantee-based policy \(\pi\),
  \(\mathrm{Perf}(\pi;e,g_x,R) \in \{0,1\}\).

Thus the normalized competence reduces to

\[
\kappa(\pi; e, g_x, R)
= \mathrm{Perf}(\pi; e, g_x, R) \in \{0,1\}.
\]

For any threshold \(\theta \in (0,1]\), the goal-based cone (now just a
subset of \([0,N]\)) is

\[
L^{\mathrm{goal}}_T(\pi; e, R, \theta)
= \{ x \in [0,N] : \kappa(\pi; e, g_x, R) \ge \theta \}
= \{ x \in [0,N] : \mathrm{Perf}(\pi; e, g_x, R) = 1 \}.
\]

By definition of capacity, there exists a maximal interval
\(C \subseteq [0,N]\) of length \(|C| = C_1\) such that \(\pi\) (or some
capacity-achieving policy) guarantees success for all \(x \in C\). Hence

\[
L^{\mathrm{goal}}_T(\pi; e, R, \theta) = C,
\qquad
|L^{\mathrm{goal}}_T(\pi; e, R, \theta)| = C_1.
\]

Thus in this example the **goal-based cone coincides with the capacity
region**.

### 6.4 Composite agent and composition theorem

Under Assumption 6.4, the self-boundary predicate holds for the
multi-agent team in the corridor environment:

\[
\mathsf{IsAgent}(\{\pi_i\}_{i=1}^k, B_{\mathrm{full}}; \{e\}, \{g_x\}, R) = \mathrm{true},
\]

for suitable choices of \(\mathcal{E}_0 = \{e\}\), localized goals
\(G_0 = \{g_x\}\), and resource regime \(R\).

Therefore we can define the **composite agent**

\[
\Pi := S^{\mathrm{agent}}(\{\pi_i\}_{i=1}^k, B_{\mathrm{full}}; \{e\}, \{g_x\}, R).
\]

By construction \(\Pi\) induces the same environment trajectories as the
coordinated team.

For \(\Pi\), the team capacity law implies

\[
|L^{\mathrm{goal}}_T(\Pi; e, R, \theta)| = C_k = \min(N, k\,C_1).
\]

We can summarize this as:

**Theorem 6.5 (Goal-cone composition law in the corridor).**  
In the 1D corridor environment with localized capture goals \(\{g_x\}\)
and threshold \(\theta \in (0,1]\):

- For a single agent \(\pi\) with capacity \(C_1\),
  \(|L^{\mathrm{goal}}_T(\pi; e, R, \theta)| = C_1\).
- For the composite agent
  \(\Pi = S^{\mathrm{agent}}(\{\pi_i\}, B_{\mathrm{full}}; \{e\}, \{g_x\}, R)\)
  formed from \(k\) identical agents under full communication,
  \(|L^{\mathrm{goal}}_T(\Pi; e, R, \theta)| = C_k = \min(N,k\,C_1)\).

Thus, in this setting, composition under strong coupling expands the
goal-based cognitive light cone linearly in the number of agents, up to
saturation at the environment size \(N\).

---

## 7. Light-cone composition: qualitative rules

The corridor example suggests general qualitative rules for how
cognitive light cones compose under fusion and fragment under loss of
coupling.

Let \(\{\pi_i\}_{i=1}^k\) be agents in environment \(e\) with resources
\(R_i\) and cones \(L_T(\pi_i; e, R_i)\),
\(L^{\mathrm{goal}}_T(\pi_i; e, R_i, \theta)\). Suppose
\(\mathsf{IsAgent}(\{\pi_i\}, B; \mathcal{E}_0,G_0,R)\) holds and
\(\Pi = S^{\mathrm{agent}}(\{\pi_i\}, B; \mathcal{E}_0,G_0,R)\).

### 7.1 Monotonicity

We expect that composition is **monotone** in the sense that

\[
L_T(\Pi; e, R) \supseteq \bigcup_{i=1}^k L_T(\pi_i; e, R_i),
\]

and similarly for goal-based cones at suitable thresholds,

\[
L^{\mathrm{goal}}_T(\Pi; e, R, \theta)
\supseteq \bigcup_{i=1}^k L^{\mathrm{goal}}_T(\pi_i; e, R_i, \theta),
\]

provided resources aggregate reasonably (e.g. \(R\) at least as large as
each \(R_i\)).

Exact statements depend on the model; in the corridor example this
monotonicity holds in the detection-limited regime.

### 7.2 Approximate additivity and saturation

In symmetric settings where agents can cover disjoint regions
efficiently (as in the corridor), we expect approximate **additivity**
of cone size up to saturation:

\[
|L_T(\Pi; e, R)| \approx
\min\Big(|\mathcal{X}_e| T,\; \sum_{i=1}^k |L_T(\pi_i; e, R_i)|\Big),
\]

and similarly for goal-based cones,

\[
|L^{\mathrm{goal}}_T(\Pi; e, R, \theta)|
\approx
\min\Big(|\mathcal{X}_e| T,\; \sum_{i=1}^k |L^{\mathrm{goal}}_T(\pi_i; e, R_i, \theta)|\Big).
\]

The corridor theorem is the 1D static-target analogue of this pattern.

### 7.3 Fragmentation and cone collapse

If coupling weakens so that
\(\mathsf{IsAgent}(\{\pi_i\}, B; \mathcal{E}_0,G_0,R)\) becomes false,
the macro-agent \(\Pi\) ceases to exist at that scale and the system
fragments into clusters \(C_1,\dots,C_m\). For each cluster where
\(\mathsf{IsAgent}\) still holds we obtain a composite agent
\(\Pi^{(j)}\); otherwise we are left with individual agents.

Qualitatively, the large cone \(L_T(\Pi;\cdot)\) collapses into a
collection of smaller cones:

\[
L_T(\Pi; e, R)
\leadsto
\{ L_T(\Pi^{(j)}; e, R_{\Pi^{(j)}}) \}_j
\cup
\{ L_T(\pi_i; e, R_i) \text{ for uncoupled } i \}.
\]

This provides a geometric picture of **self fragmentation**, as seen in
biological pathologies (e.g. cancer) or breakdown of coordinated
collectives.

---

## 8. Symbiogenesis and multi-scale agents

We conclude with a brief interpretation of the above structures in terms
of symbiogenesis and multi-scale intelligence.

### 8.1 Symbiogenesis at the replicator level

At the **replicator level** (see `01 INTELLIGENCE.md`), symbiogenesis
can be modeled by an operator

\[
S^{\mathrm{rep}} : (r_1,\dots,r_k) \mapsto r_{\mathrm{new}},
\]

which combines genomes or programs into a new replicator. The new
replicator \(r_{\mathrm{new}}\) may:

- encode combined or novel capabilities;
- carry additional “glue information” specifying coordination,
  communication, and division of labor among subcomponents;
- achieve higher fitness \(F(r_{\mathrm{new}};\cdot)\) in environments
  where cooperation at larger scales is beneficial.

### 8.2 Symbiogenesis at the agent level

Via the interpretation map \(D\), replicators induce agents
\(\pi_r = D(r)\). Symbiogenesis at the replicator level then induces
**agent-level composition** via \(S^{\mathrm{agent}}\):

- low-level agents \(\{\pi_{r_i}\}\) are coupled via shared physiology,
  communication, and regulation (captured by \(B\));
- when coupling is strong and coherent enough,
  \(\mathsf{IsAgent}(\{\pi_{r_i}\},B;\dots)\) becomes true;
- we obtain a macro-agent
  \(\Pi = S^{\mathrm{agent}}(\{\pi_{r_i}\},B;\dots)\) with its own goals
  and resources.

Biologically, examples include:

- endosymbiotic origin of mitochondria and chloroplasts;
- the emergence of multicellular organisms from unicellular ancestors;
- the emergence of tissues, organs, and whole bodies as nested
  macro-agents built from cells.

### 8.3 Expansion of cognitive light cones across scales

At each fusion step, the macro-agent \(\Pi\) typically has:

- a larger physical cognitive light cone \(L_T(\Pi;\cdot)\), because it
  spans more space, time, and degrees of freedom;
- a larger goal-based light cone
  \(L^{\mathrm{goal}}_T(\Pi;\cdot,\theta)\), because it can pursue goals
  at larger spatiotemporal scales (e.g. whole-body morphology, long-term
  survival, colony-level tasks).

The composition theorem in the 1D corridor is a simple illustration of
this pattern: composing agents multiplies the capacity region until it
saturates the environment.

### 8.4 Fragmentation and pathological loss of self

Conversely, when communication or regulatory coupling fail (e.g. loss of
gap junctional connectivity among cells), the self-boundary can shrink:

- \(\mathsf{IsAgent}\) may fail at the organism scale;
- sub-agents (e.g. individual cells) revert to smaller-scale goals
  (e.g. unregulated proliferation);
- the macro-agent’s cone collapses and is replaced by many small,
  misaligned cones.

This provides a formal lens on phenomena such as cancer, breakdown of
organ systems, or the disintegration of social collectives.

### 8.5 Intelligence as emergent from nested composition

Within this framework, **intelligence** is always evaluated via the
normalized performance functional \(I(\cdot)\), but the **agents** to
which it applies can themselves be:

- primitive (single-cell, single-robot, single-software module);
- composite (tissue, organism, team, organization);
- nested and multi-scale (cells \(\to\) tissues \(\to\) organs
  \(\to\) organisms \(\to\) collectives).

Symbiogenesis and composition operators expand cognitive light cones
across scales, enabling larger and more remote goals to be pursued,
while the intelligence functional measures how effectively these
expanded capacities are actually used under resource constraints.

This completes the core module that unifies:

- cognitive light cones (physical and goal-based),
- self-boundaries,
- composition,
- and symbiogenesis,

on top of the general intelligence functional defined in
`01 INTELLIGENCE.md`.
