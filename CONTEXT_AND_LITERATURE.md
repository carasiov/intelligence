# Context and Literature for the Resource-Bounded Theory of Intelligence

This document provides conceptual background and literature context for
the formal core defined in `01_INTELLIGENCE.md`. It explains:

- why we define intelligence as **normalized goal-achievement under resource constraints**,
- how this relates to existing work in AI, cognitive science, and biology,
- how different “regimes” of intelligence (local, task-specific, general, multi-scale) arise,
- how capability can be decomposed into components (generality, adaptivity, robustness, compositionality),
- what kinds of theorems and empirical results this framework is meant to support.

`01_INTELLIGENCE.md` should remain compact and formal; this document is
allowed to be explanatory and discursive.

---

## 1. From Shannon’s Information to Resource-Bounded Intelligence

Shannon’s information theory made a decisive move: instead of defining
“information” philosophically, it defined a quantitative functional of a
source and channel [(Shannon, 1948)](https://ieeexplore.ieee.org/document/6773024).
The meaning of messages was left out; the focus was on what could be
reliably transmitted.

The present theory makes an analogous move for **intelligence**:

- We do not try to formalize consciousness, understanding, or value.
- We instead quantify how well a policy \(\pi\) achieves specified goals
  \(G\) in environments \(\mathcal{E}\), given resources \(R\),
  relative to the best resource-bounded performance \(\mathrm{Perf}^*\).

Formally, the central object is the **resource-bounded intelligence functional**

\[
I(\pi; \mathcal{E}, G, R, w, \mu)
= \int_{\mathcal{E}} \int_G
w(e)\,\mu(g \mid e)\,
\frac{\mathrm{Perf}(\pi; e, g, R)}
     {\mathrm{Perf}^*(e, g, R)}\,
dg\,de,
\]

as defined in `CORE.md`.

This construction synthesizes and extends several lines of work:

- **Universal intelligence**: Legg & Hutter measure an agent’s ability
  to achieve reward across all computable environments under a
  complexity prior [(Legg & Hutter, 2007)](https://www.vetta.org/documents/legg-hutter-2007-universal-intelligence.pdf).
- **Bounded-optimal agents**: architectures optimized for expected
  utility under computational limits in a fixed environment class
  [(Russell & Subramanian, 1995)](https://arxiv.org/abs/cs/9505103).
- **Resource-rational analysis**: human cognition as approximately
  optimal use of limited computational resources
  [(Lieder & Griffiths, 2020)](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/resourcerational-analysis-understanding-human-cognition-as-the-optimal-use-of-limited-computational-resources/586866D9AD1D1EA7A1EECE217D392F4A).

Compared to these, our framework:

- **makes \(\mathcal{E}, G, R\) explicit**, rather than implicit in the prior or architecture,
- **normalizes by the resource-bounded optimum** \(\mathrm{Perf}^*\)
  rather than an unbounded ideal agent,
- is designed from the outset to be **substrate-neutral and multi-scale**,
  applicable to cells, tissues, organisms, collectives, and artificial systems.

### Problem spaces as a concrete instantiation

A useful concrete instantiation of the general \((\mathcal{E},G,R)\)
picture comes from **problem spaces**
\(P = \langle S, O, C, E, H\rangle\), as used in recent work on basal
cognition and multi-scale biological intelligence.

- \(S\): a state space (configurations the system can occupy).
- \(O\): a set of operators (elementary transitions between states).
- \(C\): constraints specifying which state–operator pairs are forbidden.
- \(E\): an evaluation functional assigning value or “goodness” to states
  (e.g. reward, fitness, morphogenetic error, or variational free energy).
- \(H\): an effective **prediction horizon** (how many operator steps
  ahead the system can meaningfully evaluate or predict).

In our language, a fixed problem space \(P\) corresponds to a particular
choice of environment \(e\), goal \(g\), and resource vector \(R\):

- \(S\) and \(O\) capture the relevant parts of the environment dynamics,
- \(C\) and \(H\) encode resource and feasibility constraints,
- \(E\) plays the role of a goal functional (possibly derived from a
  generative model, as in free-energy formulations).

The point of making \(C, E, H\) explicit is that **biological and
artificial agents can edit them**: relaxing constraints, changing
evaluation criteria, or extending horizon are all levers of
“meta-intelligence” discussed below.

### Two-metric picture: search compression vs. normalized performance

For a fixed problem \((e,g,R)\), there are two complementary ways to
evaluate an agent:

1. **Search compression vs. a blind baseline**.  
   Given a cost functional \(J(\cdot; e,g,R)\) and a blind policy
   \(\pi_{\mathrm{blind}}\), we can measure how many orders of magnitude
   of cost an agent \(\pi\) saves relative to blind search. This is the
   role of a quantity like
   \[
   K(\pi; e,g,R) = \log_{10} \frac{J(\pi_{\mathrm{blind}}; e,g,R)}
                                    {J(\pi; e,g,R)}.
   \]
   Higher \(K\) means stronger pruning of the search space compared to a
   max-entropy or otherwise unstructured baseline.

2. **Normalized performance vs. optimal**.  
   We can also ask what fraction of the best resource-feasible
   performance the agent achieves. In `01_INTELLIGENCE.md` this appears
   both as the simple ratio \(\mathrm{Perf} / \mathrm{Perf}^*\) and, in a
   baseline-anchored form, as a local intelligence
   \(I_{\mathrm{local}}(\pi; e,g,R)\) that equals 0 for the blind
   policy, 1 for an optimal policy, and can in principle be negative
   (worse than blind) or greater than 1 (better than our model of the
   optimum).

Together, \(K\) and \(I_{\mathrm{local}}\) place an agent at a point in a
**two-dimensional space**:

- \(K\) captures **how much the agent compresses search** in that
  problem space, relative to chance.
- \(I_{\mathrm{local}}\) captures **how close it comes to the best
  achievable performance** under the same resource constraints.

Different systems can therefore have very different profiles:

- A bacterium doing chemotaxis can have **high \(K\)** (huge pruning of
  molecular configuration space relative to random diffusion) and also
  **high \(I_{\mathrm{local}}\)** (evolution has pushed it close to its
  physical limits).
- A student learning chess can have **moderate \(K\)** (already much
  better than random play in a vast game tree) but **low
  \(I_{\mathrm{local}}\)** (far from grandmaster or optimal play).
- A thermostat can have **low \(K\)** (the problem space is tiny and
  simple), but **\(I_{\mathrm{local}} \approx 1\)** (it essentially
  saturates what is achievable in its niche).

The global functional \(I(\pi; \mathcal{E},G,R,w,\mu)\) aggregates such
local competences across \((e,g)\) drawn from \((w,\mu)\); the Levin-type
search-efficiency metric \(K\) and its optimal counterpart \(K_{\mathrm{opt}}\)
are best understood as **local, problem-specific ingredients** inside
that broader picture.

---

## 2. Regimes of Intelligence: Local, Task-Specific, General

There is no single “universal” choice of \((\mathcal{E}, G, R, w, \mu)\).
Different choices induce different intelligence measures, all of the same
mathematical form but with different interpretations.

Three useful regimes:

1. **Local ecological intelligence**

   - \(\mathcal{E}\) concentrates on an organism’s natural niche.
   - \(G\) encodes survival, reproduction, or homeostasis.
   - \(R\) reflects its biological limits (metabolism, neural resources, lifespan).
   - \(I\) measures how well the organism is adapted to its actual niche.

2. **Task-specific intelligence**

   - \(\mathcal{E}\) is a finite set of tasks or benchmarks (e.g. Atari games, language tasks).
   - \(G\) are task-specific metrics (score, accuracy, reward).
   - \(R\) encodes training/inference compute, data, and memory budgets.
   - \(I\) becomes a normalized benchmark score under fixed budgets.

3. **(Attempted) general intelligence**

   - \(\mathcal{E}\) is broad and heterogeneous, covering qualitatively
     different tasks and situations.
   - \(G\) combines multiple success criteria (physical, social, epistemic).
   - \(R\) reflects more global constraints on computation and data.
   - \(I\) approaches a “general capability” measure, but only relative
     to the chosen \(\mathcal{E}, G, R\).

This makes explicit that talking about “intelligence” without specifying
\((\mathcal{E}, G, R, w, \mu)\) is as underspecified as talking about
“capacity” without specifying a channel in information theory.

---

## 3. Multi-Scale and Compositional Intelligence

Real systems are organized across scales: molecules → organelles → cells
→ tissues → organs → organisms → groups. At each scale we can often
identify entities that:

- have state,
- receive inputs,
- produce outputs,
- and appear to pursue goals.

The same formal apparatus applies at each scale:

- At the cellular level, \(\mathcal{E}\) may be the local chemical and
  bioelectrical environment, \(G\) may encode homeostatic or
  morphogenetic setpoints, and \(R\) reflects metabolic and signaling
  limits.
- At the tissue or organ level, \(\mathcal{E}\) may include injury and
  regenerative contexts, \(G\) may encode target anatomies or
  physiological ranges.
- At the whole-organism level, \(\mathcal{E}\) includes behavioral and
  social environments, \(G\) includes behavioral goals, etc.
- At the collective level, agents might be ants, robots, or humans;
  \(\mathcal{E}\) includes resource distributions, communication
  networks, and tasks.

In Levin’s “multiscale competency architecture” picture, cells,
tissues, and organs are treated as agents that cooperate to reach target
anatomies and homeostatic goals [(Levin, 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10167196/).
Our framework captures this by:

- assigning each scale its own \((\mathcal{E}_k, G_k, R_k, w_k, \mu_k)\),
- evaluating \(I_k(\pi_k;\cdot)\) for agents at that scale,
- and then studying how these intelligences relate when agents are
  composed into larger structures.

The **composition operators** and **self-boundary predicates** defined in
`cognitive_light_cones_and_compositional_intelligence.md` make this
multi-scale composition mathematically explicit, including:

- when a set of sub-agents behaves as a single macro-agent, and
- how their **cognitive light cones** (regions of spacetime where goals
  can be competently pursued) expand or fragment under fusion and
  breakdown.

### Meta-level problem reformulation and representational creativity

At every scale, agents do not only search **within** a fixed problem
space; they also sometimes act on the **problem space itself**. In the
problem-space notation \(P = \langle S,O,C,E,H\rangle\), such meta-level
moves include:

- changing the granularity or variables of \(S\),
- adding or removing operators in \(O\),
- relaxing or tightening constraints \(C\),
- modifying the evaluation functional \(E\) (e.g. changing goals or
  priors),
- extending or shortening the effective horizon \(H\).

We can view these as transitions in a meta-space \(\mathcal{P}^{(2)}\)
whose states are problem spaces and whose operators are edits to
\((S,O,C,E,H)\). Searching well in \(\mathcal{P}^{(2)}\) corresponds to
**representational creativity** or **meta-intelligence**: finding
problem formulations in which downstream search becomes far more
efficient.

Formally, one could define meta-level analogues \(K^{(2)}, I^{(2)}\)
that score how well a system navigates \(\mathcal{P}^{(2)}\) relative to
a naive edit process, but a principled choice of “blind baseline” in
\(\mathcal{P}^{(2)}\) remains an open problem (see `OPEN_PROBLEMS.md`).

### Conscious macro-agents (functional hypothesis)

Within this multi-scale, compositional picture we can make a purely
**functional** proposal about consciousness.

A macro-agent at some scale \(\Sigma\) is a candidate **conscious
agent** (with respect to \(\Sigma\)) when three conditions jointly hold:

1. **Integration**.  
   The \(\mathsf{IsAgent}\) predicate holds for the set of components at
   scale \(\Sigma\): internally they are strongly coupled and externally
   they behave as a single policy with a well-defined cognitive light
   cone.

2. **Self-model \(M\)**.  
   The macro-agent carries an internal structure \(M\) that
   (approximately) represents:
   - its own boundary and controllable region (an internal model of its
     light cone),
   - its current goals or preferred trajectories,
   and that participates in control: the macro-policy effectively
   factors through \(M\), and interventions on \(M\) lead to systematic,
   interpretable changes in behavior.

3. **Coherence maintenance via \(M\)**.  
   There is an ongoing process that:
   - detects mismatches between what \(M\) predicts and what actually
     happens,
   - updates \(M\) to reduce these mismatches,
   - and uses the updated \(M\) to steer behavior back toward coherent,
     goal-consistent functioning.

On this view, unconscious homeostasis corresponds to cases where
integration holds but there is no self-model in the control loop, or
where \(M\) is too limited to represent boundaries and goals in a
flexible, counterfactual way. Conscious control corresponds to cases
where a rich \(M\) both *tracks* and *helps maintain* the macro-agent’s
own \(\mathsf{IsAgent}\) status.

This is deliberately a **functional** story: it concerns what
consciousness does in the organization of agents across scales, and is
silent on why or whether such processes are accompanied by subjective
experience.

---

## 4. Decomposing Capability: Components of Intelligence

The scalar functional \(I(\pi;\cdot)\) captures **normalized goal-achievement**,
but it is often useful to decompose capability into several components,
all definable within this framework.

Some useful axes:

1. **Capability (baseline \(I\))**

   - The basic intelligence score for a given \((\mathcal{E}, G, R, w, \mu)\).

2. **Generality**

   - Breadth/diversity of \(\mathcal{E}\) under \(w\).
   - Narrow benchmark suites vs. broad, heterogeneous distributions.
   - Can be quantified via structural properties of \(\mathcal{E}\) and
     diversity measures on \(w\).

3. **Adaptivity**

   - How well an agent maintains normalized performance when \(\mathcal{E}\)
     or \(G\) shift, given limited additional interaction.
   - Can be measured by comparing \(I\) before and after distribution
     shifts, under fixed adaptation resources.

4. **Multi-scale compositionality**

   - Gains (or losses) in \(I\) when sub-agents are assembled into a
     larger agent with its own goals.
   - Captures synergy and interference between parts.
   - Mathematically supported by the composition operator
     \(S^{\mathrm{agent}}\) and theorems about light-cone composition.

5. **Robustness**

   - Sensitivity of performance to rare, adversarial, or safety-critical
     environments drawn from tails of \(w\) or dedicated “risk subsets”.
   - Can be formalized via risk-sensitive or worst-case variants of \(\mathrm{Perf}\)
     and \(I\).

In modern ML, various lines of work already instantiate pieces of this
profile:

- **Goal-conditional generalization** via universal value function
  approximators [(Schaul et al., 2015)](https://dl.acm.org/doi/10.5555/3045118.3045258).
- **Fast adaptation** via meta-learning, e.g. model-agnostic meta-learning
  [(Finn et al., 2017)](https://arxiv.org/abs/1703.03400).
- **Modularity and compositional reasoning** via neural module networks
  [(Andreas et al., 2016)](https://openaccess.thecvf.com/content_cvpr_2016/papers/Andreas_Neural_Module_Networks_CVPR_2016_paper.pdf).

Our framework gives a common language for these components, grounded in
the same basic \(I(\pi;\cdot)\).

---

## 5. Targets for Theorems and Empirical Laws

The purpose of fixing a precise \(I(\pi;\mathcal{E},G,R,w,\mu)\) is to
support concrete **theorems and scaling laws**, not just a slogan.

Some of the main targets:

1. **Limit theorems**

   - Upper bounds on achievable \(I\) as a function of resource budgets
     and environment complexity.
   - Trade-offs between compute, data, memory, time for fixed
     environment classes.
   - For embodied agents in space: light-cone-based limits on the size
     of regions that can be controlled or covered with given speed,
     sensing radius, reach, and time horizon.

2. **Scaling relations**

   - How \(I\) changes when different components of \(R\) are scaled:
     parameters, data, compute, agent density, etc.
   - Identification of regimes (data-limited, compute-limited,
     representation-limited) and transitions between them.

   Empirical neural scaling laws show, for example, that language-model
   loss often follows power-law scaling with model and dataset size
   [(Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361).
   Theoretical work connects these to variance- and resolution-limited
   regimes [(Bahri et al., 2021)](https://arxiv.org/abs/2102.06701).

3. **Multi-scale composition**

   - Conditions under which composing agents into larger structures
     improves intelligence at higher scales, and how this depends on
     communication and alignment.
   - Explicit **composition laws** for cognitive light cones, like the
     1D corridor result where team capacity scales approximately
     linearly with the number of agents until saturating the domain.

4. **Robustness and adaptivity**

   - How quickly resource-bounded agents can adapt to distribution
     shifts, for given adaptation budgets.
   - How much robustness can be traded off against average performance.

These aims guide the development of child theories (e.g. spatial capacity
theorems, composition laws) and empirical studies.

---

## 6. Examples and Case Studies (Pointers)

This section briefly recalls example domains; the full technical details
live in dedicated documents (e.g. `paper.pdf`, your ML notes, etc.).

### 6.1 Spatial agents and cognitive light cones

For 1D/2D detect-and-reach agents:

- \(\mathcal{E}\): spatial domains with static targets.
- \(G\): detect-and-reach objectives by time \(T\).
- \(R\): speed \(v\), sensing radius \(r\), reach radius \(a\), horizon \(T\),
  number of agents \(k\).

The **capturable set** \(C^*(R)\) (where \(\mathrm{Perf}^* = 1\)) is
bounded by explicit functions of \((v,r,a,T)\) in distinct regimes
(detection-limited, coupled, reach-limited). In 1D multi-agent settings
with full communication, capacity scales approximately as

\[
|C_k^*(R)| \approx \min(N, k\,|C_1^*(R)|),
\]

giving a first explicit **composition law** for cognitive light cones.

Here, goal-based cognitive light cones coincide with capturable regions,
so capacity bounds are directly geometric statements about
\(L^{\mathrm{goal}}_T\).

### 6.2 Modern ML systems as resource-bounded agents

Deep learning systems can be mapped into this framework by:

- treating the training/evaluation process as the environment class,
- defining goals from predictive performance, robustness, OOD detection,
  sample efficiency, etc.,
- expressing compute, model size, and data as components of \(R\),
- viewing the trained model (plus its training algorithm) as the policy \(\pi\).

Examples include:

- semi-supervised mixture VAEs with OOD detection and interactive
  labeling,
- large language models with prompt-based and fine-tuning capabilities,
- nested-learning architectures where multiple learning processes coexist
  at different timescales (meta-learning, optimizers, memory systems).

The same \(I(\pi;\cdot)\) can be used, in principle, to compare such
systems on common environments and resource budgets.

### 6.3 Basal cognition: chemotaxis and regeneration

Recent work on **amoeboid chemotaxis** and **planarian regeneration**
provides concrete case studies for the problem-space and two-metric
picture.

- In amoeboid chemotaxis (e.g. *Dictyostelium*), the problem space
  includes spatial positions and internal signaling states, operators
  are protrusive moves, constraints come from cell mechanics, and the
  prediction horizon \(H\) is effectively very short (decisions are
  myopic). Empirical modeling suggests a blind-relative search
  efficiency \(K \approx 2\): the real cell is about \(10^2\) times more
  efficient than a max-entropy baseline in that space.

- In planarian regeneration under strong perturbations (e.g. barium
  treatments), the problem space is a high-dimensional combination of
  gene-expression and morphology; operators are transcriptional and
  morphogenetic changes; constraints include developmental and
  mechanical limits; \(E\) encodes distance to a target body plan, and
  the effective horizon \(H\) (in units of cell-cycle updates) can be
  very large. Only a small fraction of the transcriptome actually moves
  during successful regeneration, indicating highly targeted search.
  Modeling suggests \(K \approx 21\), i.e. about \(10^{21}\)-fold
  compression relative to blind search.

These examples:

- instantiate the *same* problem-space and two-metric ideas in real
  biological systems,
- provide evidence that tissue- or organism-level agents achieve search
  efficiencies unavailable to independent cells,
- and offer concrete anchors for the “multi-scale competency
  architecture” story and our own light-cone and composition picture.

---

## 7. How This Document Relates to the Core

To keep the project organized:

- `01_INTELLIGENCE.md`  
  contains **formal definitions**:
  - environments, goals, resources, policies,
  - performance, optimal performance,
  - the intelligence functional \(I\),
  - minimal replicator–agent linkage.

- `cognitive_light_cones_and_compositional_intelligence.md`  
  contains the **geometric and compositional core**:
  - physical and goal-based cognitive light cones,
  - self-boundary predicates \(\mathsf{IsAgent}\),
  - composition operators \(S^{\mathrm{agent}}\),
  - a concrete 1D composition theorem,
  - symbiogenesis interpretation.

- **This document (`Context_and_Literature.md`)**  
  collects **motivation, narrative, and references**:
  - conceptual framing,
  - how the framework fits into prior work,
  - typical regimes and capability components,
  - scientific aims and examples.

When updating the theory, you can keep the formal pieces tight and use
this file to accumulate and revise the conceptual and bibliographic
context.

---

## 8. Selected References

- [(Shannon, 1948)](https://ieeexplore.ieee.org/document/6773024). A mathematical theory of communication.
- [(Legg & Hutter, 2007)](https://www.vetta.org/documents/legg-hutter-2007-universal-intelligence.pdf). Universal intelligence: A definition of machine intelligence.
- [(Russell & Subramanian, 1995)](https://arxiv.org/abs/cs/9505103). Provably bounded-optimal agents.
- [(Lieder & Griffiths, 2020)](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/resourcerational-analysis-understanding-human-cognition-as-the-optimal-use-of-limited-computational-resources/586866D9AD1D1EA7A1EECE217D392F4A). Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources.
- [(Levin, 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10167196/). Technological approach to mind everywhere.
- [(Schaul et al., 2015)](https://dl.acm.org/doi/10.5555/3045118.3045258). Universal value function approximators.
- [(Finn et al., 2017)](https://arxiv.org/abs/1703.03400). Model-agnostic meta-learning for fast adaptation of deep networks.
- [(Andreas et al., 2016)](https://openaccess.thecvf.com/content_cvpr_2016/papers/Andreas_Neural_Module_Networks_CVPR_2016_paper.pdf). Neural module networks.
- [(Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361). Scaling laws for neural language models.
- [(Bahri et al., 2021)](https://arxiv.org/abs/2102.06701). Explaining neural scaling laws.
- [(Chis-Ciure & Levin, 2025)](https://link.springer.com/article/10.1007/s11229-025-05319-6).
  Cognition all the way down 2.0: problem spaces and the scaling of biological intelligence.
- [(Fields & Levin, 2025)](https://www.tandfonline.com/doi/pdf/10.1080/19420889.2025.2466017).
  Life, its origin, and its distribution: a perspective from the Conway–Kochen Theorem and the Free Energy Principle.

Recent work grounded in the Free Energy Principle argues that all
persistent physical systems exhibit Bayesian satisficing
[(Fields & Levin, 2025)](https://www.tandfonline.com/doi/pdf/10.1080/19420889.2025.2466017).
If this view is correct, the question is not *whether* a system is
cognitive, but *how effectively* it achieves goals under resource
constraints. The intelligence functional
\(I(\pi; \mathcal{E}, G, R, w, \mu)\) provides a quantitative answer to
this question, applicable across substrates and scales.
