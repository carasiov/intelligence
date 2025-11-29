# RCM-VAE: Responsibility-Conditioned Mixture VAE

## Project Status & Context

**Current Status:** Standalone Implementation

This directory contains the implementation and specifications for the **RCM-VAE**, a semi-supervised, interactive generative classifier where semantics emerge from latent space structure.

## Relationship to the Intelligence Framework

Although currently developed as a separate software project, this model is intended to eventually serve as a concrete instantiation and testbed for the **Resource-Bounded Intelligence** theory outlined in the parent directory (see `../AGENTS.md` and `../CORE.md`).

Future integration points include:

*   **Resource Constraints ($R$):** Using the model to study how constraints on channel capacity (sparsity penalties) and latent dimensionality affect "intelligence" (classification/generation performance).
*   **Local Competence:** Mapping the model's **responsibility** mechanism ($q(c|x)$) to the theoretical concept of **Cognitive Light Cones**—regions of the input space where specific "experts" (channels) possess high competence.
*   **Compositionality:** Viewing the mixture-of-experts architecture as a "macro-agent" composed of simpler sub-agents (channels), aligning with the composition theory in `../Light Cones and Composition.md`.

For now, please treat this directory as the practical lab bench for these theoretical concepts.
