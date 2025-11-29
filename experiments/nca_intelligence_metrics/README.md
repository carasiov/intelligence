# NCA Intelligence Metrics

Empirical validation of the Resource-Bounded Intelligence framework using Neural Cellular Automata.

## Quick Start (with Docker + GPU)

### Prerequisites
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- VS Code with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Run with VS Code Dev Container

1. Open this folder in VS Code
2. Click "Reopen in Container" when prompted (or F1 → "Dev Containers: Reopen in Container")
3. Wait for container to build (~2-5 min first time)
4. Run experiments:

```bash
# Quick test (5k steps, ~1 min on GPU)
python scripts/run_phase1.py --quick

# Full Phase 1.5 (50k steps, ~10 min on GPU)
python scripts/run_phase1.py --steps 50000

# Run tests
pytest
```

### Run with Docker directly

```bash
# Build
docker build -t nca-intel -f .devcontainer/Dockerfile .

# Run with GPU
docker run --gpus all -it -v $(pwd):/app nca-intel python scripts/run_phase1.py --quick
```

### Run with Poetry (if you have CUDA locally)

```bash
# Install
poetry install

# Verify GPU
poetry run python -c "import jax; print(jax.devices())"

# Run
poetry run python scripts/run_phase1.py --quick
```

## Project Structure

```
├── src/
│   ├── nca.py          # Neural Cellular Automata implementation
│   ├── metrics.py      # K, I_local, K_opt computation
│   ├── cones.py        # Cognitive light cone estimation
│   ├── probes.py       # H_eff (prediction horizon) measurement
│   └── training.py     # Training loop with metric logging
├── scripts/
│   └── run_phase1.py   # Main experiment runner
├── tests/
│   └── test_nca.py     # Unit tests
└── results/            # Output directory (gitignored)
```

## Experiments

### Phase 1.5: Core Metric Validation

Tests whether $K$, $I_{\text{local}}$, and cognitive light cones behave as predicted.

**Deliverables:**
- Training curves (loss, K, I_local vs steps)
- K_opt(B) vs search budget (looking for plateau)
- Resource ablation (K vs channel count)
- Task ablation (K_opt for simple vs complex patterns)
- Cone pilot (competence heatmaps, cone size vs damage size)

### Phase 2: Composition (planned)

Tests $\mathsf{IsAgent}$ predicate and composition laws.

### Phase 3: Self-Model (planned)

Tests representation, causal involvement, and coherence maintenance.

## References

See parent directory for theoretical framework:
- `../../AGENTS.md` - Framework summary
- `../../CORE.md` - Mathematical specification
- `../../Light Cones and Composition.md` - Geometric interpretation
- `./EXPERIMENT_SPEC.md` - Detailed experimental protocol
