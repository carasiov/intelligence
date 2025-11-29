# Running NCA Intelligence Metrics on GPU Cluster

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/carasiov/intelligence.git
cd intelligence/experiments/nca_intelligence_metrics

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install JAX with CUDA support (choose ONE based on your CUDA version)

# For CUDA 12.x (most modern clusters):
pip install --upgrade "jax[cuda12]"

# For CUDA 11.x (older clusters):
pip install --upgrade "jax[cuda11_cudnn86]"

# 4. Install other dependencies
pip install flax optax numpy matplotlib tqdm

# 5. Verify GPU is detected
python -c "import jax; print(f'Devices: {jax.devices()}')"
# Should show something like: Devices: [CudaDevice(id=0)]

# 6. Run the experiment
python ./scripts/run_phase1.py --steps 50000 --output ./results

# Or quick test first:
python ./scripts/run_phase1.py --quick --output ./results_quick
```

## SLURM Job Script

If your cluster uses SLURM, create a file `run_experiment.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=nca_intelligence
#SBATCH --output=nca_%j.out
#SBATCH --error=nca_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load modules (adjust for your cluster)
module load python/3.10
module load cuda/12.1

# Activate environment
cd $HOME/intelligence/experiments/nca_intelligence_metrics
source venv/bin/activate

# Verify GPU
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# Run experiment
python ./scripts/run_phase1.py --steps 100000 --output ./results

echo "Done!"
```

Submit with:
```bash
sbatch run_experiment.slurm
```

## Expected Performance

| Hardware | Steps/sec | Time for 50k steps |
|----------|-----------|-------------------|
| CPU (your laptop) | ~1 | ~14 hours |
| NVIDIA T4 | ~100 | ~8 minutes |
| NVIDIA A100 | ~500 | ~2 minutes |
| NVIDIA H100 | ~800 | ~1 minute |

## Troubleshooting

### "No GPU detected" / Falls back to CPU

```bash
# Check if CUDA is available
nvidia-smi

# Check JAX installation
python -c "import jax; print(jax.devices())"

# If it shows StreamExecutor or CPU, reinstall JAX:
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12]"
```

### CUDA version mismatch

```bash
# Check your CUDA version
nvcc --version
# or
nvidia-smi  # shows CUDA version in top right

# Install matching JAX version
# CUDA 12.x: pip install "jax[cuda12]"
# CUDA 11.8: pip install "jax[cuda11_cudnn86]"
```

### Out of memory

Reduce batch size or grid size:
```bash
# Edit src/nca.py, change grid_size from 64 to 32
# Or reduce n_baseline_samples in scripts/run_phase1.py
```

### Module not found errors

```bash
# Make sure you're in the right directory
cd ~/intelligence/experiments/nca_intelligence_metrics

# And environment is activated
source venv/bin/activate
```

## Monitoring Progress

```bash
# Watch the output file
tail -f nca_*.out

# Or if running interactively, output shows:
# Step   1000 | Loss: 0.007373 | K: 1.118 | I_local: 0.962 | K_opt: 1.398 | 487.3 steps/sec
#                                                                          ^^^^^^^^^^^^^^
#                                                                          This should be 100+ on GPU
```

## After Experiment Completes

Results will be in `./results/`:
```
results/
├── task_simple/
│   ├── metrics.json          # Training curves
│   └── checkpoint_*/         # Model weights
├── task_complex/
│   └── ...
├── target_simple.npy         # Target patterns
├── target_complex.npy
└── all_results.json          # Summary
```

Download results:
```bash
# From your local machine
scp -r cluster:~/intelligence/experiments/nca_intelligence_metrics/results ./
```
