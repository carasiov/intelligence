"""
Phase 1.5 experiment runner.

Implements the minimal viable experiment from EXPERIMENT_SPEC.md §6.1:
1. Baseline estimation
2. Training with K, I_local logging
3. Resource and task ablations
4. Cone pilot
5. H_eff pilot
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import random
import json
import argparse
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

from src import (
    NCAConfig, 
    TrainingConfig,
    ConeEstimationConfig,
    HorizonConfig,
    train_nca,
    estimate_cone,
    estimate_H_eff,
    make_seed,
    rollout
)


# =============================================================================
# Target patterns
# =============================================================================

def make_simple_target(size: int = 64) -> jnp.ndarray:
    """
    Simple target: centered circle.
    """
    y, x = jnp.meshgrid(jnp.arange(size), jnp.arange(size))
    center = size // 2
    radius = size // 4
    
    dist = jnp.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = (dist < radius).astype(jnp.float32)
    
    # RGBA: red circle
    target = jnp.zeros((size, size, 4))
    target = target.at[:, :, 0].set(mask)       # R
    target = target.at[:, :, 3].set(mask)       # A
    
    return target


def make_complex_target(size: int = 64) -> jnp.ndarray:
    """
    Complex target: bilateral symmetric pattern with fine structure.
    """
    y, x = jnp.meshgrid(jnp.arange(size), jnp.arange(size))
    center = size // 2
    
    # Body: ellipse
    body = ((x - center) ** 2 / (size // 3) ** 2 + 
            (y - center) ** 2 / (size // 5) ** 2) < 1
    
    # Head: circle at top
    head_center_y = center - size // 4
    head = ((x - center) ** 2 + (y - head_center_y) ** 2) < (size // 8) ** 2
    
    # Eyes: two small circles (bilateral symmetric)
    eye_offset_x = size // 12
    eye_offset_y = center - size // 4
    left_eye = ((x - (center - eye_offset_x)) ** 2 + 
                (y - eye_offset_y) ** 2) < (size // 20) ** 2
    right_eye = ((x - (center + eye_offset_x)) ** 2 + 
                 (y - eye_offset_y) ** 2) < (size // 20) ** 2
    
    # Combine
    pattern = body | head
    eyes = left_eye | right_eye
    
    # RGBA
    target = jnp.zeros((size, size, 4))
    target = target.at[:, :, 0].set(pattern.astype(jnp.float32) * 0.8)  # R
    target = target.at[:, :, 1].set(pattern.astype(jnp.float32) * 0.5)  # G
    target = target.at[:, :, 2].set(eyes.astype(jnp.float32))            # B (eyes)
    target = target.at[:, :, 3].set(pattern.astype(jnp.float32))         # A
    
    return target


# =============================================================================
# Experiment runners
# =============================================================================

def run_baseline_experiment(
    output_dir: Path,
    key: jnp.ndarray,
    tasks: Dict[str, jnp.ndarray],
    nca_config: NCAConfig,
    train_config: TrainingConfig
) -> Dict[str, Any]:
    """
    Run Phase 1.5 core experiment.
    """
    results = {}
    
    for task_name, target in tasks.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")
        
        key, task_key = random.split(key)
        
        task_config = TrainingConfig(
            n_steps=train_config.n_steps,
            rollout_steps=train_config.rollout_steps,
            learning_rate=train_config.learning_rate,
            log_every=train_config.log_every,
            checkpoint_every=train_config.checkpoint_every,
            n_baseline_samples=train_config.n_baseline_samples,
            budget_levels=train_config.budget_levels,
            output_dir=str(output_dir),
            run_name=f"task_{task_name}"
        )
        
        state = train_nca(target, nca_config, task_config, task_key)
        
        results[task_name] = {
            'best_cost': state.best_cost,
            'K_opt_curve': state.budget_tracker.get_K_opt_curve(),
            'metric_history': state.metric_history,
            'baselines': {
                'J_blind_rw': state.baselines.J_blind_rw,
                'J_blind_triv': state.baselines.J_blind_triv,
            }
        }
    
    return results


def run_resource_ablation(
    output_dir: Path,
    key: jnp.ndarray,
    target: jnp.ndarray,
    channel_counts: list = [8, 16, 32],
    train_steps: int = 50000
) -> Dict[int, Any]:
    """
    Run resource ablation: vary channel count.
    """
    results = {}
    
    for n_channels in channel_counts:
        print(f"\n{'='*60}")
        print(f"Channels: {n_channels}")
        print(f"{'='*60}")
        
        key, ablation_key = random.split(key)
        
        nca_config = NCAConfig(n_channels=n_channels)
        train_config = TrainingConfig(
            n_steps=train_steps,
            output_dir=str(output_dir),
            run_name=f"channels_{n_channels}"
        )
        
        state = train_nca(target, nca_config, train_config, ablation_key)
        
        results[n_channels] = {
            'best_cost': state.best_cost,
            'K_opt_curve': state.budget_tracker.get_K_opt_curve(),
            'final_K': state.metric_history[-1]['K_vs_rw'] if state.metric_history else 0
        }
    
    return results


def run_cone_pilot(
    output_dir: Path,
    params: dict,
    nca_config: NCAConfig,
    target: jnp.ndarray,
    key: jnp.ndarray
) -> Dict[str, Any]:
    """
    Run cone estimation pilot.
    """
    print("\n" + "="*60)
    print("Cone Estimation Pilot")
    print("="*60)
    
    cone_config = ConeEstimationConfig(
        spatial_grid_size=5,
        time_points=5,
        damage_sizes=(0.1, 0.3),  # Just 2 sizes for pilot
        n_trials=10               # Fewer trials for pilot
    )
    
    cone = estimate_cone(
        params, nca_config, target, cone_config, key,
        n_steps=64, verbose=True
    )
    
    # Summarize
    from src.cones import summarize_cone_sizes
    sizes = summarize_cone_sizes(cone, threshold=0.5)
    
    print(f"\nCone sizes at threshold=0.5:")
    for delta, size in sizes.items():
        print(f"  δ={delta:.2f}: {size:.1%} of probed space")
    
    return {
        'cone_sizes': sizes,
        'undamaged_cost': cone.undamaged_cost
    }


def run_horizon_pilot(
    output_dir: Path,
    params: dict,
    nca_config: NCAConfig,
    key: jnp.ndarray,
    label: str = "trained"
) -> Dict[str, Any]:
    """
    Run H_eff estimation pilot.
    """
    print(f"\n{'='*60}")
    print(f"H_eff Estimation: {label}")
    print("="*60)
    
    horizon_config = HorizonConfig(
        horizons=(1, 2, 4, 8, 16, 32),
        n_trajectories=20,      # Fewer for pilot
        probe_train_steps=500   # Fewer for pilot
    )
    
    result = estimate_H_eff(
        params, nca_config, horizon_config, key,
        n_steps=64, verbose=True
    )
    
    return result.to_dict()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Phase 1.5 experiments")
    parser.add_argument("--output", type=str, default="./results", 
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--steps", type=int, default=50000,
                        help="Training steps per run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (fewer steps/samples)")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Fall back to /tmp if we can't write to the specified directory
        print(f"Warning: Cannot write to {output_dir}, using /tmp/nca_results instead")
        output_dir = Path("/tmp/nca_results")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    key = random.PRNGKey(args.seed)
    
    # Quick mode settings
    if args.quick:
        n_steps = 5000
        n_baseline_samples = 10
        log_every = 100
    else:
        n_steps = args.steps
        n_baseline_samples = 50
        log_every = 500
    
    # Create targets
    print("Creating target patterns...")
    simple_target = make_simple_target(64)
    complex_target = make_complex_target(64)
    
    # Save targets
    np.save(output_dir / "target_simple.npy", np.array(simple_target))
    np.save(output_dir / "target_complex.npy", np.array(complex_target))
    
    # Default configs
    nca_config = NCAConfig()
    train_config = TrainingConfig(
        n_steps=n_steps,
        n_baseline_samples=n_baseline_samples,
        log_every=log_every,
        checkpoint_every=n_steps // 5,
        output_dir=str(output_dir)
    )
    
    all_results = {}
    
    # 1. Baseline experiment: simple vs complex
    key, exp_key = random.split(key)
    task_results = run_baseline_experiment(
        output_dir, exp_key,
        tasks={'simple': simple_target, 'complex': complex_target},
        nca_config=nca_config,
        train_config=train_config
    )
    all_results['tasks'] = task_results
    
    # 2. Resource ablation
    key, ablation_key = random.split(key)
    if not args.quick:
        resource_results = run_resource_ablation(
            output_dir, ablation_key, complex_target,
            channel_counts=[8, 16],
            train_steps=n_steps
        )
        all_results['resource_ablation'] = resource_results
    
    # 3. Cone pilot (using best model from complex task)
    key, cone_key = random.split(key)
    # Load best params from complex task
    from src.training import TrainingState
    complex_run_dir = output_dir / "task_complex"
    # For now, we'll use the training state we just computed
    # In practice, you'd load from checkpoint
    
    # 4. H_eff pilot
    # TODO: Compare trained vs untrained
    
    # Save all results
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("Experiment complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
