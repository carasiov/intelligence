"""
Main training loop with metric logging.

Implements the training protocol from EXPERIMENT_SPEC.md:
- Train NCA on target pattern
- Log K, I_local at each checkpoint
- Track K_opt(B) vs budget
- Save checkpoints for cone/H_eff estimation
"""

import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state, checkpoints
import optax
from functools import partial
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

from .nca import (
    NCAConfig, 
    make_seed, 
    rollout, 
    mse_loss, 
    create_train_state, 
    train_step,
    estimate_baseline_cost
)
from .metrics import (
    BaselineEstimates,
    BudgetTracker,
    MetricSnapshot,
    compute_all_metrics,
    compute_K_opt
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training
    n_steps: int = 100000           # Total training steps
    rollout_steps: int = 64         # NCA steps per training iteration
    learning_rate: float = 1e-3
    
    # Logging
    log_every: int = 100            # Log metrics every N steps
    checkpoint_every: int = 10000   # Save checkpoint every N steps
    
    # Baseline estimation
    n_baseline_samples: int = 50    # Samples for π_blind estimation
    
    # Budget tracking
    budget_levels: Tuple[int, ...] = (10000, 50000, 100000, 200000)
    
    # Paths
    output_dir: str = "./results"
    run_name: str = "nca_run"


@dataclass
class TrainingState:
    """Full training state including metrics history."""
    train_state: train_state.TrainState
    step: int
    baselines: BaselineEstimates
    budget_tracker: BudgetTracker
    metric_history: List[Dict] = field(default_factory=list)
    best_cost: float = float('inf')
    best_params: Optional[dict] = None


def initialize_training(
    nca_config: NCAConfig,
    train_config: TrainingConfig,
    target: jnp.ndarray,
    key: jnp.ndarray
) -> TrainingState:
    """
    Initialize training state including baseline estimation.
    
    Args:
        nca_config: NCA configuration
        train_config: training configuration
        target: target pattern (H, W, 4)
        key: PRNG key
        
    Returns:
        TrainingState ready for training loop
    """
    print("Initializing training...")
    
    key, init_key, baseline_key = random.split(key, 3)
    
    # Initialize model
    ts = create_train_state(nca_config, init_key, train_config.learning_rate)
    
    # Estimate baselines
    print(f"Estimating π_blind (N={train_config.n_baseline_samples})...")
    J_blind_rw, J_blind_rw_std = estimate_baseline_cost(
        nca_config, target, baseline_key,
        n_weight_samples=train_config.n_baseline_samples,
        n_steps=train_config.rollout_steps
    )
    print(f"  J(π_blind_rw) = {J_blind_rw:.6f} ± {J_blind_rw_std:.6f}")
    
    # Trivial baseline (seed only)
    seed = make_seed(nca_config)
    visible_seed = seed[:, :, :nca_config.n_visible]
    J_blind_triv = float(jnp.mean((visible_seed - target) ** 2))
    print(f"  J(π_blind_triv) = {J_blind_triv:.6f}")
    
    # Initial cost (untrained model)
    key, eval_key = random.split(key)
    final_state = rollout(ts.params, seed, nca_config, eval_key, train_config.rollout_steps)
    visible = final_state[:, :, :nca_config.n_visible]
    initial_cost = float(jnp.mean((visible - target) ** 2))
    print(f"  J(π_initial) = {initial_cost:.6f}")
    
    # Create baseline estimates (use initial as "optimal" for now)
    baselines = BaselineEstimates(
        J_blind_rw=J_blind_rw,
        J_blind_rw_std=J_blind_rw_std,
        J_blind_triv=J_blind_triv,
        J_optimal_B=initial_cost,  # Will be updated during training
        budget_B=0
    )
    
    # Create budget tracker
    budget_tracker = BudgetTracker(
        list(train_config.budget_levels),
        J_blind_rw
    )
    
    return TrainingState(
        train_state=ts,
        step=0,
        baselines=baselines,
        budget_tracker=budget_tracker,
        metric_history=[],
        best_cost=initial_cost,
        best_params=ts.params
    )


def training_loop(
    state: TrainingState,
    nca_config: NCAConfig,
    train_config: TrainingConfig,
    target: jnp.ndarray,
    key: jnp.ndarray,
    verbose: bool = True
) -> TrainingState:
    """
    Main training loop.
    
    Args:
        state: initial training state
        nca_config: NCA configuration
        train_config: training configuration
        target: target pattern
        key: PRNG key
        verbose: print progress
        
    Returns:
        Updated TrainingState with trained model and metrics
    """
    output_dir = Path(train_config.output_dir) / train_config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seed = make_seed(nca_config)
    
    if verbose:
        print(f"\nStarting training for {train_config.n_steps} steps...")
        print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    for step in range(state.step, train_config.n_steps):
        key, step_key = random.split(key)
        
        # Training step
        state.train_state, loss = train_step(
            state.train_state, seed, target, nca_config, 
            step_key, train_config.rollout_steps
        )
        loss = float(loss)
        state.step = step + 1
        
        # Update best model
        if loss < state.best_cost:
            state.best_cost = loss
            state.best_params = state.train_state.params
            
            # Update baselines with new best
            state.baselines = BaselineEstimates(
                J_blind_rw=state.baselines.J_blind_rw,
                J_blind_rw_std=state.baselines.J_blind_rw_std,
                J_blind_triv=state.baselines.J_blind_triv,
                J_optimal_B=loss,
                budget_B=step + 1
            )
        
        # Update budget tracker
        state.budget_tracker.update(step + 1, loss)
        
        # Logging
        if (step + 1) % train_config.log_every == 0:
            metrics = compute_all_metrics(loss, state.baselines, step + 1)
            state.metric_history.append(metrics.to_dict())
            
            if verbose:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                print(
                    f"Step {step + 1:6d} | "
                    f"Loss: {loss:.6f} | "
                    f"K: {metrics.K_vs_rw:.3f} | "
                    f"I_local: {metrics.I_local:.3f} | "
                    f"K_opt: {metrics.K_opt:.3f} | "
                    f"{steps_per_sec:.1f} steps/sec"
                )
        
        # Checkpointing
        if (step + 1) % train_config.checkpoint_every == 0:
            ckpt_path = output_dir / f"checkpoint_{step + 1}"
            checkpoints.save_checkpoint(
                str(ckpt_path),
                state.train_state,
                step + 1,
                keep=3
            )
            
            # Save metrics
            with open(output_dir / "metrics.json", 'w') as f:
                json.dump({
                    'metric_history': state.metric_history,
                    'baselines': {
                        'J_blind_rw': state.baselines.J_blind_rw,
                        'J_blind_rw_std': state.baselines.J_blind_rw_std,
                        'J_blind_triv': state.baselines.J_blind_triv,
                        'J_optimal_B': state.baselines.J_optimal_B,
                        'budget_B': state.baselines.budget_B
                    },
                    'K_opt_curve': state.budget_tracker.get_K_opt_curve(),
                    'best_cost': state.best_cost
                }, f, indent=2)
            
            if verbose:
                print(f"  Saved checkpoint to {ckpt_path}")
    
    # Final save
    final_path = output_dir / "final"
    checkpoints.save_checkpoint(str(final_path), state.train_state, state.step)
    
    with open(output_dir / "metrics_final.json", 'w') as f:
        json.dump({
            'metric_history': state.metric_history,
            'baselines': {
                'J_blind_rw': state.baselines.J_blind_rw,
                'J_blind_rw_std': state.baselines.J_blind_rw_std,
                'J_blind_triv': state.baselines.J_blind_triv,
                'J_optimal_B': state.baselines.J_optimal_B,
                'budget_B': state.baselines.budget_B
            },
            'K_opt_curve': state.budget_tracker.get_K_opt_curve(),
            'best_cost': state.best_cost,
            'plateaued': state.budget_tracker.has_plateaued()
        }, f, indent=2)
    
    if verbose:
        total_time = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Best cost: {state.best_cost:.6f}")
        print(f"  Final K: {compute_all_metrics(state.best_cost, state.baselines).K_vs_rw:.3f}")
        print(f"  K_opt plateau: {state.budget_tracker.has_plateaued()}")
    
    return state


# =============================================================================
# Convenience functions
# =============================================================================

def train_nca(
    target: jnp.ndarray,
    nca_config: Optional[NCAConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    key: Optional[jnp.ndarray] = None
) -> TrainingState:
    """
    High-level training function.
    
    Args:
        target: (H, W, 4) target RGBA image
        nca_config: NCA configuration (uses defaults if None)
        train_config: training configuration (uses defaults if None)
        key: PRNG key (uses random seed if None)
        
    Returns:
        TrainingState with trained model
    """
    if nca_config is None:
        nca_config = NCAConfig()
    
    if train_config is None:
        train_config = TrainingConfig()
    
    if key is None:
        key = random.PRNGKey(42)
    
    key, init_key, train_key = random.split(key, 3)
    
    # Initialize
    state = initialize_training(nca_config, train_config, target, init_key)
    
    # Train
    state = training_loop(state, nca_config, train_config, target, train_key)
    
    return state
