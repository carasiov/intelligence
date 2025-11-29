"""
Effective prediction horizon (H_eff) estimation.

Implements the H_eff measurement from EXPERIMENT_SPEC.md §5:
- Train probes to predict future states from current hidden state
- Measure prediction error decay with horizon k
- Define H_eff as max k where prediction beats variance threshold
"""

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial
from typing import Tuple, List, Dict, NamedTuple
from dataclasses import dataclass
import numpy as np

from .nca import NCAConfig, rollout, make_seed


@dataclass
class HorizonConfig:
    """Configuration for H_eff estimation."""
    horizons: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)  # k values to test
    probe_hidden_dim: int = 128
    probe_train_steps: int = 1000
    probe_lr: float = 1e-3
    n_trajectories: int = 50        # Number of rollouts for training data
    train_fraction: float = 0.8     # Train/test split
    variance_threshold: float = 0.5 # H_eff threshold (E_k < θ * Var)


class PredictionProbe(nn.Module):
    """
    MLP probe to predict future visible state from current hidden state.
    """
    output_dim: int
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            hidden_state: (H, W, C_hidden) current hidden channels
            
        Returns:
            prediction: (H, W, output_dim) predicted future visible state
        """
        # Flatten spatial dimensions for global processing
        H, W, C = hidden_state.shape
        
        # Option 1: Per-cell MLP (preserves spatial structure)
        x = nn.Dense(self.hidden_dim)(hidden_state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        
        return x


def collect_trajectory_data(
    params: dict,
    nca_config: NCAConfig,
    key: jnp.ndarray,
    n_trajectories: int,
    n_steps: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Collect trajectory data for probe training.
    
    Args:
        params: NCA parameters
        nca_config: NCA configuration
        key: PRNG key
        n_trajectories: number of rollouts
        n_steps: steps per rollout
        
    Returns:
        hidden_states: (n_traj, T, H, W, C_hidden) hidden channel trajectories
        visible_states: (n_traj, T, H, W, C_visible) visible channel trajectories
    """
    keys = random.split(key, n_trajectories)
    seed = make_seed(nca_config)
    
    hidden_list = []
    visible_list = []
    
    for traj_key in keys:
        # Get full trajectory
        trajectory = rollout(
            params, seed, nca_config, traj_key, 
            n_steps, return_trajectory=True
        )  # (T+1, H, W, C)
        
        # Split into hidden and visible
        visible = trajectory[:, :, :, :nca_config.n_visible]
        hidden = trajectory[:, :, :, nca_config.n_visible:]
        
        hidden_list.append(hidden)
        visible_list.append(visible)
    
    return jnp.stack(hidden_list), jnp.stack(visible_list)


def prepare_probe_data(
    hidden_states: jnp.ndarray,
    visible_states: jnp.ndarray,
    horizon_k: int,
    train_fraction: float = 0.8
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Prepare training data for a specific horizon k.
    
    Creates (hidden_t, visible_{t+k}) pairs.
    
    Args:
        hidden_states: (n_traj, T, H, W, C_hidden)
        visible_states: (n_traj, T, H, W, C_visible)
        horizon_k: prediction horizon
        train_fraction: fraction for training (rest is test)
        
    Returns:
        X_train, Y_train, X_test, Y_test
    """
    n_traj, T, H, W, C_h = hidden_states.shape
    C_v = visible_states.shape[-1]
    
    # Create pairs
    X = []
    Y = []
    
    for traj_idx in range(n_traj):
        for t in range(T - horizon_k):
            X.append(hidden_states[traj_idx, t])
            Y.append(visible_states[traj_idx, t + horizon_k])
    
    X = jnp.stack(X)  # (N, H, W, C_h)
    Y = jnp.stack(Y)  # (N, H, W, C_v)
    
    # Train/test split
    N = len(X)
    n_train = int(N * train_fraction)
    
    # Shuffle
    perm = np.random.permutation(N)
    X = X[perm]
    Y = Y[perm]
    
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def train_probe(
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    config: HorizonConfig,
    key: jnp.ndarray
) -> Tuple[dict, List[float]]:
    """
    Train a prediction probe.
    
    Args:
        X_train: (N, H, W, C_hidden) input hidden states
        Y_train: (N, H, W, C_visible) target visible states
        config: horizon estimation config
        key: PRNG key
        
    Returns:
        params: trained probe parameters
        losses: training loss history
    """
    output_dim = Y_train.shape[-1]
    probe = PredictionProbe(output_dim=output_dim, hidden_dim=config.probe_hidden_dim)
    
    # Initialize
    dummy_input = X_train[0]
    params = probe.init(key, dummy_input)['params']
    
    # Optimizer
    tx = optax.adam(config.probe_lr)
    opt_state = tx.init(params)
    
    @jit
    def loss_fn(params, X, Y):
        pred = vmap(lambda x: probe.apply({'params': params}, x))(X)
        return jnp.mean((pred - Y) ** 2)
    
    @jit
    def update_step(params, opt_state, X, Y):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, Y)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Training loop
    losses = []
    batch_size = min(64, len(X_train))
    
    for step in range(config.probe_train_steps):
        # Sample batch
        idx = np.random.choice(len(X_train), batch_size, replace=False)
        X_batch = X_train[idx]
        Y_batch = Y_train[idx]
        
        params, opt_state, loss = update_step(params, opt_state, X_batch, Y_batch)
        losses.append(float(loss))
    
    return params, losses


def evaluate_probe(
    params: dict,
    X_test: jnp.ndarray,
    Y_test: jnp.ndarray,
    config: HorizonConfig
) -> Tuple[float, float]:
    """
    Evaluate probe on test set.
    
    Args:
        params: probe parameters
        X_test: (N, H, W, C_hidden) test inputs
        Y_test: (N, H, W, C_visible) test targets
        config: horizon estimation config
        
    Returns:
        mse: mean squared error on test set
        variance: variance of test targets (for normalization)
    """
    output_dim = Y_test.shape[-1]
    probe = PredictionProbe(output_dim=output_dim, hidden_dim=config.probe_hidden_dim)
    
    pred = vmap(lambda x: probe.apply({'params': params}, x))(X_test)
    mse = float(jnp.mean((pred - Y_test) ** 2))
    variance = float(jnp.var(Y_test))
    
    return mse, variance


@dataclass
class HorizonEstimate:
    """Results of H_eff estimation."""
    horizons: np.ndarray           # k values tested
    prediction_errors: np.ndarray   # E_k for each k
    target_variances: np.ndarray   # Var(s_{t+k}) for each k
    normalized_errors: np.ndarray  # E_k / Var for each k
    H_eff: int                     # Effective horizon
    threshold: float               # Threshold used
    
    def to_dict(self) -> dict:
        return {
            'horizons': self.horizons.tolist(),
            'prediction_errors': self.prediction_errors.tolist(),
            'target_variances': self.target_variances.tolist(),
            'normalized_errors': self.normalized_errors.tolist(),
            'H_eff': self.H_eff,
            'threshold': self.threshold
        }


def estimate_H_eff(
    nca_params: dict,
    nca_config: NCAConfig,
    config: HorizonConfig,
    key: jnp.ndarray,
    n_steps: int = 64,
    verbose: bool = True
) -> HorizonEstimate:
    """
    Estimate effective prediction horizon H_eff.
    
    Args:
        nca_params: trained NCA parameters
        nca_config: NCA configuration
        config: horizon estimation config
        key: PRNG key
        n_steps: NCA rollout steps
        verbose: print progress
        
    Returns:
        HorizonEstimate with all results
    """
    key, data_key = random.split(key)
    
    # Collect trajectory data
    if verbose:
        print("Collecting trajectory data...")
    
    hidden_states, visible_states = collect_trajectory_data(
        nca_params, nca_config, data_key, 
        config.n_trajectories, n_steps
    )
    
    if verbose:
        print(f"Collected {config.n_trajectories} trajectories of {n_steps} steps")
    
    # Test each horizon
    horizons = []
    errors = []
    variances = []
    
    for k in config.horizons:
        if k >= n_steps:
            if verbose:
                print(f"Skipping k={k} (>= n_steps)")
            continue
            
        if verbose:
            print(f"Testing horizon k={k}...")
        
        key, probe_key = random.split(key)
        
        # Prepare data
        X_train, Y_train, X_test, Y_test = prepare_probe_data(
            hidden_states, visible_states, k, config.train_fraction
        )
        
        if len(X_test) < 10:
            if verbose:
                print(f"  Skipping k={k}: not enough test data")
            continue
        
        # Train probe
        probe_params, _ = train_probe(X_train, Y_train, config, probe_key)
        
        # Evaluate
        mse, var = evaluate_probe(probe_params, X_test, Y_test, config)
        
        horizons.append(k)
        errors.append(mse)
        variances.append(var)
        
        if verbose:
            print(f"  E_{k} = {mse:.6f}, Var = {var:.6f}, E/Var = {mse/var:.3f}")
    
    horizons = np.array(horizons)
    errors = np.array(errors)
    variances = np.array(variances)
    normalized = errors / (variances + 1e-8)
    
    # Compute H_eff: largest k where normalized error < threshold
    passing = normalized < config.variance_threshold
    if passing.any():
        H_eff = int(horizons[passing].max())
    else:
        H_eff = 0
    
    if verbose:
        print(f"\nH_eff = {H_eff} (threshold = {config.variance_threshold})")
    
    return HorizonEstimate(
        horizons=horizons,
        prediction_errors=errors,
        target_variances=variances,
        normalized_errors=normalized,
        H_eff=H_eff,
        threshold=config.variance_threshold
    )
