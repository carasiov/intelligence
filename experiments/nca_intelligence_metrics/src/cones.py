"""
Cognitive light cone estimation.

Implements the competence function C(x, t_d, δ) from EXPERIMENT_SPEC.md:
- Probe grid of damage locations and times
- Estimate recovery probability
- Visualize cone structure
"""

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial
from typing import Tuple, NamedTuple, List
from dataclasses import dataclass
import numpy as np

from .nca import NCAConfig, rollout, apply_damage, make_seed


@dataclass
class ConeEstimationConfig:
    """Configuration for cone estimation."""
    spatial_grid_size: int = 5      # 5x5 = 25 probe locations
    time_points: int = 5            # 5 damage times
    damage_sizes: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)
    n_trials: int = 20              # Trials per condition
    recovery_threshold: float = 1.5  # MSE ratio to consider "recovered"
    

@dataclass
class ConeEstimate:
    """Results of cone estimation for one trained NCA."""
    
    # Competence array: (n_x, n_y, n_t, n_delta)
    competence: np.ndarray
    
    # Probe coordinates
    x_coords: np.ndarray      # Spatial x coordinates
    y_coords: np.ndarray      # Spatial y coordinates
    t_coords: np.ndarray      # Damage time coordinates
    delta_values: np.ndarray  # Damage size values
    
    # Configuration used
    config: ConeEstimationConfig
    nca_config: NCAConfig
    
    # Reference costs
    undamaged_cost: float     # Cost without any damage
    
    def get_cone_at_delta(self, delta_idx: int, threshold: float = 0.5) -> np.ndarray:
        """
        Get binary cone mask at a specific damage size.
        
        Args:
            delta_idx: index into damage_sizes
            threshold: competence threshold for "inside cone"
            
        Returns:
            mask: (n_x, n_y, n_t) binary array
        """
        return self.competence[:, :, :, delta_idx] >= threshold
    
    def get_cone_size(self, delta_idx: int, threshold: float = 0.5) -> float:
        """Get fraction of probed spacetime inside the cone."""
        mask = self.get_cone_at_delta(delta_idx, threshold)
        return float(mask.sum()) / mask.size


def make_probe_grid(
    nca_config: NCAConfig,
    cone_config: ConeEstimationConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create grid of probe locations and times.
    
    Returns:
        x_coords: (n_x,) array of x coordinates
        y_coords: (n_y,) array of y coordinates  
        t_coords: (n_t,) array of damage times
    """
    # Spatial grid: spread across grid, avoiding edges
    margin = nca_config.grid_size // 8
    x_coords = np.linspace(
        margin, 
        nca_config.grid_size - margin, 
        cone_config.spatial_grid_size
    ).astype(int)
    y_coords = x_coords.copy()
    
    # Time grid: spread across rollout, avoiding very end
    # Assume T=64 by default; will be parameterized
    T = 64  # TODO: parameterize
    t_coords = np.linspace(0, T * 0.8, cone_config.time_points).astype(int)
    
    return x_coords, y_coords, t_coords


def estimate_recovery_probability(
    params: dict,
    nca_config: NCAConfig,
    target: jnp.ndarray,
    damage_center: Tuple[int, int],
    damage_time: int,
    damage_size: float,
    key: jnp.ndarray,
    n_trials: int,
    n_steps: int,
    undamaged_cost: float,
    recovery_threshold: float = 1.5
) -> Tuple[float, float]:
    """
    Estimate probability of recovery from damage.
    
    Args:
        params: trained NCA parameters
        nca_config: NCA configuration
        target: target pattern
        damage_center: (x, y) center of damage
        damage_time: step at which to apply damage
        damage_size: fraction of grid to damage (delta)
        key: PRNG key
        n_trials: number of trials
        n_steps: total rollout steps
        undamaged_cost: MSE of undamaged rollout (for comparison)
        recovery_threshold: max ratio to undamaged to count as "recovered"
        
    Returns:
        recovery_prob: fraction of trials that recovered
        mean_cost: average final cost across trials
    """
    keys = random.split(key, n_trials)
    seed = make_seed(nca_config)
    
    costs = []
    recoveries = 0
    
    for trial_key in keys:
        # Split key for pre-damage and post-damage rollouts
        key1, key2 = random.split(trial_key)
        
        # Run until damage time
        if damage_time > 0:
            state = rollout(params, seed, nca_config, key1, damage_time)
        else:
            state = seed
        
        # Apply damage
        damaged_state = apply_damage(state, damage_center, damage_size, nca_config)
        
        # Continue rollout
        remaining_steps = n_steps - damage_time
        if remaining_steps > 0:
            final_state = rollout(params, damaged_state, nca_config, key2, remaining_steps)
        else:
            final_state = damaged_state
        
        # Compute cost
        visible = final_state[:, :, :nca_config.n_visible]
        cost = float(jnp.mean((visible - target) ** 2))
        costs.append(cost)
        
        # Check recovery
        if cost <= undamaged_cost * recovery_threshold:
            recoveries += 1
    
    return recoveries / n_trials, float(np.mean(costs))


def estimate_cone(
    params: dict,
    nca_config: NCAConfig,
    target: jnp.ndarray,
    cone_config: ConeEstimationConfig,
    key: jnp.ndarray,
    n_steps: int = 64,
    verbose: bool = True
) -> ConeEstimate:
    """
    Estimate cognitive light cone for a trained NCA.
    
    Args:
        params: trained NCA parameters
        nca_config: NCA configuration
        target: target pattern
        cone_config: cone estimation configuration
        key: PRNG key
        n_steps: total rollout steps
        verbose: print progress
        
    Returns:
        ConeEstimate with competence values at all probe points
    """
    # Get probe coordinates
    x_coords, y_coords, t_coords = make_probe_grid(nca_config, cone_config)
    delta_values = np.array(cone_config.damage_sizes)
    
    # Compute undamaged cost for reference
    seed = make_seed(nca_config)
    key, subkey = random.split(key)
    final_undamaged = rollout(params, seed, nca_config, subkey, n_steps)
    visible = final_undamaged[:, :, :nca_config.n_visible]
    undamaged_cost = float(jnp.mean((visible - target) ** 2))
    
    if verbose:
        print(f"Undamaged cost: {undamaged_cost:.6f}")
    
    # Initialize competence array
    n_x = len(x_coords)
    n_y = len(y_coords)
    n_t = len(t_coords)
    n_delta = len(delta_values)
    
    competence = np.zeros((n_x, n_y, n_t, n_delta))
    
    # Estimate at each probe point
    total_probes = n_x * n_y * n_t * n_delta
    probe_count = 0
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            for k, t_d in enumerate(t_coords):
                for l, delta in enumerate(delta_values):
                    key, subkey = random.split(key)
                    
                    prob, _ = estimate_recovery_probability(
                        params=params,
                        nca_config=nca_config,
                        target=target,
                        damage_center=(int(x), int(y)),
                        damage_time=int(t_d),
                        damage_size=delta,
                        key=subkey,
                        n_trials=cone_config.n_trials,
                        n_steps=n_steps,
                        undamaged_cost=undamaged_cost,
                        recovery_threshold=cone_config.recovery_threshold
                    )
                    
                    competence[i, j, k, l] = prob
                    probe_count += 1
                    
                    if verbose and probe_count % 100 == 0:
                        print(f"Progress: {probe_count}/{total_probes} probes")
    
    return ConeEstimate(
        competence=competence,
        x_coords=x_coords,
        y_coords=y_coords,
        t_coords=t_coords,
        delta_values=delta_values,
        config=cone_config,
        nca_config=nca_config,
        undamaged_cost=undamaged_cost
    )


# =============================================================================
# Visualization helpers
# =============================================================================

def cone_to_heatmap(
    cone: ConeEstimate,
    delta_idx: int,
    spatial_slice: str = 'center'
) -> Tuple[np.ndarray, str]:
    """
    Convert cone estimate to 2D heatmap for visualization.
    
    Args:
        cone: ConeEstimate object
        delta_idx: which damage size to visualize
        spatial_slice: 'center' (x at center), 'diagonal', or 'max' (max over space)
        
    Returns:
        heatmap: (n_spatial, n_t) array
        description: string describing the slice
    """
    C = cone.competence[:, :, :, delta_idx]  # (n_x, n_y, n_t)
    
    if spatial_slice == 'center':
        # Slice at center x and y
        cx = len(cone.x_coords) // 2
        cy = len(cone.y_coords) // 2
        # Take max over both dimensions to get (n_t,) then tile
        # Actually, let's do a 1D x slice at center y
        heatmap = C[:, cy, :]  # (n_x, n_t)
        desc = f"C(x, t) at y={cone.y_coords[cy]}, δ={cone.delta_values[delta_idx]:.2f}"
        
    elif spatial_slice == 'max':
        # Max over spatial dimensions
        heatmap = C.max(axis=(0, 1))  # (n_t,)
        heatmap = heatmap[None, :]  # (1, n_t) for plotting
        desc = f"max_{{x,y}} C(x,y,t) at δ={cone.delta_values[delta_idx]:.2f}"
        
    elif spatial_slice == 'diagonal':
        # Diagonal slice
        n = min(len(cone.x_coords), len(cone.y_coords))
        heatmap = np.array([C[i, i, :] for i in range(n)])
        desc = f"C(x=y, t) at δ={cone.delta_values[delta_idx]:.2f}"
        
    else:
        raise ValueError(f"Unknown spatial_slice: {spatial_slice}")
    
    return heatmap, desc


def summarize_cone_sizes(cone: ConeEstimate, threshold: float = 0.5) -> dict:
    """
    Summarize cone sizes across damage levels.
    
    Args:
        cone: ConeEstimate object
        threshold: competence threshold
        
    Returns:
        dict mapping delta -> cone size (fraction of probed space)
    """
    return {
        delta: cone.get_cone_size(i, threshold)
        for i, delta in enumerate(cone.delta_values)
    }
