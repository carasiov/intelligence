"""
Neural Cellular Automata implementation in JAX.

Design principles:
- Pure functions with explicit state
- Vectorized operations (no Python loops over cells)
- Stochastic updates via explicit PRNG keys
"""

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from flax import linen as nn
from flax.training import train_state
import optax
from typing import NamedTuple, Tuple, Optional
from functools import partial


# =============================================================================
# Configuration
# =============================================================================

class NCAConfig(NamedTuple):
    """NCA architecture and training configuration."""
    grid_size: int = 64
    n_channels: int = 16          # Total channels (visible + hidden)
    n_visible: int = 4            # RGBA visible channels
    hidden_dim: int = 128         # Hidden layer size in update rule
    kernel_size: int = 3          # Perception kernel size
    cell_fire_rate: float = 0.5   # Stochastic update probability
    
    @property
    def n_hidden(self) -> int:
        return self.n_channels - self.n_visible


# =============================================================================
# Perception: Sobel filters + identity
# =============================================================================

def get_perception_kernels() -> jnp.ndarray:
    """
    Create perception kernels: identity + Sobel-x + Sobel-y.
    
    Returns:
        kernels: (3, 3, 3) array of perception filters
    """
    identity = jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=jnp.float32)
    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32) / 8.0
    sobel_y = sobel_x.T
    return jnp.stack([identity, sobel_x, sobel_y], axis=-1)


def perceive(state: jnp.ndarray, config: NCAConfig) -> jnp.ndarray:
    """
    Apply perception kernels to get local neighborhood features.
    
    Args:
        state: (H, W, C) grid state
        config: NCA configuration
        
    Returns:
        perception: (H, W, C * 3) concatenated filter responses
    """
    kernels = get_perception_kernels()  # (3, 3, 3)
    
    # Apply each kernel to each channel
    # Input: (H, W, C), Kernel: (3, 3, 3)
    # Output: (H, W, C, 3) -> reshape to (H, W, C*3)
    
    def apply_kernel(channel, kernel):
        """Apply single kernel to single channel."""
        # Expand dims for conv: (H, W) -> (1, H, W, 1)
        x = channel[None, :, :, None]
        k = kernel[:, :, None, None]
        result = jax.lax.conv_general_dilated(
            x, k, window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        return result[0, :, :, 0]
    
    # Vectorize over channels and kernels
    perceptions = []
    for k_idx in range(3):
        kernel = kernels[:, :, k_idx]
        for c_idx in range(config.n_channels):
            channel = state[:, :, c_idx]
            perceptions.append(apply_kernel(channel, kernel))
    
    return jnp.stack(perceptions, axis=-1)


# =============================================================================
# Update Rule (the learned part)
# =============================================================================

class UpdateRule(nn.Module):
    """
    Learned update rule: perception -> hidden -> delta.
    
    This is a small MLP applied identically to each cell.
    """
    config: NCAConfig
    
    @nn.compact
    def __call__(self, perception: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            perception: (H, W, C*3) perception features
            
        Returns:
            delta: (H, W, C) state update
        """
        # perception: (H, W, C*3)
        x = nn.Dense(self.config.hidden_dim)(perception)
        x = nn.relu(x)
        delta = nn.Dense(self.config.n_channels, kernel_init=nn.initializers.zeros)(x)
        return delta


# =============================================================================
# NCA Step Function
# =============================================================================

def nca_step(
    params: dict,
    state: jnp.ndarray,
    config: NCAConfig,
    key: jnp.ndarray,
    alive_threshold: float = 0.1
) -> jnp.ndarray:
    """
    Single NCA update step.
    
    Args:
        params: UpdateRule parameters
        state: (H, W, C) current grid state
        config: NCA configuration
        key: PRNG key for stochastic updates
        alive_threshold: alpha threshold for "alive" cells
        
    Returns:
        new_state: (H, W, C) updated grid state
    """
    # 1. Perceive local neighborhood
    perception = perceive(state, config)
    
    # 2. Compute update
    update_rule = UpdateRule(config)
    delta = update_rule.apply({'params': params}, perception)
    
    # 3. Stochastic cell updates (fire rate mask)
    fire_mask = random.uniform(key, (config.grid_size, config.grid_size, 1))
    fire_mask = (fire_mask < config.cell_fire_rate).astype(jnp.float32)
    
    # 4. Apply update with mask
    new_state = state + delta * fire_mask
    
    # 5. Alive masking: cells with low alpha are "dead"
    # A cell is alive if it or any neighbor has alpha > threshold
    alpha = state[:, :, 3:4]  # Alpha channel
    
    # Max-pool to check if any neighbor is alive
    alive = nn.max_pool(
        alpha[None, :, :, :], 
        window_shape=(3, 3), 
        strides=(1, 1), 
        padding='SAME'
    )[0]
    alive = (alive > alive_threshold).astype(jnp.float32)
    
    new_state = new_state * alive
    
    return new_state


# =============================================================================
# Rollout: Run NCA for T steps
# =============================================================================

def rollout(
    params: dict,
    initial_state: jnp.ndarray,
    config: NCAConfig,
    key: jnp.ndarray,
    n_steps: int,
    return_trajectory: bool = False
) -> jnp.ndarray:
    """
    Run NCA forward for n_steps.
    
    Args:
        params: UpdateRule parameters
        initial_state: (H, W, C) initial grid
        config: NCA configuration
        key: PRNG key
        n_steps: number of steps to run
        return_trajectory: if True, return all intermediate states
        
    Returns:
        If return_trajectory: (T+1, H, W, C) full trajectory
        Else: (H, W, C) final state
    """
    keys = random.split(key, n_steps)
    
    def step_fn(state, key):
        new_state = nca_step(params, state, config, key)
        return new_state, state if return_trajectory else None
    
    final_state, trajectory = jax.lax.scan(step_fn, initial_state, keys)
    
    if return_trajectory:
        # Prepend initial state
        trajectory = jnp.concatenate([initial_state[None], trajectory], axis=0)
        return trajectory
    else:
        return final_state


# =============================================================================
# Seed: Initial state with a single "seed" cell
# =============================================================================

def make_seed(config: NCAConfig, center: bool = True) -> jnp.ndarray:
    """
    Create initial seed state: single cell with alpha=1 at center.
    
    Args:
        config: NCA configuration
        center: if True, place seed at center; else at (0, 0)
        
    Returns:
        seed: (H, W, C) initial state
    """
    state = jnp.zeros((config.grid_size, config.grid_size, config.n_channels))
    
    if center:
        cx, cy = config.grid_size // 2, config.grid_size // 2
    else:
        cx, cy = 0, 0
    
    # Set alpha (channel 3) to 1 at seed position
    state = state.at[cx, cy, 3].set(1.0)
    
    return state


# =============================================================================
# Damage: Apply damage to state
# =============================================================================

def apply_damage(
    state: jnp.ndarray,
    damage_center: Tuple[int, int],
    damage_size: float,
    config: NCAConfig
) -> jnp.ndarray:
    """
    Apply damage by zeroing out a square region.
    
    Args:
        state: (H, W, C) current state
        damage_center: (x, y) center of damage
        damage_size: fraction of grid to damage (delta in spec)
        config: NCA configuration
        
    Returns:
        damaged_state: (H, W, C) state with damage applied
    """
    # Compute damage region size
    side = int(jnp.sqrt(damage_size * config.grid_size ** 2))
    half = side // 2
    
    cx, cy = damage_center
    x_start = max(0, cx - half)
    x_end = min(config.grid_size, cx + half + 1)
    y_start = max(0, cy - half)
    y_end = min(config.grid_size, cy + half + 1)
    
    # Create damage mask
    mask = jnp.ones((config.grid_size, config.grid_size, 1))
    mask = mask.at[x_start:x_end, y_start:y_end, :].set(0.0)
    
    return state * mask


# =============================================================================
# Loss: MSE to target
# =============================================================================

def mse_loss(
    params: dict,
    initial_state: jnp.ndarray,
    target: jnp.ndarray,
    config: NCAConfig,
    key: jnp.ndarray,
    n_steps: int
) -> jnp.ndarray:
    """
    Compute MSE loss between final visible state and target.
    
    Args:
        params: UpdateRule parameters
        initial_state: (H, W, C) initial grid
        target: (H, W, 4) target RGBA image
        config: NCA configuration
        key: PRNG key
        n_steps: number of steps to run
        
    Returns:
        loss: scalar MSE loss
    """
    final_state = rollout(params, initial_state, config, key, n_steps)
    visible = final_state[:, :, :config.n_visible]
    return jnp.mean((visible - target) ** 2)


# =============================================================================
# Training utilities
# =============================================================================

def create_train_state(
    config: NCAConfig,
    key: jnp.ndarray,
    learning_rate: float = 1e-3
) -> train_state.TrainState:
    """
    Initialize training state with random parameters.
    
    Args:
        config: NCA configuration
        key: PRNG key for initialization
        learning_rate: optimizer learning rate
        
    Returns:
        state: Flax TrainState
    """
    # Create dummy input for initialization
    dummy_perception = jnp.zeros((config.grid_size, config.grid_size, config.n_channels * 3))
    
    update_rule = UpdateRule(config)
    params = update_rule.init(key, dummy_perception)['params']
    
    tx = optax.adam(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=update_rule.apply,
        params=params,
        tx=tx
    )


@partial(jit, static_argnums=(3, 5))
def train_step(
    state: train_state.TrainState,
    initial: jnp.ndarray,
    target: jnp.ndarray,
    config: NCAConfig,
    key: jnp.ndarray,
    n_steps: int
) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """
    Single training step.
    
    Args:
        state: current TrainState
        initial: (H, W, C) initial state
        target: (H, W, 4) target image
        config: NCA configuration
        key: PRNG key
        n_steps: steps per training iteration
        
    Returns:
        new_state: updated TrainState
        loss: scalar loss value
    """
    def loss_fn(params):
        return mse_loss(params, initial, target, config, key, n_steps)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss


# =============================================================================
# Batch operations for baseline estimation
# =============================================================================

@partial(jit, static_argnums=(2, 4))
def batch_rollout(
    params: dict,
    initial_states: jnp.ndarray,
    config: NCAConfig,
    keys: jnp.ndarray,
    n_steps: int
) -> jnp.ndarray:
    """
    Run rollouts for a batch of initial states.
    
    Args:
        params: UpdateRule parameters
        initial_states: (B, H, W, C) batch of initial states
        config: NCA configuration
        keys: (B,) PRNG keys
        n_steps: number of steps
        
    Returns:
        final_states: (B, H, W, C) batch of final states
    """
    rollout_fn = partial(rollout, config=config, n_steps=n_steps, return_trajectory=False)
    return vmap(lambda init, k: rollout_fn(params, init, k))(initial_states, keys)


def estimate_baseline_cost(
    config: NCAConfig,
    target: jnp.ndarray,
    key: jnp.ndarray,
    n_weight_samples: int = 50,
    n_steps: int = 64
) -> Tuple[float, float]:
    """
    Estimate J(π_blind) by averaging MSE over random weight samples.
    
    Args:
        config: NCA configuration
        target: (H, W, 4) target image
        key: PRNG key
        n_weight_samples: number of random weight initializations (N in spec)
        n_steps: rollout steps
        
    Returns:
        mean_cost: E[MSE] over random weights
        std_cost: std of MSE over random weights
    """
    keys = random.split(key, n_weight_samples * 2)
    init_keys = keys[:n_weight_samples]
    rollout_keys = keys[n_weight_samples:]
    
    seed = make_seed(config)
    costs = []
    
    for i in range(n_weight_samples):
        # Random initialization
        state = create_train_state(config, init_keys[i])
        
        # Rollout
        final = rollout(state.params, seed, config, rollout_keys[i], n_steps)
        visible = final[:, :, :config.n_visible]
        
        cost = jnp.mean((visible - target) ** 2)
        costs.append(float(cost))
    
    return float(jnp.mean(jnp.array(costs))), float(jnp.std(jnp.array(costs)))
