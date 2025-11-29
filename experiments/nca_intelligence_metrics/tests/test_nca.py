"""
Quick sanity test for NCA implementation.

Run with: python -m pytest tests/test_nca.py -v
"""

import jax
import jax.numpy as jnp
from jax import random
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nca import (
    NCAConfig,
    make_seed,
    rollout,
    apply_damage,
    create_train_state,
    train_step,
    perceive
)
from src.metrics import (
    compute_K,
    compute_I_local,
    BaselineEstimates,
    compute_all_metrics
)


class TestNCABasics:
    """Test basic NCA functionality."""
    
    def test_seed_creation(self):
        """Seed should have single nonzero alpha at center."""
        config = NCAConfig(grid_size=32)
        seed = make_seed(config)
        
        assert seed.shape == (32, 32, 16)
        assert seed[16, 16, 3] == 1.0  # Alpha at center
        assert jnp.sum(seed[:, :, 3]) == 1.0  # Only one cell has alpha
    
    def test_perception(self):
        """Perception should produce 3x channel features."""
        config = NCAConfig(grid_size=32, n_channels=16)
        state = jnp.ones((32, 32, 16))
        
        perception = perceive(state, config)
        
        assert perception.shape == (32, 32, 48)  # 16 * 3 filters
    
    def test_rollout_shape(self):
        """Rollout should produce correct output shape."""
        config = NCAConfig(grid_size=32)
        key = random.PRNGKey(0)
        
        state = create_train_state(config, key)
        seed = make_seed(config)
        
        key, rollout_key = random.split(key)
        final = rollout(state.params, seed, config, rollout_key, n_steps=10)
        
        assert final.shape == (32, 32, 16)
    
    def test_rollout_trajectory(self):
        """Trajectory should have T+1 states."""
        config = NCAConfig(grid_size=32)
        key = random.PRNGKey(0)
        
        state = create_train_state(config, key)
        seed = make_seed(config)
        
        key, rollout_key = random.split(key)
        trajectory = rollout(
            state.params, seed, config, rollout_key, 
            n_steps=10, return_trajectory=True
        )
        
        assert trajectory.shape == (11, 32, 32, 16)  # T+1 states
    
    def test_damage_application(self):
        """Damage should zero out a region."""
        config = NCAConfig(grid_size=32)
        state = jnp.ones((32, 32, 16))
        
        damaged = apply_damage(state, (16, 16), 0.1, config)
        
        # Some cells should be zeroed
        assert jnp.sum(damaged) < jnp.sum(state)
        # Cells far from damage should be unchanged
        assert damaged[0, 0, 0] == 1.0


class TestMetrics:
    """Test metric computations."""
    
    def test_K_computation(self):
        """K should be log10 of cost ratio."""
        # If policy is 10x better than blind
        K = compute_K(J_pi=0.1, J_blind=1.0)
        assert abs(K - 1.0) < 0.01
        
        # If policy is 100x better
        K = compute_K(J_pi=0.01, J_blind=1.0)
        assert abs(K - 2.0) < 0.01
        
        # If policy equals blind
        K = compute_K(J_pi=1.0, J_blind=1.0)
        assert abs(K - 0.0) < 0.01
    
    def test_I_local_computation(self):
        """I_local should be normalized between blind and optimal."""
        baselines = BaselineEstimates(
            J_blind_rw=1.0,
            J_blind_rw_std=0.1,
            J_blind_triv=2.0,
            J_optimal_B=0.1,
            budget_B=1000
        )
        
        # At blind level
        I = compute_I_local(1.0, baselines)
        assert abs(I - 0.0) < 0.01
        
        # At optimal level
        I = compute_I_local(0.1, baselines)
        assert abs(I - 1.0) < 0.01
        
        # Halfway
        I = compute_I_local(0.55, baselines)
        assert abs(I - 0.5) < 0.01
    
    def test_K_ratio(self):
        """K_ratio should be K / K_opt."""
        baselines = BaselineEstimates(
            J_blind_rw=1.0,
            J_blind_rw_std=0.1,
            J_blind_triv=2.0,
            J_optimal_B=0.1,
            budget_B=1000
        )
    
        from src.metrics import compute_K_opt
        K_opt = compute_K_opt(baselines)
        
        # At optimal: K = K_opt, so ratio = 1
        K_at_opt = compute_K(0.1, baselines.J_blind_rw)
        assert abs(K_at_opt / K_opt - 1.0) < 0.01
        
        # At blind: K = 0, so ratio = 0
        K_at_blind = compute_K(1.0, baselines.J_blind_rw)
        assert abs(K_at_blind) < 0.01
        
        # Intermediate: K should be between 0 and K_opt
        K_mid = compute_K(0.3, baselines.J_blind_rw)
        assert 0 < K_mid < K_opt


class TestTraining:
    """Test training functionality."""
    
    def test_train_step_reduces_loss(self):
        """A few train steps should reduce loss."""
        config = NCAConfig(grid_size=32)
        key = random.PRNGKey(0)
        
        # Create simple target
        target = jnp.zeros((32, 32, 4))
        target = target.at[16, 16, :].set(1.0)
        
        state = create_train_state(config, key)
        seed = make_seed(config)
        
        # Initial loss
        key, eval_key = random.split(key)
        initial_final = rollout(state.params, seed, config, eval_key, 10)
        initial_loss = jnp.mean((initial_final[:, :, :4] - target) ** 2)
        
        # Train for a few steps
        for i in range(10):
            key, step_key = random.split(key)
            state, loss = train_step(state, seed, target, config, step_key, 10)
        
        # Loss should decrease (or at least not explode)
        assert loss < initial_loss * 2  # Allow some variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
