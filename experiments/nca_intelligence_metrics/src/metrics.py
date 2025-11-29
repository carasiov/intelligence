"""
Intelligence metrics: K, I_local, and related quantities.

Implements the core metrics from EXPERIMENT_SPEC.md:
- K (search compression)
- I_local (normalized local intelligence)
- K_opt (optimal search compression)
"""

import jax.numpy as jnp
from typing import NamedTuple, Optional
from dataclasses import dataclass


@dataclass
class BaselineEstimates:
    """Cached baseline cost estimates."""
    J_blind_rw: float       # Random-weight baseline cost
    J_blind_rw_std: float   # Standard error
    J_blind_triv: float     # Trivial baseline cost (seed only)
    J_optimal_B: float      # Best cost found under budget B
    budget_B: int           # Budget used to find J_optimal_B
    
    def is_valid(self) -> bool:
        """Check if estimates are usable."""
        return (
            self.J_blind_rw > 0 and 
            self.J_optimal_B > 0 and
            self.J_blind_rw > self.J_optimal_B  # Blind should be worse than optimal
        )


def compute_K(J_pi: float, J_blind: float) -> float:
    """
    Compute search compression K.
    
    K = log10(J_blind / J_pi)
    
    Interpretation:
    - K = 0: policy equals blind baseline
    - K = 1: policy is 10x better than blind
    - K = 2: policy is 100x better than blind
    - K < 0: policy is worse than blind
    
    Args:
        J_pi: cost (MSE) of the policy being evaluated
        J_blind: cost of the blind baseline
        
    Returns:
        K: search compression in orders of magnitude
    """
    if J_pi <= 0:
        # Perfect performance (or numerical issue)
        return float('inf')
    
    return float(jnp.log10(J_blind / J_pi))


def compute_K_vs_random(J_pi: float, baselines: BaselineEstimates) -> float:
    """Compute K relative to random-weight baseline."""
    return compute_K(J_pi, baselines.J_blind_rw)


def compute_K_vs_trivial(J_pi: float, baselines: BaselineEstimates) -> float:
    """Compute K relative to trivial (seed-only) baseline."""
    return compute_K(J_pi, baselines.J_blind_triv)


def compute_K_opt(baselines: BaselineEstimates) -> float:
    """
    Compute optimal search compression K_opt.
    
    K_opt = log10(J_blind / J_optimal)
    
    This measures how much "room for intelligence" exists in the task.
    """
    return compute_K(baselines.J_optimal_B, baselines.J_blind_rw)


def compute_I_local(J_pi: float, baselines: BaselineEstimates) -> float:
    """
    Compute normalized local intelligence I_local.
    
    I_local = (J_blind - J_pi) / (J_blind - J_optimal)
    
    Interpretation:
    - I_local = 0: policy equals blind baseline
    - I_local = 1: policy equals best-known optimal
    - I_local > 1: policy beats current optimal (update optimal!)
    - I_local < 0: policy is worse than blind
    
    Args:
        J_pi: cost of the policy being evaluated
        baselines: baseline estimates including J_blind and J_optimal
        
    Returns:
        I_local: normalized performance
    """
    J_blind = baselines.J_blind_rw
    J_opt = baselines.J_optimal_B
    
    denominator = J_blind - J_opt
    
    if denominator <= 0:
        # Optimal is not better than blind (shouldn't happen)
        return float('nan')
    
    return float((J_blind - J_pi) / denominator)


def compute_K_ratio(J_pi: float, baselines: BaselineEstimates) -> float:
    """
    Compute K as fraction of K_opt achieved.
    
    K_ratio = K / K_opt = log(J_blind/J_pi) / log(J_blind/J_opt)
    
    This measures what fraction of the "available" search compression
    the policy achieves. Ranges from 0 (at blind) to 1 (at optimal).
    
    Note: K and I_local are complementary metrics, not related by a
    simple identity. K measures log-scale compression; I_local measures
    linear-scale normalized performance.
    """
    K = compute_K(J_pi, baselines.J_blind_rw)
    K_opt = compute_K_opt(baselines)
    
    if K_opt <= 0:
        return 0.0
    
    return float(K / K_opt)


@dataclass
class MetricSnapshot:
    """All metrics at a point in training."""
    step: int
    loss: float                 # Raw MSE loss
    K_vs_rw: float             # K relative to random-weight baseline
    K_vs_triv: float           # K relative to trivial baseline
    I_local: float             # Normalized local intelligence
    K_opt: float               # Optimal search compression (from baselines)
    K_ratio: float             # K / K_opt (fraction of available compression)
    
    def to_dict(self) -> dict:
        return {
            'step': self.step,
            'loss': self.loss,
            'K_vs_rw': self.K_vs_rw,
            'K_vs_triv': self.K_vs_triv,
            'I_local': self.I_local,
            'K_opt': self.K_opt,
            'K_ratio': self.K_ratio
        }


def compute_all_metrics(
    J_pi: float,
    baselines: BaselineEstimates,
    step: int = 0
) -> MetricSnapshot:
    """
    Compute all metrics for a given policy cost.
    
    Args:
        J_pi: cost (MSE) of the policy
        baselines: baseline estimates
        step: training step (for logging)
        
    Returns:
        MetricSnapshot with all computed metrics
    """
    K_vs_rw = compute_K_vs_random(J_pi, baselines)
    K_vs_triv = compute_K_vs_trivial(J_pi, baselines)
    I_local = compute_I_local(J_pi, baselines)
    K_opt = compute_K_opt(baselines)
    K_ratio = compute_K_ratio(J_pi, baselines)
    
    return MetricSnapshot(
        step=step,
        loss=J_pi,
        K_vs_rw=K_vs_rw,
        K_vs_triv=K_vs_triv,
        I_local=I_local,
        K_opt=K_opt,
        K_ratio=K_ratio
    )


# =============================================================================
# Budget tracking for K_opt(B) curves
# =============================================================================

@dataclass
class BudgetTracker:
    """Track best model and K_opt as function of search budget."""
    
    budget_levels: list       # List of budget thresholds
    best_cost_at_budget: dict # budget -> best cost seen
    K_opt_at_budget: dict     # budget -> K_opt at that budget
    J_blind: float            # Baseline cost (fixed)
    
    def __init__(self, budget_levels: list, J_blind: float):
        self.budget_levels = sorted(budget_levels)
        self.best_cost_at_budget = {b: float('inf') for b in budget_levels}
        self.K_opt_at_budget = {b: 0.0 for b in budget_levels}
        self.J_blind = J_blind
    
    def update(self, current_budget: int, cost: float):
        """Update best costs for all budget levels <= current_budget."""
        for b in self.budget_levels:
            if current_budget >= b:
                if cost < self.best_cost_at_budget[b]:
                    self.best_cost_at_budget[b] = cost
                    self.K_opt_at_budget[b] = compute_K(cost, self.J_blind)
    
    def get_K_opt_curve(self) -> dict:
        """Return K_opt as function of budget."""
        return dict(self.K_opt_at_budget)
    
    def has_plateaued(self, threshold: float = 0.1) -> bool:
        """
        Check if K_opt has plateaued (consecutive levels differ by < threshold).
        
        Args:
            threshold: maximum difference to consider "plateaued"
            
        Returns:
            True if last two budget levels show plateau
        """
        if len(self.budget_levels) < 2:
            return False
        
        K_values = [self.K_opt_at_budget[b] for b in self.budget_levels]
        
        # Check last two levels
        diff = abs(K_values[-1] - K_values[-2])
        return diff < threshold
