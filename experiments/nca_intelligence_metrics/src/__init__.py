"""
NCA Intelligence Metrics - Source Package

Core modules:
- nca: Neural Cellular Automata implementation
- metrics: K, I_local, and related quantities
- cones: Cognitive light cone estimation
- probes: Effective prediction horizon (H_eff) estimation
- training: Training loop with metric logging
"""

from .nca import NCAConfig, make_seed, rollout, apply_damage
from .metrics import (
    BaselineEstimates, 
    MetricSnapshot, 
    compute_K, 
    compute_I_local,
    compute_all_metrics
)
from .cones import ConeEstimationConfig, ConeEstimate, estimate_cone
from .probes import HorizonConfig, HorizonEstimate, estimate_H_eff
from .training import TrainingConfig, TrainingState, train_nca

__all__ = [
    # NCA
    'NCAConfig',
    'make_seed',
    'rollout',
    'apply_damage',
    
    # Metrics
    'BaselineEstimates',
    'MetricSnapshot',
    'compute_K',
    'compute_I_local',
    'compute_all_metrics',
    
    # Cones
    'ConeEstimationConfig',
    'ConeEstimate',
    'estimate_cone',
    
    # Probes
    'HorizonConfig',
    'HorizonEstimate',
    'estimate_H_eff',
    
    # Training
    'TrainingConfig',
    'TrainingState',
    'train_nca',
]
