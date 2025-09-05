# Multi-scale GNN module for particle-based simulations
# This module provides multi-scale graph neural network capabilities
# with hierarchical grid structures where each level is a subset 
# of the previous level (grid → mesh levels)

from .multi_scale_graph import MultiScaleGraph, MultiScaleConfig
from .multi_scale_gnn import MultiScaleGNN

__all__ = [
    'MultiScaleGraph',
    'MultiScaleConfig', 
    'ProgressiveSampler', 
    'MultiScaleGNN'
]

__version__ = '0.3.0'
