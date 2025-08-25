#!/usr/bin/env python3
"""
Data Loader Adapter for Taylor Impact Dataset

This module provides a drop-in replacement for the original gns.data_loader module,
specifically designed for the Taylor Impact Bar dataset. It maintains the same
interface while using the optimized, dataset-specific implementation.
"""

# Import our dataset-specific data loader
from .taylor_impact_data_loader import (
    get_data_loader_by_samples,
    get_data_loader_by_trajectories,
    get_dataset_info
)

# Re-export the functions with the same names for compatibility
__all__ = [
    'get_data_loader_by_samples',
    'get_data_loader_by_trajectories',
    'get_dataset_info'
]

# Optional: Add backward compatibility aliases if needed
# get_data_loader_by_samples = get_data_loader_by_samples
# get_data_loader_by_trajectories = get_data_loader_by_trajectories
