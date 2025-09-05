#!/usr/bin/env python3
"""
Multi-Scale Data Loader for Taylor Impact Dataset

This module extends the Taylor Impact data loader to support multi-scale graph training.
It builds static multi-scale graphs during data loading and provides them to the simulator.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datasets.taylor_impact_2d.taylor_impact_data_loader import (
    BaseTaylorImpactDataset,
    TaylorImpactSamplesDataset,
    TaylorImpactTrajectoriesDataset
)


def build_static_multi_scale_graph(initial_positions: torch.Tensor,
                                   num_scales: int = 3,
                                   window_size: int = 3,
                                   radius_multiplier: float = 2.0) -> Dict[str, Any]:
    """Build static multi-scale graph from initial particle positions.
    
    This function should be called during data processing to create the static
    graph structure that will be reused throughout the simulation.
    
    Args:
        initial_positions: Initial particle positions (nparticles, dim)
        num_scales: Number of scales in the hierarchy (typically 3: grid + 2 mesh levels)
        window_size: Sampling window size for mesh levels
        radius_multiplier: Multiplier for all connectivity radius calculations
        
    Returns:
        Dictionary containing:
        - 'graph_hierarchy': Dict with scale data (sampling_indices, spacing, num_particles)
        - 'grid2mesh_edges': Edge indices for grid to mesh connections
        - 'mesh2mesh_edges': Edge indices for mesh to mesh connections  
        - 'mesh2grid_edges': Edge indices for mesh to grid connections
    """
    from gns.multi_scale.multi_scale_graph import MultiScaleGraph, MultiScaleConfig
    
    # Create configuration
    config = MultiScaleConfig(num_scales=num_scales, window_size=window_size, radius_multiplier=radius_multiplier)
    
    # Create multi-scale graph
    multi_scale_graph = MultiScaleGraph(config)
    
    # Build the complete graph structure
    graph_data = multi_scale_graph.create_all_edges(initial_positions)
    
    return graph_data


class MultiScaleTaylorImpactSamplesDataset(TaylorImpactSamplesDataset):
    """
    Multi-scale version of TaylorImpactSamplesDataset that builds static graphs.
    
    Each sample contains:
    - Input: positions sequence, particle types, particle count
    - Output: next position, next stress (normalized)
    - Meta: trajectory index, time index
    - Graph: static multi-scale graph data
    """
    
    def __init__(self, 
                 data_path: str, 
                 input_length_sequence: int = 3, 
                 load_stress_stats: bool = True,
                 num_scales: int = 3,
                 window_size: int = 3,
                 radius_multiplier: float = 2.0):
        """
        Initialize the multi-scale dataset.
        
        Args:
            data_path: Path to the NPZ file (e.g., 'train.npz')
            input_length_sequence: Number of timesteps to use as input (default: 3)
            load_stress_stats: Whether to load stress statistics for denormalization
            num_scales: Number of scales in the multi-scale hierarchy
            window_size: Sampling window size for mesh levels
            radius_multiplier: Multiplier for all connectivity radius calculations
        """
        super().__init__(data_path, input_length_sequence, load_stress_stats)
        
        self._num_scales = num_scales
        self._window_size = window_size
        
        # Build static graphs for each trajectory
        print(f"Building static multi-scale graphs for {len(self._data)} trajectories...")
        self._static_graphs = {}
        
        for i, (positions, particle_types, stresses) in enumerate(self._data):
            # Use initial positions to build static graph
            initial_positions = torch.tensor(positions[0], dtype=torch.float32)
            
            # Build static multi-scale graph
            graph_data = build_static_multi_scale_graph(
                initial_positions, 
                num_scales=num_scales, 
                window_size=window_size,
                radius_multiplier=radius_multiplier
            )
            
            self._static_graphs[i] = graph_data
            
            if i % 5 == 0:  # Progress indicator
                print(f"  Built graph {i+1}/{len(self._data)}")
        
        print(f"✅ Built {len(self._static_graphs)} static multi-scale graphs")
        print(f"   - Scales: {num_scales}, Window size: {window_size}")
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single training sample with static graph data.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input, output, meta, and graph data
        """
        # Get the base sample from parent class
        sample = super().__getitem__(idx)
        
        # Add static graph data
        trajectory_idx = sample['meta']['trajectory_idx']
        sample['graph'] = self._static_graphs[trajectory_idx]
        
        return sample


class MultiScaleTaylorImpactTrajectoriesDataset(TaylorImpactTrajectoriesDataset):
    """
    Multi-scale version of TaylorImpactTrajectoriesDataset that builds static graphs.
    
    Each item contains a complete simulation trajectory with static graph data.
    """
    
    def __init__(self, 
                 data_path: str, 
                 load_stress_stats: bool = True,
                 num_scales: int = 3,
                 window_size: int = 3,
                 radius_multiplier: float = 2.0):
        """
        Initialize the multi-scale trajectories dataset.
        
        Args:
            data_path: Path to the NPZ file (e.g., 'valid.npz', 'test.npz')
            load_stress_stats: Whether to load stress statistics for denormalization
            num_scales: Number of scales in the multi-scale hierarchy
            window_size: Sampling window size for mesh levels
            radius_multiplier: Multiplier for all connectivity radius calculations
        """
        super().__init__(data_path, load_stress_stats)
        
        self._num_scales = num_scales
        self._window_size = window_size
        
        # Build static graphs for each trajectory
        print(f"Building static multi-scale graphs for {len(self._data)} trajectories...")
        self._static_graphs = {}
        
        for i, (positions, particle_types, stresses) in enumerate(self._data):
            # Use initial positions to build static graph
            initial_positions = torch.tensor(positions[0], dtype=torch.float32)
            
            # Build static multi-scale graph
            graph_data = build_static_multi_scale_graph(
                initial_positions, 
                num_scales=num_scales, 
                window_size=window_size,
                radius_multiplier=radius_multiplier
            )
            
            self._static_graphs[i] = graph_data
            
            if i % 5 == 0:  # Progress indicator
                print(f"  Built graph {i+1}/{len(self._data)}")
        
        print(f"✅ Built {len(self._static_graphs)} static multi-scale graphs")
        print(f"   - Scales: {num_scales}, Window size: {window_size}")
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a complete trajectory with static graph data.
        
        Args:
            idx: Trajectory index
            
        Returns:
            Dictionary with trajectory and graph data
        """
        # Get the base trajectory from parent class
        trajectory = super().__getitem__(idx)
        
        # Add static graph data
        trajectory['graph'] = self._static_graphs[idx]
        
        return trajectory


def multi_scale_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for batching multi-scale samples.
    
    Args:
        batch: List of sample dictionaries with graph data
        
    Returns:
        Batched dictionary with tensors and graph data
    """
    # Use the original collate function for basic data
    from datasets.taylor_impact_2d.taylor_impact_data_loader import collate_fn
    batched_data = collate_fn(batch)
    
    # Add graph data (all samples in batch should have the same graph structure)
    # We'll use the graph from the first sample
    if 'graph' in batch[0]:
        batched_data['graph'] = batch[0]['graph']
    
    return batched_data


def get_multi_scale_data_loader_by_samples(
    path: str, 
    input_length_sequence: int = 3, 
    batch_size: int = 2, 
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    load_stress_stats: bool = True,
    num_scales: int = 3,
    window_size: int = 3,
    radius_multiplier: float = 2.0
) -> torch.utils.data.DataLoader:
    """
    Get a multi-scale data loader for training with individual samples.
    
    Args:
        path: Path to the NPZ file (e.g., 'train.npz')
        input_length_sequence: Number of timesteps to use as input
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        load_stress_stats: Whether to load stress statistics for denormalization
        num_scales: Number of scales in the multi-scale hierarchy
        window_size: Sampling window size for mesh levels
        radius_multiplier: Multiplier for all connectivity radius calculations
        
    Returns:
        PyTorch DataLoader with static graph data
    """
    dataset = MultiScaleTaylorImpactSamplesDataset(
        path, input_length_sequence, load_stress_stats, num_scales, window_size, radius_multiplier
    )
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=multi_scale_collate_fn
    )


def get_multi_scale_data_loader_by_trajectories(
    path: str,
    num_workers: int = 0,
    pin_memory: bool = True,
    load_stress_stats: bool = True,
    num_scales: int = 3,
    window_size: int = 3,
    radius_multiplier: float = 2.0
) -> torch.utils.data.DataLoader:
    """
    Get a multi-scale data loader for evaluation with complete trajectories.
    
    Args:
        path: Path to the NPZ file (e.g., 'valid.npz', 'test.npz')
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        load_stress_stats: Whether to load stress statistics for denormalization
        num_scales: Number of scales in the multi-scale hierarchy
        window_size: Sampling window size for mesh levels
        radius_multiplier: Multiplier for all connectivity radius calculations
        
    Returns:
        PyTorch DataLoader with static graph data
    """
    dataset = MultiScaleTaylorImpactTrajectoriesDataset(
        path, load_stress_stats, num_scales, window_size, radius_multiplier
    )
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Multi-Scale Taylor Impact Data Loader')
    parser.add_argument('data_path', help='Path to NPZ file')
    parser.add_argument('--mode', choices=['samples', 'trajectories'], default='samples', 
                       help='Data loading mode')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for samples mode')
    parser.add_argument('--num_scales', type=int, default=3, help='Number of scales')
    parser.add_argument('--window_size', type=int, default=3, help='Window size')
    
    args = parser.parse_args()
    
    print("Testing Multi-Scale Data Loader:")
    print(f"  Data path: {args.data_path}")
    print(f"  Mode: {args.mode}")
    print(f"  Scales: {args.num_scales}, Window: {args.window_size}")
    print()
    
    if args.mode == 'samples':
        print("Testing Samples Mode:")
        loader = get_multi_scale_data_loader_by_samples(
            args.data_path, 
            batch_size=args.batch_size,
            num_scales=args.num_scales,
            window_size=args.window_size
        )
        for i, batch in enumerate(loader):
            print(f"Batch {i}:")
            print(f"  Positions: {batch['input']['positions'].shape}")
            print(f"  Particle types: {batch['input']['particle_type'].shape}")
            print(f"  Next positions: {batch['output']['next_position'].shape}")
            print(f"  Graph hierarchy keys: {list(batch['graph']['graph_hierarchy'].keys())}")
            print(f"  G2M edges: {batch['graph']['grid2mesh_edges'].shape}")
            print(f"  M2M edges: {batch['graph']['mesh2mesh_edges'].shape}")
            print(f"  M2G edges: {batch['graph']['mesh2grid_edges'].shape}")
            if i >= 2:
                break
    else:
        print("Testing Trajectories Mode:")
        loader = get_multi_scale_data_loader_by_trajectories(
            args.data_path,
            num_scales=args.num_scales,
            window_size=args.window_size
        )
        for i, trajectory in enumerate(loader):
            print(f"Trajectory {i}:")
            print(f"  Positions: {trajectory['positions'].shape}")
            print(f"  Strains: {trajectory['strains'].shape}")
            print(f"  Graph hierarchy keys: {list(trajectory['graph']['graph_hierarchy'].keys())}")
            if i >= 2:
                break
