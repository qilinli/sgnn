#!/usr/bin/env python3
"""
Taylor Impact Bar Dataset Data Loader

This module provides PyTorch data loaders specifically designed for the Taylor Impact Bar dataset.
It handles the unique data structure (positions, particle_types, stresses) and provides both
sample-based and trajectory-based loading modes for training and evaluation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class BaseTaylorImpactDataset:
    """Base class with common functionality for Taylor Impact datasets."""
    
    def _load_stress_stats_from_metadata(self, data_path: str) -> Optional[Dict]:
        """
        Load stress statistics from metadata for denormalization.
        
        Args:
            data_path: Path to the NPZ file
            
        Returns:
            Dictionary with 'mean' and 'std' keys, or None if not found
        """
        data_path = Path(data_path)
        metadata_path = data_path.parent / "metadata.json"
        
        if not metadata_path.exists():
            print(f"Warning: No metadata.json found at {metadata_path}. Cannot load stress stats.")
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            stress_mean = metadata.get('stress_mean')
            stress_std = metadata.get('stress_std')
            
            if stress_mean is not None and stress_std is not None:
                return {'mean': stress_mean, 'std': stress_std}
            else:
                print("Warning: Stress statistics not found in metadata.")
                return None
                
        except Exception as e:
            print(f"Warning: Failed to load stress statistics: {e}")
            return None
    
    def denormalize_stress(self, normalized_stress: np.ndarray) -> np.ndarray:
        """
        Denormalize stress values back to original scale.
        
        Args:
            normalized_stress: Normalized stress values (mean=0, std=1)
            
        Returns:
            Denormalized stress values in original units
        """
        if not hasattr(self, '_stress_stats') or self._stress_stats is None:
            return normalized_stress
        
        return normalized_stress * self._stress_stats['std'] + self._stress_stats['mean']
    
    def _load_data(self, data_path: str) -> List[Tuple]:
        """
        Load data from NPZ file and convert to list of tuples.
        
        Args:
            data_path: Path to the NPZ file
            
        Returns:
            List of (positions, particle_types, stresses) tuples
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with np.load(data_path, allow_pickle=True) as data:
            trajectories = data['trajectories'].item()
        
        data_list = []
        for name, trajectory_data in trajectories.items():
            if isinstance(trajectory_data, tuple) and len(trajectory_data) == 3:
                data_list.append(trajectory_data)
            else:
                print(f"Warning: Skipping trajectory {name} with unexpected format")
        
        return data_list


class TaylorImpactSamplesDataset(torch.utils.data.Dataset, BaseTaylorImpactDataset):
    """
    Dataset for training that provides individual samples with input sequences.
    
    Each sample contains:
    - Input: positions sequence, particle types, particle count
    - Output: next position, next stress (normalized)
    - Meta: trajectory index, time index
    """
    
    def __init__(self, data_path: str, input_length_sequence: int = 6, load_stress_stats: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the NPZ file (e.g., 'train.npz')
            input_length_sequence: Number of timesteps to use as input (default: 6)
            load_stress_stats: Whether to load stress statistics for denormalization
        """
        super().__init__()
        
        self._data = self._load_data(data_path)
        self._input_length_sequence = input_length_sequence
        self._load_stress_stats = load_stress_stats
        
        if len(self._data) == 0:
            raise ValueError(f"No trajectories found in {data_path}")
        
        self._dimension = self._data[0][0].shape[-1]
        
        if self._load_stress_stats:
            self._stress_stats = self._load_stress_stats_from_metadata(data_path)
        
        self._data_lengths = [x.shape[0] - self._input_length_sequence for x, _, _ in self._data]
        self._length = sum(self._data_lengths)
        self._cumulative_lengths = np.cumsum([0] + self._data_lengths[:-1])
        
        print(f"Loaded {len(self._data)} trajectories with {self._length} total samples")
        print(f"Input sequence length: {input_length_sequence}, Dimension: {self._dimension}")
        if self._load_stress_stats and self._stress_stats:
            print(f"Stress normalization info loaded: mean={self._stress_stats['mean']:.6f}, std={self._stress_stats['std']:.6f}")
        else:
            print("Stress normalization info not loaded")
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input, output, and meta data
        """
        trajectory_idx = np.searchsorted(self._cumulative_lengths, idx, side='right') - 1
        start_idx = self._cumulative_lengths[trajectory_idx]
        time_idx = self._input_length_sequence + (idx - start_idx)
        
        positions, particle_types, stresses = self._data[trajectory_idx]
        
        input_positions = positions[time_idx - self._input_length_sequence:time_idx]
        input_positions = np.transpose(input_positions, (1, 0, 2))
        
        next_position = positions[time_idx]
        next_stress = stresses[time_idx]
        
        particle_type_array = np.full(input_positions.shape[0], particle_types[0], dtype=int)
        
        return {
            'input': {
                'positions': input_positions.astype(np.float32),
                'particle_type': particle_type_array,
                'n_particles_per_example': input_positions.shape[0]
            },
            'output': {
                'next_position': next_position.astype(np.float32),
                'next_strain': next_stress.astype(np.float32)
            },
            'meta': {
                'trajectory_idx': trajectory_idx,
                'time_idx': time_idx
            }
        }


class TaylorImpactTrajectoriesDataset(torch.utils.data.Dataset, BaseTaylorImpactDataset):
    """
    Dataset for evaluation that provides complete trajectories.
    
    Each item contains a complete simulation trajectory for rollout evaluation.
    """
    
    def __init__(self, data_path: str, load_stress_stats: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the NPZ file (e.g., 'valid.npz', 'test.npz')
            load_stress_stats: Whether to load stress statistics for denormalization
        """
        super().__init__()
        
        self._data = self._load_data(data_path)
        self._load_stress_stats = load_stress_stats
        self._dimension = self._data[0][0].shape[-1] if len(self._data) > 0 else 2
        self._length = len(self._data)
        
        if self._load_stress_stats:
            self._stress_stats = self._load_stress_stats_from_metadata(data_path)
        
        print(f"Loaded {self._length} trajectories for evaluation")
        print(f"Dimension: {self._dimension}")
        if self._load_stress_stats and self._stress_stats:
            print(f"Stress normalization info loaded: mean={self._stress_stats['mean']:.6f}, std={self._stress_stats['std']:.6f}")
        else:
            print("Stress normalization info not loaded")
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a complete trajectory.
        
        Args:
            idx: Trajectory index
            
        Returns:
            Dictionary with trajectory data
        """
        positions, particle_types, stresses = self._data[idx]
        
        positions = np.transpose(positions, (1, 0, 2))
        particle_type_array = np.full(positions.shape[0], particle_types[0], dtype=int)
        
        return {
            'positions': torch.tensor(positions.astype(np.float32)).contiguous(),
            'particle_type': torch.tensor(particle_type_array).contiguous(),
            'n_particles_per_example': torch.tensor(positions.shape[0]).contiguous(),
            'strains': torch.tensor(stresses.astype(np.float32)).contiguous(),
            'trajectory_idx': idx
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for batching samples.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary with tensors
    """
    position_list = []
    particle_type_list = []
    n_particles_list = []
    next_position_list = []
    next_strain_list = []
    trajectory_idx_list = []
    time_idx_list = []
    
    for sample in batch:
        position_list.append(sample['input']['positions'])
        particle_type_list.append(sample['input']['particle_type'])
        n_particles_list.append(sample['input']['n_particles_per_example'])
        next_position_list.append(sample['output']['next_position'])
        next_strain_list.append(sample['output']['next_strain'])
        trajectory_idx_list.append(sample['meta']['trajectory_idx'])
        time_idx_list.append(sample['meta']['time_idx'])
    
    return {
        'input': {
            'positions': torch.tensor(np.vstack(position_list), dtype=torch.float32).contiguous(),
            'particle_type': torch.tensor(np.concatenate(particle_type_list)).contiguous(),
            'n_particles_per_example': torch.tensor(n_particles_list).contiguous()
        },
        'output': {
            'next_position': torch.tensor(np.vstack(next_position_list), dtype=torch.float32).contiguous(),
            'next_strain': torch.tensor(np.concatenate(next_strain_list), dtype=torch.float32).contiguous()
        },
        'meta': {
            'trajectory_idx': torch.tensor(trajectory_idx_list).contiguous(),
            'time_idx': torch.tensor(time_idx_list).contiguous()
        }
    }


def get_data_loader_by_samples(
    path: str, 
    input_length_sequence: int = 6, 
    batch_size: int = 2, 
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    load_stress_stats: bool = True
) -> torch.utils.data.DataLoader:
    """
    Get a data loader for training with individual samples.
    
    Args:
        path: Path to the NPZ file (e.g., 'train.npz')
        input_length_sequence: Number of timesteps to use as input
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        load_stress_stats: Whether to load stress statistics for denormalization
        
    Returns:
        PyTorch DataLoader
    """
    dataset = TaylorImpactSamplesDataset(path, input_length_sequence, load_stress_stats)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def get_data_loader_by_trajectories(
    path: str,
    num_workers: int = 0,
    pin_memory: bool = True,
    load_stress_stats: bool = True
) -> torch.utils.data.DataLoader:
    """
    Get a data loader for evaluation with complete trajectories.
    
    Args:
        path: Path to the NPZ file (e.g., 'valid.npz', 'test.npz')
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        load_stress_stats: Whether to load stress statistics for denormalization
        
    Returns:
        PyTorch DataLoader
    """
    dataset = TaylorImpactTrajectoriesDataset(path, load_stress_stats)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def get_dataset_info(data_path: str) -> Dict:
    """
    Get information about the dataset without loading all data.
    
    Args:
        data_path: Path to the NPZ file
        
    Returns:
        Dictionary with dataset information
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with np.load(data_path, allow_pickle=True) as data:
        trajectories = data['trajectories'].item()
    
    num_trajectories = len(trajectories)
    if num_trajectories == 0:
        return {'num_trajectories': 0, 'error': 'No trajectories found'}
    
    first_traj = next(iter(trajectories.values()))
    if isinstance(first_traj, tuple) and len(first_traj) == 3:
        positions, particle_types, stresses = first_traj
        return {
            'num_trajectories': num_trajectories,
            'dimension': positions.shape[-1],
            'max_timesteps': positions.shape[0],
            'num_particles': positions.shape[1],
            'particle_types': list(np.unique(particle_types)),
            'stress_range': [float(stresses.min()), float(stresses.max())]
        }
    else:
        return {'num_trajectories': num_trajectories, 'error': 'Unexpected data format'}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Taylor Impact Data Loader')
    parser.add_argument('data_path', help='Path to NPZ file')
    parser.add_argument('--mode', choices=['samples', 'trajectories'], default='samples', 
                       help='Data loading mode')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for samples mode')
    
    args = parser.parse_args()
    
    print("Dataset Info:")
    print(json.dumps(get_dataset_info(args.data_path), indent=2))
    print()
    
    if args.mode == 'samples':
        print("Testing Samples Mode:")
        loader = get_data_loader_by_samples(args.data_path, batch_size=args.batch_size)
        for i, batch in enumerate(loader):
            print(f"Batch {i}:")
            print(f"  Positions: {batch['input']['positions'].shape}")
            print(f"  Particle types: {batch['input']['particle_type'].shape}")
            print(f"  Next positions: {batch['output']['next_position'].shape}")
            if i >= 2:
                break
    else:
        print("Testing Trajectories Mode:")
        loader = get_data_loader_by_trajectories(args.data_path)
        for i, trajectory in enumerate(loader):
            print(f"Trajectory {i}:")
            print(f"  Positions: {trajectory['positions'].shape}")
            print(f"  Strains: {trajectory['strains'].shape}")
            if i >= 2:
                break
