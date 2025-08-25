#!/usr/bin/env python3
"""
Taylor Impact Bar Dataset Builder

This script converts raw NPZ simulation data into train/val/test splits
for GNS training, with enhanced features for second-order dynamics.

Replaces the functionality of read_LSDYNA_2d_T.ipynb with clean, maintainable code.

Note: The NPZ files contain a field named 'strains' which actually contains
von Mises stress data. This naming inconsistency is preserved for compatibility.
"""

import json
import random
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class TaylorImpactDatasetBuilder:
    """Builds Taylor Impact Bar dataset for GNS training"""
    
    def __init__(self, config_path: str):
        """
        Initialize the dataset builder.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set paths from config with intelligent resolution
        self.input_dir = self._resolve_input_path(self.config['raw_data_path'], config_path)
        self.output_dir = Path(config_path).parent / self.config['output_data_path']
        
        # Set processing parameters from config
        self.step_size = self.config['step_size']
        self.total_steps = self.config['total_steps']
        self.random_seed = self.config['random_seed']
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation and test set identifiers
        self.val_set = self.config['val_set']
        self.test_set = self.config['test_set']
        
        # Additional configurable parameters
        self.boundary_particles_to_remove = self.config['boundary_particles_to_remove']
        self.stress_threshold = self.config['stress_threshold']
        
        # Statistics collection for on-the-fly computation
        self.velocities = np.array([]).reshape(0, 2)
        self.accelerations = np.array([]).reshape(0, 2)
        self.all_stresses = np.array([]).reshape(0)
        self.all_particle_types = np.array([]).reshape(0)  # Track all particle types
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from: {config_path}")
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")
    
    def _resolve_input_path(self, raw_data_path: str, config_path: str) -> Path:
        """
        Intelligently resolve the input data path to work across different computers.
        
        Args:
            raw_data_path: Raw data path from config (can be relative or absolute)
            config_path: Path to the config file for relative path resolution
            
        Returns:
            Resolved Path object pointing to the input directory
        """
        config_dir = Path(config_path).parent
        input_path = Path(raw_data_path)
        
        # If it's already an absolute path, use it as-is
        if input_path.is_absolute():
            return input_path
        
        # Priority: config file → script → CWD → OneDrive locations
        
        # Try relative to config file first
        resolved_path = config_dir / input_path
        if resolved_path.exists():
            return resolved_path.resolve()
        
        # Try relative to script location (more robust)
        script_dir = Path(__file__).parent
        resolved_path = script_dir / input_path
        if resolved_path.exists():
            print(f"Found data relative to script at: {resolved_path}")
            return resolved_path.resolve()
        
        # Try relative to current working directory
        resolved_path = Path.cwd() / input_path
        if resolved_path.exists():
            print(f"Found data relative to current directory at: {resolved_path}")
            return resolved_path.resolve()
        
        # Try common OneDrive locations
        common_paths = [
            Path.home() / "OneDrive - Curtin" / "research" / "civil_engineering" / "data" / "2D-Copper-Bar-Taylor-Impact" / "npz",
            Path.home() / "OneDrive" / "research" / "civil_engineering" / "data" / "2D-Copper-Bar-Taylor-Impact" / "npz",
        ]
        
        for path in common_paths:
            if path.exists():
                print(f"Found data at common location: {path}")
                return path.resolve()
        
        return config_dir / input_path
        
    def build_dataset(self):
        """Build the complete dataset with train/val/test splits"""
        print("Building Taylor Impact Bar Dataset")
        print("=" * 50)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Step size: {self.step_size}")
        print(f"Total steps: {self.total_steps}")
        print(f"Random seed: {self.random_seed}")
        print()
        
        # Get all NPZ files
        npz_files = list(self.input_dir.glob("*.npz"))
        if not npz_files:
            raise ValueError(f"No NPZ files found in {self.input_dir}. Check dataset_config.yaml 'raw_data_path' setting.")
        
        print(f"Found {len(npz_files)} NPZ files")
        
        # Shuffle files for random split
        random.shuffle(npz_files)
        
        # Split files into train/val/test
        train_files, val_files, test_files = self._split_files(npz_files)
        
        print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        print()
        
        # Process each split
        train_data, train_stats = self._process_split(train_files, "train")
        val_data, val_stats = self._process_split(val_files, "val")
        test_data, test_stats = self._process_split(test_files, "test")
        
        # Save datasets and metadata together
        self._save_datasets_and_metadata(train_data, val_data, test_data, train_stats, val_stats, test_stats)
        
        print(f"\nDataset built successfully!")
        print(f"Output directory: {self.output_dir}")
        
    def _split_files(self, files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        """Split files into train/val/test based on filename patterns"""
        train_files = []
        val_files = []
        test_files = []
        
        for file_path in files:
            filename = file_path.name
            
            # Check if file belongs to validation set
            if any(name in filename for name in self.val_set):
                val_files.append(file_path)
            # Check if file belongs to test set
            elif any(name in filename for name in self.test_set):
                test_files.append(file_path)
            # Otherwise, add to training set
            else:
                train_files.append(file_path)
        
        return train_files, val_files, test_files
    
    def _process_split(
        self, 
        files: List[Path], 
        split_name: str
    ) -> Tuple[Dict, Dict]:
        """
        Process a single split (train/val/test).
        
        Args:
            files: List of NPZ file paths to process
            split_name: Name of the split for logging
            
        Returns:
            Tuple of (processed_data, split_statistics)
        """
        print(f"Processing {split_name} split...")
        
        split_data = {}
        split_stats = {
            "num_simulations": len(files),
            "filenames": [f.name for f in files],
            "total_timesteps": 0
        }
        
        for i, file_path in enumerate(files):
            print(f"  [{i+1}/{len(files)}] Processing {file_path.name}...")
            
            with np.load(file_path) as data:
                positions = data['positions']      # [T, N, 2]
                particle_types = data['particle_types']  # [N]
                stresses = data['strains']         # [T, N] - von Mises stress
            
            # Find first timestep with stress above threshold
            mean_stress = stresses.mean(axis=1)
            first_nonzero_step_idx = next(
                (i for i, x in enumerate(mean_stress) if x > self.stress_threshold), None
            )
            
            if first_nonzero_step_idx is None:
                print(f"    Warning: No von Mises stress above threshold {self.stress_threshold} found in {file_path.name}")
                continue
            
            # Extract timesteps and remove boundary particles
            positions = positions[
                first_nonzero_step_idx-1:first_nonzero_step_idx-1+self.total_steps:self.step_size,
                :-self.boundary_particles_to_remove,
                :
            ]
            
            # Remove boundary particles and remap to single type
            particle_types = particle_types[:-self.boundary_particles_to_remove]
            particle_types = np.zeros_like(particle_types)
            
            # Collect statistics and store data
            self.all_particle_types = np.concatenate((self.all_particle_types, particle_types), axis=0)
            
            stresses = stresses[
                first_nonzero_step_idx-1:first_nonzero_step_idx-1+self.total_steps:self.step_size,
                :-self.boundary_particles_to_remove
            ]
            
            self.all_stresses = np.concatenate((self.all_stresses, stresses.reshape(-1)), axis=0)
            
            vel_trajectory = positions[1:, :, :] - positions[:-1, :, :]
            acc_trajectory = vel_trajectory[1:, :, :] - vel_trajectory[:-1, :, :]
            
            self.velocities = np.concatenate((self.velocities, vel_trajectory.reshape(-1, 2)), axis=0)
            self.accelerations = np.concatenate((self.accelerations, acc_trajectory.reshape(-1, 2)), axis=0)
            
            split_data[file_path.stem] = (positions, particle_types, stresses)
            split_stats["total_timesteps"] += positions.shape[0]
            
            pos_min, pos_max = positions.min(axis=(0,1)), positions.max(axis=(0,1))
            stress_min, stress_max = stresses.min(axis=(0,1)), stresses.max(axis=(0,1))
            print(f"    Shape: {positions.shape}, Stress: [{stress_min:.3f}, {stress_max:.3f}], Pos: [{pos_min[0]:.3f}, {pos_max[0]:.3f}] x [{pos_min[1]:.3f}, {pos_max[1]:.3f}]")
        
        print(f"  {split_name.capitalize()} split complete: {len(split_data)} trajectories")
        print()
        
        return split_data, split_stats
    
    def _normalize_stresses(self, data: Dict, stress_mean: float, stress_std: float) -> Dict:
        """
        Normalize stresses using Z-score standardization.
        
        Args:
            data: Dictionary containing trajectory data
            stress_mean: Global mean stress value
            stress_std: Global standard deviation of stress values
            
        Returns:
            Dictionary with normalized stresses
        """
        if stress_mean is None or stress_std is None:
            print("  Warning: Cannot normalize stresses - missing statistics")
            return data
        
        normalized_data = {}
        for trajectory_name, (positions, particle_types, stresses) in data.items():
            # Apply Z-score normalization: (x - mean) / std
            normalized_stresses = (stresses - stress_mean) / stress_std
            normalized_data[trajectory_name] = (positions, particle_types, normalized_stresses)
        
        print(f"  Normalized stresses: mean=0, std=1 (original: mean={stress_mean:.6f}, std={stress_std:.6f})")
        return normalized_data
    

    
    def _save_datasets_and_metadata(self, train_data: Dict, val_data: Dict, test_data: Dict, 
                                   train_stats: Dict, val_stats: Dict, test_stats: Dict):
        """
        Save the processed datasets and metadata together for consistency.
        
        Args:
            train_data: Training dataset trajectories
            val_data: Validation dataset trajectories  
            test_data: Test dataset trajectories
            train_stats: Training split statistics
            val_stats: Validation split statistics
            test_stats: Test split statistics
        """
        print("Saving datasets and metadata...")
        
        # Save train dataset
        train_filepath = self.output_dir / "train.npz"
        np.savez(train_filepath, trajectories=train_data)
        print(f"  Train: {len(train_data)} trajectories → {train_filepath}")
        
        # Save validation dataset
        val_filepath = self.output_dir / "valid.npz"
        np.savez(val_filepath, trajectories=val_data)
        print(f"  Validation: {len(val_data)} trajectories → {val_filepath}")
        
        # Save test dataset
        test_filepath = self.output_dir / "test.npz"
        np.savez(test_filepath, trajectories=test_data)
        print(f"  Test: {len(test_data)} trajectories → {test_filepath}")
        
        print("Computing global statistics...")
        
        vel_mean, vel_std = list(self.velocities.mean(axis=0)), list(self.velocities.std(axis=0))
        acc_mean, acc_std = list(self.accelerations.mean(axis=0)), list(self.accelerations.std(axis=0))
        
        if len(self.all_stresses) > 0:
            stress_mean, stress_std = self.all_stresses.mean(), self.all_stresses.std()
            print(f"  Von Mises stress: mean={stress_mean:.6f}, std={stress_std:.6f}")
        else:
            stress_mean = stress_std = None
            print("  Warning: No stress data available for statistics")
        
        print("Normalizing stresses using Z-score standardization...")
        train_data = self._normalize_stresses(train_data, stress_mean, stress_std)
        val_data = self._normalize_stresses(val_data, stress_mean, stress_std)
        test_data = self._normalize_stresses(test_data, stress_mean, stress_std)
        
        np.savez(train_filepath, trajectories=train_data)
        np.savez(val_filepath, trajectories=val_data)
        np.savez(test_filepath, trajectories=test_data)
        print("  Datasets re-saved with normalized stresses")
        
        if len(self.all_particle_types) > 0:
            actual_particle_types = len(np.unique(self.all_particle_types))
            print(f"  Particle types found: [0.] (count: {actual_particle_types})")
        else:
            actual_particle_types = 1
            print("  Warning: No particle type data available, using default: 1")
        
        metadata = {
            'dataset_name': 'Taylor-Impact-2D',
            'dim': 2,
            'sequence_length': self.total_steps // self.step_size,
            'dt': 0.002 * self.step_size,
            'default_connectivity_radius': self.config.get('default_connectivity_radius', 0.6),
            'num_particle_types': actual_particle_types,
            'vel_mean': vel_mean, 'vel_std': vel_std,
            'acc_mean': acc_mean, 'acc_std': acc_std,
            'stress_mean': stress_mean, 'stress_std': stress_std,
            'file_train': train_stats['filenames'],
            'file_valid': val_stats['filenames'],
            'file_test': test_stats['filenames'],
            'step_size': self.step_size,
            'total_steps': self.total_steps,
            'random_seed': self.random_seed,
            'total_simulations': train_stats['num_simulations'] + val_stats['num_simulations'] + test_stats['num_simulations'],
            'train_simulations': train_stats['num_simulations'],
            'val_simulations': val_stats['num_simulations'],
            'test_simulations': test_stats['num_simulations'],
            'total_timesteps': train_stats['total_timesteps'] + val_stats['total_timesteps'] + test_stats['total_timesteps'],
            'train_timesteps': train_stats['total_timesteps'],
            'val_timesteps': val_stats['total_timesteps'],
            'test_timesteps': test_stats['total_timesteps'],
            'sph_config': self.config.get('sph_config', {})
        }
        
        metadata_filepath = self.output_dir / "metadata.json"
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Metadata saved → {metadata_filepath}")
        print(f"  Velocity stats: mean={vel_mean}, std={vel_std}")
        print(f"  Acceleration stats: mean={acc_mean}, std={acc_std}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Build Taylor Impact Bar dataset for GNS training')
    parser.add_argument('--config', type=str, default='dataset_config.yaml',
                       help='Path to configuration file (default: dataset_config.yaml)')
    
    args = parser.parse_args()
    
    # Resolve config path robustly
    config_path = args.config
    if not Path(config_path).is_absolute():
        script_dir = Path(__file__).parent
        script_config = script_dir / config_path
        if script_config.exists():
            config_path = str(script_config)
            print(f"Using config file: {config_path}")
        else:
            cwd_config = Path.cwd() / config_path
            if cwd_config.exists():
                config_path = str(cwd_config)
                print(f"Using config file: {config_path}")
            else:
                print(f"Warning: Config file not found at {script_config} or {cwd_config}")
    
    builder = TaylorImpactDatasetBuilder(config_path=config_path)
    builder.build_dataset()


if __name__ == "__main__":
    main()
