#!/usr/bin/env python3
"""
Debug script for multi-scale GNN inference issues.
This script helps identify why autoregressive prediction fails and strain prediction is poor.
"""

import sys
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from gns.multi_scale.multi_scale_simulator import MultiScaleSimulator
from gns.multi_scale.data_loader_multi_scale import get_multi_scale_data_loader_by_trajectories
from gns.multi_scale import validate_multi_scale
from gns import reading_utils


def debug_single_trajectory(
    data_path: str,
    model_path: str,
    model_file: str,
    device: str = 'cuda',
    num_scales: int = 3,
    window_size: int = 3,
    radius_multiplier: float = 2.0,
    trajectory_idx: int = 0
):
    """Debug a single trajectory with detailed analysis."""
    
    print(f"🔍 DEBUGGING Multi-Scale GNN Inference")
    print(f"   - Data path: {data_path}")
    print(f"   - Model: {model_path}/{model_file}")
    print(f"   - Device: {device}")
    print(f"   - Trajectory: {trajectory_idx}")
    
    # Load metadata
    metadata = reading_utils.read_metadata(data_path)
    print(f"   - Sequence length: {metadata['sequence_length']}")
    print(f"   - Dimension: {metadata['dim']}")
    
    # Load model
    simulator = MultiScaleSimulator(
        kinematic_dimensions=metadata['dim'],
        num_scales=num_scales,
        window_size=window_size,
        radius_multiplier=radius_multiplier,
        normalization_stats={
            'acceleration': {
                'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
                'std': torch.FloatTensor(metadata['acc_std']).to(device),
            },
            'velocity': {
                'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
                'std': torch.FloatTensor(metadata['vel_std']).to(device),
            },
        },
        device=device
    )
    
    # Load model weights
    simulator.load(model_path + model_file)
    simulator.eval()
    simulator.to(device)
    
    # Load validation data
    data_trajs = get_multi_scale_data_loader_by_trajectories(
        path=f"{data_path}valid.npz",
        num_scales=num_scales,
        window_size=window_size,
        radius_multiplier=radius_multiplier
    )
    
    # Get specific trajectory
    data_traj = list(data_trajs)[trajectory_idx]
    simulator.set_static_graph(data_traj['graph'])
    
    # Extract data
    positions = data_traj['positions'].to(device)
    particle_type = data_traj['particle_type'].to(device)
    n_particles_per_example = data_traj['n_particles_per_example'].to(device)
    strains = data_traj['strains'].to(device)
    
    print(f"\n📊 Trajectory Analysis:")
    print(f"   - Positions shape: {positions.shape}")
    print(f"   - Strains shape: {strains.shape}")
    print(f"   - Particle types: {torch.unique(particle_type)}")
    print(f"   - Position range: [{positions.min():.3f}, {positions.max():.3f}]")
    print(f"   - Strain range: [{strains.min():.3f}, {strains.max():.3f}]")
    
    # Test both inference modes
    for mode in ['one_step', 'autoregressive']:
        print(f"\n🎯 Testing {mode} inference:")
        
        result = validate_multi_scale.evaluate_multi_scale_rollout(
            simulator=simulator,
            positions=positions,
            particle_type=particle_type,
            n_particles_per_example=n_particles_per_example,
            strains=strains,
            nsteps=metadata['sequence_length'] - 3,  # Use 3 input steps
            dim=metadata['dim'],
            device=device,
            input_sequence_length=3,
            inference_mode=mode,
            debug=True
        )
        
        print(f"   - Final position RMSE: {result['rmse_position'][-1]:.6f}")
        print(f"   - Final strain RMSE: {result['rmse_strain'][-1]:.6f}")
        print(f"   - Position RMSE trend: {result['rmse_position'][:5]}")
        print(f"   - Strain RMSE trend: {result['rmse_strain'][:5]}")
        
        # Analyze debug info if available
        if 'debug_info' in result:
            debug_info = result['debug_info']
            print(f"\n🔍 Detailed Analysis for {mode}:")
            
            # Check if errors are growing
            position_rmse = [step['position_rmse'] for step in debug_info['step_errors']]
            strain_rmse = [step['strain_rmse'] for step in debug_info['step_errors']]
            
            print(f"   - Position RMSE growth: {position_rmse[0]:.6f} -> {position_rmse[-1]:.6f}")
            print(f"   - Strain RMSE growth: {strain_rmse[0]:.6f} -> {strain_rmse[-1]:.6f}")
            
            # Check if predictions are reasonable
            if len(debug_info['position_predictions']) > 0:
                pred_pos = debug_info['position_predictions'][0]  # First step
                target_pos = debug_info['target_positions'][0]
                print(f"   - First step position error: {np.mean(np.linalg.norm(pred_pos - target_pos, axis=1)):.6f}")
                
            if len(debug_info['strain_predictions']) > 0:
                pred_strain = debug_info['strain_predictions'][0]  # First step
                target_strain = debug_info['target_strains'][0]
                print(f"   - First step strain error: {np.mean(np.abs(pred_strain - target_strain)):.6f}")


def analyze_training_data(data_path: str, num_scales: int = 3, window_size: int = 3, radius_multiplier: float = 2.0):
    """Analyze the training data to understand the problem better."""
    
    print(f"📊 ANALYZING Training Data")
    
    # Load training data
    data_trajs = get_multi_scale_data_loader_by_trajectories(
        path=f"{data_path}train.npz",
        num_scales=num_scales,
        window_size=window_size,
        radius_multiplier=radius_multiplier
    )
    
    all_position_errors = []
    all_strain_errors = []
    
    for i, data_traj in enumerate(data_trajs):
        if i >= 5:  # Analyze first 5 trajectories
            break
            
        positions = data_traj['positions']
        strains = data_traj['strains']
        
        # Calculate velocity and acceleration from positions
        velocities = positions[:, 1:] - positions[:, :-1]  # (nparticles, timesteps-1, dim)
        accelerations = velocities[:, 1:] - velocities[:, :-1]  # (nparticles, timesteps-2, dim)
        
        # Calculate strain rate
        strain_rates = strains[1:] - strains[:-1]  # (timesteps-1, nparticles)
        
        print(f"\nTrajectory {i}:")
        print(f"   - Position range: [{positions.min():.3f}, {positions.max():.3f}]")
        print(f"   - Velocity range: [{velocities.min():.3f}, {velocities.max():.3f}]")
        print(f"   - Acceleration range: [{accelerations.min():.3f}, {accelerations.max():.3f}]")
        print(f"   - Strain range: [{strains.min():.3f}, {strains.max():.3f}]")
        print(f"   - Strain rate range: [{strain_rates.min():.3f}, {strain_rates.max():.3f}]")
        
        # Check for any NaN or infinite values
        if torch.isnan(positions).any():
            print(f"   ⚠️  NaN values in positions!")
        if torch.isnan(strains).any():
            print(f"   ⚠️  NaN values in strains!")
        if torch.isinf(positions).any():
            print(f"   ⚠️  Infinite values in positions!")
        if torch.isinf(strains).any():
            print(f"   ⚠️  Infinite values in strains!")


def main():
    """Main debugging function."""
    
    # Configuration
    data_path = "datasets/taylor_impact_2d/data_processed/"
    model_path = "models/"
    model_file = "latest"  # or specific model file
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("🚀 Starting Multi-Scale GNN Debug Analysis")
    print("=" * 60)
    
    # Step 1: Analyze training data
    print("\n1️⃣ ANALYZING TRAINING DATA")
    analyze_training_data(data_path)
    
    # Step 2: Debug single trajectory
    print("\n2️⃣ DEBUGGING SINGLE TRAJECTORY")
    debug_single_trajectory(
        data_path=data_path,
        model_path=model_path,
        model_file=model_file,
        device=device,
        trajectory_idx=0
    )
    
    print("\n✅ Debug analysis complete!")
    print("\n🔍 Key things to check:")
    print("   1. Are there NaN/infinite values in the data?")
    print("   2. Are the position/strain ranges reasonable?")
    print("   3. Is the model learning (check training loss)?")
    print("   4. Are the static graphs being built correctly?")
    print("   5. Is the normalization working properly?")


if __name__ == "__main__":
    main()
