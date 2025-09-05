#!/usr/bin/env python3
"""
Basic test script to check if the multi-scale model is working correctly.
"""

import sys
import os
import json
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from gns.multi_scale.multi_scale_simulator import MultiScaleSimulator
from gns.multi_scale.data_loader_multi_scale import get_multi_scale_data_loader_by_trajectories
from gns import reading_utils


def test_model_forward_pass(data_path: str, model_path: str, model_file: str):
    """Test if the model can do a forward pass without errors."""
    
    print("🧪 Testing Multi-Scale Model Forward Pass")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   - Device: {device}")
    
    # Load metadata
    metadata = reading_utils.read_metadata(data_path)
    
    # Create simulator with proper parameters
    simulator = MultiScaleSimulator(
        kinematic_dimensions=metadata['dim'],
        nnode_in=5,  # (input_sequence_length - 1) * dim + 1
        nedge_in=3,  # dim + 1
        nedge_out=32,  # latent_dim // 2
        latent_dim=64,
        nmessage_passing_steps=5,
        nmlp_layers=1,
        nparticle_types=1,
        particle_type_embedding_size=9,
        num_scales=3,
        window_size=3,
        radius_multiplier=2.0,
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
    
    # Load model
    try:
        simulator.load(model_path + model_file)
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False
    
    simulator.eval()
    simulator.to(device)
    
    # Load one trajectory
    data_trajs = get_multi_scale_data_loader_by_trajectories(
        path=f"{data_path}valid.npz",
        num_scales=3,
        window_size=3,
        radius_multiplier=2.0
    )
    
    data_traj = next(iter(data_trajs))
    simulator.set_static_graph(data_traj['graph'])
    
    # Test forward pass
    try:
        positions = data_traj['positions'].to(device)
        particle_type = data_traj['particle_type'].to(device)
        n_particles_per_example = data_traj['n_particles_per_example'].to(device)
        
        # Test predict_accelerations
        current_positions = positions[:, :3]  # First 3 timesteps
        pred_acc, target_acc, pred_strain = simulator.predict_accelerations(
            next_positions=positions[:, 3],  # 4th timestep
            position_sequence_noise=torch.zeros_like(current_positions),
            position_sequence=current_positions,
            nparticles_per_example=n_particles_per_example,
            particle_types=particle_type
        )
        
        print(f"   ✅ Forward pass successful")
        print(f"   - Pred acc shape: {pred_acc.shape}")
        print(f"   - Target acc shape: {target_acc.shape}")
        print(f"   - Pred strain shape: {pred_strain.shape}")
        print(f"   - Pred acc range: [{pred_acc.min():.6f}, {pred_acc.max():.6f}]")
        print(f"   - Target acc range: [{target_acc.min():.6f}, {target_acc.max():.6f}]")
        print(f"   - Pred strain range: [{pred_strain.min():.6f}, {pred_strain.max():.6f}]")
        
        # Test predict_positions
        next_pos, next_strain = simulator.predict_positions(
            current_positions=current_positions,
            nparticles_per_example=n_particles_per_example,
            particle_types=particle_type
        )
        
        print(f"   ✅ Position prediction successful")
        print(f"   - Next pos shape: {next_pos.shape}")
        print(f"   - Next strain shape: {next_strain.shape}")
        print(f"   - Next pos range: [{next_pos.min():.6f}, {next_pos.max():.6f}]")
        print(f"   - Next strain range: [{next_strain.min():.6f}, {next_strain.max():.6f}]")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_static_graph(data_path: str):
    """Test if static graphs are being built correctly."""
    
    print("\n🔗 Testing Static Graph Construction")
    
    data_trajs = get_multi_scale_data_loader_by_trajectories(
        path=f"{data_path}valid.npz",
        num_scales=3,
        window_size=3,
        radius_multiplier=2.0
    )
    
    data_traj = next(iter(data_trajs))
    graph = data_traj['graph']
    
    print(f"   - Graph keys: {list(graph.keys())}")
    
    # Check graph hierarchy
    if 'graph_hierarchy' in graph:
        hierarchy = graph['graph_hierarchy']
        print(f"   - Graph hierarchy scales: {list(hierarchy.keys())}")
        for scale, scale_data in hierarchy.items():
            print(f"   - Scale {scale}:")
            print(f"     - Keys: {list(scale_data.keys())}")
            if 'particles' in scale_data:
                print(f"     - Particles: {scale_data['particles'].shape}")
            if 'spacing' in scale_data:
                print(f"     - Spacing: {scale_data['spacing']:.6f}")
    
    # Check edge counts
    print(f"   - G2M edges: {graph['grid2mesh_edges'].shape}")
    print(f"   - M2M edges: {graph['mesh2mesh_edges'].shape}")
    print(f"   - M2G edges: {graph['mesh2grid_edges'].shape}")
    
    # Check for any issues
    if graph['grid2mesh_edges'].shape[1] == 0:
        print(f"   ⚠️  No G2M edges!")
    if graph['mesh2mesh_edges'].shape[1] == 0:
        print(f"   ⚠️  No M2M edges!")
    if graph['mesh2grid_edges'].shape[1] == 0:
        print(f"   ⚠️  No M2G edges!")


def main():
    """Main test function."""
    
    data_path = "datasets/taylor_impact_2d/data_processed/"
    model_path = "models/"
    model_file = "latest"
    
    print("🚀 Multi-Scale Model Basic Tests")
    print("=" * 50)
    
    # Test 1: Static graph construction
    test_static_graph(data_path)
    
    # Test 2: Model forward pass
    success = test_model_forward_pass(data_path, model_path, model_file)
    
    if success:
        print("\n✅ All basic tests passed!")
        print("   The model can do forward passes. The issue might be in:")
        print("   1. Training convergence")
        print("   2. Data normalization")
        print("   3. Loss function")
        print("   4. Learning rate")
    else:
        print("\n❌ Basic tests failed!")
        print("   There are fundamental issues with the model setup.")


if __name__ == "__main__":
    main()
