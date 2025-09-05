import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Any

from gns.multi_scale.multi_scale_simulator import MultiScaleSimulator
from gns.multi_scale.data_loader_multi_scale import (
    MultiScaleTaylorImpactSamplesDataset,
    MultiScaleTaylorImpactTrajectoriesDataset
)
from gns import evaluate


def validate_multi_scale_simulator(
    simulator: MultiScaleSimulator,
    data_path: str,
    metadata: Dict[str, Any],
    device: str,
    input_sequence_length: int = 3,
    inference_mode: str = 'autoregressive',
    num_scales: int = 3,
    window_size: int = 3,
    radius_multiplier: float = 2.0
) -> Dict[str, float]:
    """Validate the multi-scale simulator on validation trajectories.
    
    Args:
        simulator: Trained MultiScaleSimulator
        data_path: Path to validation data
        metadata: Dataset metadata
        device: PyTorch device
        input_sequence_length: Number of input timesteps
        inference_mode: 'autoregressive' or 'one_step'
        num_scales: Number of scales in multi-scale hierarchy
        window_size: Sampling window size for mesh levels
        radius_multiplier: Multiplier for all connectivity radius calculations
        
    Returns:
        Dictionary with validation metrics
    """
    print(f"🔍 Starting multi-scale validation...")
    print(f"   - Data path: {data_path}")
    print(f"   - Input sequence length: {input_sequence_length}")
    print(f"   - Inference mode: {inference_mode}")
    print(f"   - Multi-scale scales: {num_scales}")
    print(f"   - Window size: {window_size}")
    
    # Load validation trajectories
    valid_dataset = MultiScaleTaylorImpactTrajectoriesDataset(
        data_path=data_path,
        num_scales=num_scales,
        window_size=window_size,
        radius_multiplier=radius_multiplier
    )
    
    simulator.eval()
    simulator.to(device)
    
    eval_loss_total = []
    eval_loss_position = []
    eval_loss_strain = []
    eval_loss_oneStep = []
    eval_times = []
    
    with torch.no_grad():
        for example_i, data_traj in enumerate(valid_dataset):
            print(f"   Processing validation example {example_i+1}/{len(valid_dataset)}...")
            
            # Set static graph for this trajectory
            simulator.set_static_graph(data_traj['graph'])
            
            # Extract trajectory data
            nsteps = metadata['sequence_length'] - input_sequence_length
            n_particles_per_example = data_traj['n_particles_per_example'].to(device)
            positions = data_traj['positions'].to(device)
            particle_type = data_traj['particle_type'].to(device)
            strains = data_traj['strains'].to(device)
            
            # Time the prediction
            start_time = time.time()
            
            # Predict example rollout using multi-scale simulator
            example_output = evaluate_multi_scale_rollout(
                simulator=simulator,
                positions=positions,
                particle_type=particle_type,
                n_particles_per_example=n_particles_per_example,
                strains=strains,
                nsteps=nsteps,
                dim=metadata['dim'],
                device=device,
                input_sequence_length=input_sequence_length,
                inference_mode=inference_mode
            )
            
            end_time = time.time()
            run_time = end_time - start_time
            
            # Calculate losses
            loss_total = example_output['rmse_position'][-1] + example_output['rmse_strain'][-1]
            loss_position = example_output['rmse_position'][-1]
            loss_strain = example_output['rmse_strain'][-1]
            loss_oneStep = example_output['rmse_position'][0] + example_output['rmse_strain'][0]
            
            print(f"     Example {example_i+1} - Loss total: {loss_total:.6f}, "
                  f"Position: {loss_position:.6f}, Strain: {loss_strain:.6f}, "
                  f"Time: {run_time:.2f}s")
            
            eval_loss_total.append(loss_total)
            eval_loss_position.append(loss_position)
            eval_loss_strain.append(loss_strain)
            eval_loss_oneStep.append(loss_oneStep)
            eval_times.append(run_time)
    
    # Calculate mean metrics
    metrics = {
        'val/loss_total': np.mean(eval_loss_total),
        'val/loss_position': np.mean(eval_loss_position),
        'val/loss_strain': np.mean(eval_loss_strain),
        'val/loss_oneStep': np.mean(eval_loss_oneStep),
        'val/mean_time': np.mean(eval_times),
        'val/std_time': np.std(eval_times)
    }
    
    print(f"✅ Multi-scale validation completed:")
    print(f"   - Mean total loss: {metrics['val/loss_total']:.6f}")
    print(f"   - Mean position loss: {metrics['val/loss_position']:.6f}")
    print(f"   - Mean strain loss: {metrics['val/loss_strain']:.6f}")
    print(f"   - Mean one-step loss: {metrics['val/loss_oneStep']:.6f}")
    print(f"   - Mean prediction time: {metrics['val/mean_time']:.2f}s ± {metrics['val/std_time']:.2f}s")
    
    return metrics


def evaluate_multi_scale_rollout(
    simulator: MultiScaleSimulator,
    positions: torch.Tensor,
    particle_type: torch.Tensor,
    n_particles_per_example: torch.Tensor,
    strains: torch.Tensor,
    nsteps: int,
    dim: int,
    device: str,
    input_sequence_length: int,
    inference_mode: str = 'autoregressive'
) -> Dict[str, Any]:
    """Evaluate rollout using multi-scale simulator.
    
    This is a multi-scale version of the evaluate.rollout function.
    """
    # Initialize rollout
    current_positions = positions[:, :input_sequence_length].clone()  # (nparticles, timesteps, dim)
    current_strains = strains[:input_sequence_length, :].clone()  # (timesteps, nparticles)
    
    predicted_positions = []
    predicted_strains = []
    rmse_positions = []
    rmse_strains = []
    
    for step in range(nsteps):
        # Get target for this step
        target_position = positions[:, input_sequence_length + step]
        target_strain = strains[input_sequence_length + step, :]  # strains is (timesteps, nparticles)
        
        # Predict next step
        if inference_mode == 'autoregressive':
            # Use predicted positions for next step
            next_positions, next_strains = simulator.predict_positions(
                current_positions=current_positions,
                nparticles_per_example=n_particles_per_example,
                particle_types=particle_type
            )
        else:  # one_step
            # Always use ground truth for next step
            next_positions, next_strains = simulator.predict_positions(
                current_positions=current_positions,
                nparticles_per_example=n_particles_per_example,
                particle_types=particle_type
            )
        
        # Calculate RMSE for this step
        position_error = torch.norm(next_positions - target_position, dim=-1)
        strain_error = torch.abs(next_strains - target_strain)
        
        rmse_position = torch.sqrt(torch.mean(position_error ** 2))
        rmse_strain = torch.sqrt(torch.mean(strain_error ** 2))
        
        # Store results
        predicted_positions.append(next_positions.cpu().numpy())
        predicted_strains.append(next_strains.cpu().numpy())
        rmse_positions.append(rmse_position.item())
        rmse_strains.append(rmse_strain.item())
        
        # Update current positions for next iteration
        if inference_mode == 'autoregressive':
            # Shift window: remove oldest, add predicted
            current_positions = torch.cat([
                current_positions[:, 1:],
                next_positions.unsqueeze(1)
            ], dim=1)
            # For strains: current_strains is (timesteps, nparticles), next_strains is (nparticles,)
            current_strains = torch.cat([
                current_strains[1:, :],  # Remove first timestep
                next_strains.unsqueeze(0)  # Add new timestep
            ], dim=0)
        else:
            # For one_step mode, we still need to update the window
            # but we use ground truth for the actual prediction
            current_positions = torch.cat([
                current_positions[:, 1:],
                target_position.unsqueeze(1)
            ], dim=1)
            current_strains = torch.cat([
                current_strains[1:, :],  # Remove first timestep
                target_strain.unsqueeze(0)  # Add new timestep
            ], dim=0)
    
    return {
        'predicted_positions': np.array(predicted_positions),
        'predicted_strains': np.array(predicted_strains),
        'rmse_position': np.array(rmse_positions),
        'rmse_strain': np.array(rmse_strains),
        'run_time': 0.0  # Will be set by caller
    }


def validate_during_training(
    simulator: MultiScaleSimulator,
    data_path: str,
    metadata: Dict[str, Any],
    device: str,
    input_sequence_length: int = 3,
    inference_mode: str = 'autoregressive',
    num_scales: int = 3,
    window_size: int = 3,
    radius_multiplier: float = 2.0,
    max_examples: int = 5
) -> Dict[str, float]:
    """Lightweight validation during training (limited examples for speed).
    
    Args:
        simulator: Trained MultiScaleSimulator
        data_path: Path to validation data
        metadata: Dataset metadata
        device: PyTorch device
        input_sequence_length: Number of input timesteps
        inference_mode: 'autoregressive' or 'one_step'
        num_scales: Number of scales in multi-scale hierarchy
        window_size: Sampling window size for mesh levels
        radius_multiplier: Multiplier for all connectivity radius calculations
        max_examples: Maximum number of examples to validate (for speed)
        
    Returns:
        Dictionary with validation metrics
    """
    print(f"🔍 Quick validation during training (max {max_examples} examples)...")
    
    # Load validation trajectories
    valid_dataset = MultiScaleTaylorImpactTrajectoriesDataset(
        data_path=data_path,
        num_scales=num_scales,
        window_size=window_size,
        radius_multiplier=radius_multiplier
    )
    
    simulator.eval()
    simulator.to(device)
    
    eval_loss_total = []
    eval_loss_position = []
    eval_loss_strain = []
    
    with torch.no_grad():
        for example_i, data_traj in enumerate(valid_dataset):
            if example_i >= max_examples:
                break
                
            # Set static graph for this trajectory
            simulator.set_static_graph(data_traj['graph'])
            
            # Extract trajectory data
            nsteps = metadata['sequence_length'] - input_sequence_length
            n_particles_per_example = data_traj['n_particles_per_example'].to(device)
            positions = data_traj['positions'].to(device)
            particle_type = data_traj['particle_type'].to(device)
            strains = data_traj['strains'].to(device)
            
            # Predict example rollout
            example_output = evaluate_multi_scale_rollout(
                simulator=simulator,
                positions=positions,
                particle_type=particle_type,
                n_particles_per_example=n_particles_per_example,
                strains=strains,
                nsteps=nsteps,
                dim=metadata['dim'],
                device=device,
                input_sequence_length=input_sequence_length,
                inference_mode=inference_mode
            )
            
            # Calculate losses
            loss_total = example_output['rmse_position'][-1] + example_output['rmse_strain'][-1]
            loss_position = example_output['rmse_position'][-1]
            loss_strain = example_output['rmse_strain'][-1]
            
            eval_loss_total.append(loss_total)
            eval_loss_position.append(loss_position)
            eval_loss_strain.append(loss_strain)
    
    # Calculate mean metrics
    metrics = {
        'val/loss_total': np.mean(eval_loss_total),
        'val/loss_position': np.mean(eval_loss_position),
        'val/loss_strain': np.mean(eval_loss_strain)
    }
    
    print(f"   Quick validation - Total: {metrics['val/loss_total']:.6f}, "
          f"Position: {metrics['val/loss_position']:.6f}, "
          f"Strain: {metrics['val/loss_strain']:.6f}")
    
    return metrics
