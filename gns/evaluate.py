
import numpy as np
import torch
import tree
import time
from absl import flags
from absl import app

from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from datasets.taylor_impact_2d.taylor_impact_data_loader import (
    get_data_loader_by_trajectories
)

# INPUT_SEQUENCE_LENGTH is now configurable via FLAGS.input_sequence_length from train.py
EROSIONAL_PARTICLE_ID = -1  # so broken particle will not contribute to loss

# Configuration constants
RMSE_PRINT_INTERVAL = 3  # Steps between RMSE prints


def rollout_rmse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Rollout error in accumulated RMSE.
  
    Args:
        pred: prediction of shape [timesteps, nparticles, dim]
        gt: groundtruth of the same shape
    Returns:
        loss: accumulated rmse loss of shape (timesteps,), where
        loss[t] is the average rmse loss of rollout prediction of t steps
    """
    # Validate input shapes
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    num_timesteps = gt.shape[0]
    squared_diff = np.square(pred - gt).reshape(num_timesteps, -1)
    loss = np.sqrt(np.cumsum(np.mean(squared_diff, axis=1), axis=0)/np.arange(1, num_timesteps+1))

    # Print RMSE at regular intervals
    for show_step in range(0, num_timesteps, RMSE_PRINT_INTERVAL):
        if show_step < num_timesteps:
            print('Testing rmse @ step %d loss: %.2e'%(show_step, loss[show_step]))
        else: 
            break

    return loss


@torch.no_grad()
def rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.Tensor,
        particle_types: torch.Tensor,
        n_particles_per_example: torch.Tensor,
        strains: torch.Tensor,
        nsteps: int,
        particle_dim: int,
        device: torch.device,
        input_sequence_length: int = 3,
        inference_mode: str = 'autoregressive') -> dict:
    """Rolls out a trajectory by applying the model recursively.
  
    Args:
      simulator: Learned simulator.
      position: Position tensor of shape [nparticles, timesteps, dim].
      particle_types: Particle type tensor of shape [nparticles].
      n_particles_per_example: Number of particles per example.
      strains: Strain tensor of shape [timesteps, nparticles].
      nsteps: Number of steps.
      particle_dim: Particle dimension.
      device: Device to run on.
      input_sequence_length: Number of input timesteps for velocity calculation.
             inference_mode: Inference mode - 'autoregressive' for rollout using predicted positions, 'one_step' for pure one-step prediction.
    
    Returns:
      dict: Dictionary containing rollout results including:
        - initial_positions: Initial position sequence
        - initial_strains: Initial strain sequence  
        - predicted_rollout: Predicted trajectory
        - ground_truth_rollout: Ground truth trajectory
        - ground_truth_strain: Ground truth strain
        - predicted_strain: Predicted strain
        - particle_types: Particle type information
        - rmse_position: Position RMSE over time
        - rmse_strain: Strain RMSE over time
        - run_time: Total execution time
        - inference_mode: The inference mode used
    """
    # Input validation
    if position.dim() != 3:
        raise ValueError(f"Position tensor must be 3D, got {position.dim()}D")
    if strains.dim() != 2:
        raise ValueError(f"Strains tensor must be 2D, got {strains.dim()}D")
    if position.shape[0] != strains.shape[1]:
        raise ValueError(f"Number of particles mismatch: position {position.shape[0]} vs strains {strains.shape[1]}")
    if position.shape[1] < input_sequence_length:
        raise ValueError(f"Position sequence length {position.shape[1]} must be >= input_sequence_length {input_sequence_length}")
    
    # position is of shape [nparticles, timestep, dim], strains [timestep, nparticles]
    initial_positions = position[:, :input_sequence_length]  # First input_sequence_length timesteps
    initial_strains = strains[:input_sequence_length,:]      # First input_sequence_length timesteps
    ground_truth_positions = position[:, input_sequence_length:]  # Remaining timesteps for prediction
    ground_truth_strains = strains[input_sequence_length:, :]     # Remaining timesteps for prediction
    nsteps = ground_truth_strains.shape[0]  # For 2D-T, nsteps vary between 29 and 30
    
    current_positions = initial_positions
    pred_positions = []
    pred_strains = []
    
    # Create mask for erosional particles (broken particles that follow prescribed trajectory)
    erosional_mask = (particle_types == EROSIONAL_PARTICLE_ID).clone().detach().to(device)
    erosional_mask = erosional_mask.bool()[:, None].expand(-1, particle_dim)
    
    start_time = time.time()
    for step in range(nsteps):
        # Get next position with shape (nnodes, dim)
        next_position, pred_strain = simulator.predict_positions(
            current_positions,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_types,
        )

        # Update erosional particles from prescribed trajectory.
        next_position_ground_truth = ground_truth_positions[:, step]
        next_strain_ground_truth = ground_truth_strains[step, :]
        next_position = torch.where(
            erosional_mask, next_position_ground_truth, next_position)
        pred_strain = torch.where(
            erosional_mask[:, 0], next_strain_ground_truth, pred_strain)
        pred_positions.append(next_position)
        pred_strains.append(pred_strain)

        # Update current_positions based on inference mode
        if inference_mode == 'autoregressive':
            # Autoregressive mode: use predicted position for next step
            current_positions = torch.cat(
                [current_positions[:, 1:], next_position[:, None, :]], dim=1)
        elif inference_mode == 'one_step':
            # One-step mode: always use ground truth for next step
            current_positions = torch.cat(
                [current_positions[:, 1:], next_position_ground_truth[:, None, :]], dim=1)
        else:
            raise ValueError(f"Unknown inference_mode: {inference_mode}. Must be 'autoregressive' or 'one_step'") 
    
    run_time = time.time() - start_time
    
    # Predictions with shape (time, nnodes, dim)
    pred_positions = torch.stack(pred_positions)
    pred_strains = torch.stack(pred_strains)
    ground_truth_positions = ground_truth_positions.permute(1, 0, 2)
    
    # Note that this rmse loss is not comparable with training loss (MSE)
    # Besides, training loss is measured on acceleration,
    # while rollout loss is on position
    rmse_position = rollout_rmse(pred_positions.cpu().numpy(), 
                                 ground_truth_positions.cpu().numpy())
    rmse_strain = rollout_rmse(pred_strains.cpu().numpy(), 
                               ground_truth_strains.cpu().numpy())  
    output_dict = {
        'initial_positions': initial_positions.permute(1, 0, 2).cpu().numpy(),
        'initial_strains': initial_strains.cpu().numpy(),
        'predicted_rollout': pred_positions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
        'ground_truth_strain': ground_truth_strains.cpu().numpy(),
        'predicted_strain': pred_strains.cpu().numpy(),
        'particle_types': particle_types.cpu().numpy(),
        'rmse_position': rmse_position,
        'rmse_strain': rmse_strain,
        'run_time': run_time,
        'inference_mode': inference_mode
    }

    return output_dict