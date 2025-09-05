#!/usr/bin/env python3
"""
Multi-Scale GNN Training Script for Taylor Impact Dataset

This script trains a MultiScaleSimulator on the Taylor Impact dataset using static multi-scale graphs.
It extends the original training script to support multi-scale graph neural networks.
"""

import collections
import json
import numpy as np
import os
import os.path as osp
import sys
import torch
import pickle
import glob
import re

import tree
import wandb
import time

from absl import flags
from absl import app

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from gns.multi_scale.multi_scale_simulator import MultiScaleSimulator
from gns import noise_utils
from gns import reading_utils
from gns.multi_scale.data_loader_multi_scale import (
    get_multi_scale_data_loader_by_samples,
    get_multi_scale_data_loader_by_trajectories
)
from gns.multi_scale import validate_multi_scale
from gns.multi_scale.validate_multi_scale import validate_during_training, validate_multi_scale_simulator
from gns import evaluate

# Meta parameters
flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'], help=(
        'Train model, validation or rollout evaluation.'))
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')

# Multi-scale parameters
flags.DEFINE_integer('num_scales', 3, help='Number of scales in the multi-scale hierarchy.')
flags.DEFINE_integer('window_size', 3, help='Sampling window size for mesh levels.')
flags.DEFINE_float('radius_multiplier', 2.0, help='Multiplier for all connectivity radius calculations.')

# Model parameters
flags.DEFINE_float('connection_radius', 0.6, help='connectivity radius for graph.')
flags.DEFINE_integer('layers', 5, help='Number of GNN layers.')
flags.DEFINE_integer('hidden_dim', 64, help='Number of neurons in hidden layers.')
flags.DEFINE_integer('dim', 3, help='The dimension of concrete simulation.')
flags.DEFINE_integer('particle_type_embedding_size', 9, help='Embedding size for particle types.')
flags.DEFINE_integer('input_sequence_length', 3, help='Number of input timesteps for velocity calculation.')

# Training parameters
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_float('noise_std', 2e-2, help='The std deviation of the noise.')
flags.DEFINE_integer('ntraining_steps', int(1E6), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')

# Debug parameters
flags.DEFINE_bool('debug_graph', False, help='Enable graph connectivity debugging and testing.')

# Inference mode parameters
flags.DEFINE_enum('inference_mode', 'autoregressive', ['autoregressive', 'one_step'], help=(
    'Inference mode: autoregressive or one_step prediction.'))

# Continue training parameters
flags.DEFINE_string('model_file', None, help=(
    'Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', 'train_state.pt', help=(
    'Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-3, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(4e5), help='Learning rate decay steps.')

# Wandb log parameters
flags.DEFINE_bool('log', False, help='if use wandb log.')
flags.DEFINE_string('project_name', 'GNS-tmp', help='project name for wandb log.')
flags.DEFINE_string('run_name', 'runrunrun', help='Run name for wandb log.')

FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])


def predict(
        simulator: MultiScaleSimulator,
        metadata: json,
        device: str):
    """Predict rollouts.
  
    Args:
      simulator: Trained simulator if not will exit.
      metadata: Metadata for test set.
  
    """
    # Load simulator
    try:
        simulator.load(FLAGS.model_path + FLAGS.model_file)
    except:
        print("Failed to load model weights!")
        sys.exit(1)

    simulator.to(device)
    simulator.eval()

    # Output path
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    # Use `valid`` set for eval mode if not use `test`
    split = 'test' if FLAGS.mode == 'rollout' else 'valid'

    data_trajs = get_multi_scale_data_loader_by_trajectories(
        path=osp.join(FLAGS.data_path, f'{split}.npz'),
        num_scales=FLAGS.num_scales,
        window_size=FLAGS.window_size,
        radius_multiplier=FLAGS.radius_multiplier
    )

    eval_loss = []
    with torch.no_grad():
        for example_i, data_traj in enumerate(data_trajs):
            nsteps = metadata['sequence_length'] - FLAGS.input_sequence_length
            n_particles_per_example = data_traj['n_particles_per_example'].to(device)
            positions = data_traj['positions'].to(device)
            particle_type = data_traj['particle_type'].to(device)
            strains = data_traj['strains'].to(device)
            
            # Set static graph for this trajectory
            simulator.set_static_graph(data_traj['graph'])
                            
            # Predict example rollout using multi-scale evaluate function
            example_output = validate_multi_scale.evaluate_multi_scale_rollout(
                simulator=simulator,
                positions=positions,
                particle_type=particle_type,
                n_particles_per_example=n_particles_per_example,
                strains=strains,
                nsteps=nsteps,
                dim=FLAGS.dim,
                device=device,
                input_sequence_length=FLAGS.input_sequence_length,
                inference_mode=FLAGS.inference_mode
            )

            example_output['metadata'] = metadata

            # RMSE loss with shape (time,)
            loss_total = example_output['rmse_position'][-1] + example_output['rmse_strain'][-1]
            loss_position = example_output['rmse_position'][-1]
            loss_strain = example_output['rmse_strain'][-1]
            loss_oneStep = example_output['rmse_position'][0] + example_output['rmse_strain'][0]  

            print(f'''Predicting example {example_i}-
                  {example_output['metadata']['file_valid'][example_i]} 
                  loss_toal: {loss_total}, 
                  loss_position: {loss_position}, 
                  loss_strain: {loss_strain}''')
            print(f"Prediction example {example_i} takes {example_output['run_time']}")
            eval_loss.append(loss_total)

            # Save rollout in testing
            if FLAGS.mode == 'rollout':
                example_output['metadata'] = metadata
                # Use the actual case name from metadata instead of generic rollout_i
                simulation_name = metadata['file_test'][example_i]
                # Remove .npz extension and create .pkl filename
                case_name = simulation_name.replace('.npz', '')
                
                # Store the actual case name used for this rollout
                example_output['case_name'] = case_name
                
                filename = f'{case_name}.pkl'
                
                # Create subfolder using run_name, similar to model saving
                save_dir = osp.join(FLAGS.output_path, FLAGS.run_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                filename = os.path.join(save_dir, filename)
                with open(filename, 'wb') as f:
                    pickle.dump(example_output, f)

    print("Mean loss on rollout prediction: {}".format(
        sum(eval_loss) / len(eval_loss)))


def load_model(simulator, FLAGS, device):
    """Load model and training state."""
    model_path = FLAGS.model_path + FLAGS.run_name + '/'
    
    if os.path.exists(model_path + FLAGS.model_file) and os.path.exists(
        model_path + FLAGS.train_state_file):
        # load model
        simulator.load(model_path + FLAGS.model_file)

        # load train state
        train_state = torch.load(model_path + FLAGS.train_state_file)
        # set optimizer state
        optimizer = torch.optim.Adam(simulator.parameters())
        optimizer.load_state_dict(train_state["optimizer_state"])
        optimizer_to(optimizer, device)
        # set global train state
        step = train_state["global_train_state"].pop("step")

    else:
        msg = f'''Specified model_file {model_path + FLAGS.model_file}
        and train_state_file {model_path + FLAGS.train_state_file} not found.'''
        raise FileNotFoundError(msg)
    
    return simulator, step


def optimizer_to(optimizer, device):
    """Move optimizer to device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def train(
        simulator: MultiScaleSimulator,
        metadata: json,
        device: str):
    """Train the model.
  
    Args:
      simulator: Get MultiScaleSimulator.
    """
    optimizer = torch.optim.Adam(simulator.parameters(), lr=FLAGS.lr_init)
    
    # Load training data
    data_samples = get_multi_scale_data_loader_by_samples(
        path=osp.join(FLAGS.data_path, 'train.npz'),
        input_length_sequence=FLAGS.input_sequence_length,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_scales=FLAGS.num_scales,
        window_size=FLAGS.window_size,
        radius_multiplier=FLAGS.radius_multiplier
    )
    
    # Set static graph for the simulator
    # Note: In a real implementation, you might want to handle multiple trajectories
    # For now, we'll use the first batch's graph
    first_batch = next(iter(data_samples))
    simulator.set_static_graph(first_batch['graph'])
    
    print(f"🚀 Starting multi-scale GNN training...")
    print(f"   - Scales: {FLAGS.num_scales}, Window: {FLAGS.window_size}")
    print(f"   - Batch size: {FLAGS.batch_size}")
    print(f"   - Training steps: {FLAGS.ntraining_steps}")
    print(f"   - Learning rate: {FLAGS.lr_init}")
    
    step = 0
    not_reached_nsteps = True
    lowest_eval_loss = 10000
    
    try:
        while not_reached_nsteps:
            for data_sample in data_samples:
                log = {}  # wandb logging
                # Move data to device
                position = data_sample['input']['positions'].to(device)
                particle_type = data_sample['input']['particle_type'].to(device)
                n_particles_per_example = data_sample['input']['n_particles_per_example'].to(device)
                next_position = data_sample['output']['next_position'].to(device)
                next_strain = data_sample['output']['next_strain'].to(device)
                
                # Set static graph for this batch
                simulator.set_static_graph(data_sample['graph'])
                
                # Sample noise for training
                sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
                    position, noise_std_last_step=FLAGS.noise_std
                ).to(device)
                
                # Forward pass
                optimizer.zero_grad()
                
                pred_acc, target_acc, pred_strain = simulator.predict_accelerations(
                    next_positions=next_position,
                    position_sequence_noise=sampled_noise,
                    position_sequence=position,
                    nparticles_per_example=n_particles_per_example,
                    particle_types=particle_type
                )
                
                # Calculate the loss
                loss_pos = (pred_acc - target_acc) ** 2
                loss_xy = loss_pos.mean(axis=0)  # for log purpose

                # if 1d, compute loss on x-axis only
                loss_pos = loss_pos.sum(dim=-1)

                # Calculate strain loss
                loss_strain = (pred_strain - next_strain) ** 2
                loss = loss_pos + loss_strain
                loss = loss.mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update learning rate
                lr_new = FLAGS.lr_init * (FLAGS.lr_decay ** (step / FLAGS.lr_decay_steps)) + 1e-6
                for param in optimizer.param_groups:
                    param['lr'] = lr_new
                
                step += 1
                
                # WandB logging
                log["train/loss"] = loss
                log["train/loss-x"] = loss_xy[0]
                log["train/loss-y"] = loss_xy[1]
                if FLAGS.dim == 3:
                    log["train/loss-z"] = loss_xy[2]
                log["train/loss-strain"] = loss_strain.mean()
                log["lr"] = lr_new
                
                if step % 10 == 0:
                    print(f"Step {step}: Total Loss = {loss.item():.6f}, Position Loss = {loss_pos.mean().item():.6f}, Strain Loss = {loss_strain.mean().item():.6f}")
                
                # Save model and validate periodically
                if step % FLAGS.nsave_steps == 0 and step > 0:
                    save_dir = osp.join(FLAGS.model_path, FLAGS.run_name)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    simulator.save(osp.join(save_dir, f'model-{step:06}.pt'))
                    train_state = dict(
                        optimizer_state=optimizer.state_dict(), 
                        global_train_state={"step": step}
                    )
                    torch.save(train_state, osp.join(save_dir, f'train_state-{step:06}.pt'))
                    print(f"💾 Model saved at step {step}")
                    
                    # Full validation during training
                    print(f"🔍 Running full validation at step {step}...")
                    simulator.eval()
                    
                    # Load validation trajectories
                    data_trajs = get_multi_scale_data_loader_by_trajectories(
                        path=f"{FLAGS.data_path}valid.npz",
                        num_scales=FLAGS.num_scales,
                        window_size=FLAGS.window_size,
                        radius_multiplier=FLAGS.radius_multiplier
                    )
                    
                    eval_loss_total, eval_loss_position, eval_loss_strain, eval_loss_oneStep = [], [], [], []
                    with torch.no_grad():
                        for example_i, data_traj in enumerate(data_trajs):
                            nsteps = metadata['sequence_length'] - FLAGS.input_sequence_length
                            n_particles_per_example = data_traj['n_particles_per_example'].to(device)
                            positions = data_traj['positions'].to(device)
                            particle_type = data_traj['particle_type'].to(device)
                            strains = data_traj['strains'].to(device)
                            
                            # Set static graph for this trajectory
                            simulator.set_static_graph(data_traj['graph'])
                            
                            # Predict example rollout using multi-scale evaluate function
                            example_output = validate_multi_scale.evaluate_multi_scale_rollout(
                                simulator=simulator,
                                positions=positions,
                                particle_type=particle_type,
                                n_particles_per_example=n_particles_per_example,
                                strains=strains,
                                nsteps=nsteps,
                                dim=FLAGS.dim,
                                device=device,
                                input_sequence_length=FLAGS.input_sequence_length,
                                inference_mode=FLAGS.inference_mode
                            )
                            
                            example_output['metadata'] = metadata
                            
                            # RMSE loss with shape (time,)
                            loss_total = example_output['rmse_position'][-1] + example_output['rmse_strain'][-1]
                            loss_position = example_output['rmse_position'][-1]
                            loss_strain = example_output['rmse_strain'][-1]
                            loss_oneStep = example_output['rmse_position'][0] + example_output['rmse_strain'][0]
                            
                            print(f'''Predicting example {example_i}-
                                  {example_output['metadata']['file_valid'][example_i]} 
                                  loss_total: {loss_total}, 
                                  loss_position: {loss_position}, 
                                  loss_strain: {loss_strain}''')
                            print(f"Prediction example {example_i} takes {example_output['run_time']}")
                            eval_loss_total.append(loss_total)
                            eval_loss_position.append(loss_position)
                            eval_loss_strain.append(loss_strain)
                            eval_loss_oneStep.append(loss_oneStep)
                        
                        eval_loss_mean = sum(eval_loss_total) / len(eval_loss_total)
                        print(f"Mean loss on valid-set rollout prediction: {eval_loss_mean}. Current lowest eval loss is {lowest_eval_loss}.")
                        
                        # Save the current best model based on eval loss
                        if eval_loss_mean < lowest_eval_loss:
                            print(f"===================Better model obtained.=============================")
                            lowest_eval_loss = eval_loss_mean
                            if step > 1000:
                                print(f"===================Saving best model.=============================")
                                simulator.save(osp.join(save_dir, f'model-best-{step:06}.pt'))
                        
                        # Log validation metrics
                        log["val/loss"] = sum(eval_loss_total) / len(eval_loss_total)
                        log["val/loss-position"] = sum(eval_loss_position) / len(eval_loss_position)
                        log["val/loss-strain"] = sum(eval_loss_strain) / len(eval_loss_strain)
                        log["val/rmse-oneStep"] = sum(eval_loss_oneStep) / len(eval_loss_oneStep)
                    
                    # Set back to training mode
                    simulator.train()
                
                if FLAGS.log:
                    wandb.log(log, step=step)
                    
                if step >= FLAGS.ntraining_steps:
                    not_reached_nsteps = False
                    break
                    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    # Save model
    save_dir = osp.join(FLAGS.model_path, FLAGS.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    simulator.save(osp.join(save_dir, f'model-{step:06}.pt'))
    train_state = dict(
        optimizer_state=optimizer.state_dict(), 
        global_train_state={"step": step}
    )
    torch.save(train_state, osp.join(save_dir, f'train_state-{step:06}.pt'))
    
    print(f"✅ Training completed! Model saved to {save_dir}")


def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: str) -> MultiScaleSimulator:
    """Instantiates the multi-scale simulator.
  
    Args:
      metadata: JSON object with metadata.
      acc_noise_std: Acceleration noise std deviation.
      vel_noise_std: Velocity noise std deviation.
      device: PyTorch device 'cpu' or 'cuda'.
    """

    # Normalization stats
    normalization_stats = {
        'acceleration': {
            'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
            'std': torch.sqrt(torch.FloatTensor(metadata['acc_std']) ** 2 +
                              acc_noise_std ** 2).to(device),
        },
        'velocity': {
            'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
            'std': torch.sqrt(torch.FloatTensor(metadata['vel_std']) ** 2 +
                              vel_noise_std ** 2).to(device),
        },
    }
    
    # Compute number of particle types dynamically from metadata
    num_particle_types = metadata.get('num_particle_types', 1)
    print(f"Detected {num_particle_types} particle types from metadata")
    
    # Calculate node input features
    # (input_sequence_length-1) velocity timesteps * dim + wall distance + particle type embedding (if multiple types)
    nnode_in = (FLAGS.input_sequence_length - 1) * FLAGS.dim + 1
    if num_particle_types > 1:
        nnode_in += FLAGS.particle_type_embedding_size
    
    simulator = MultiScaleSimulator(
        kinematic_dimensions=FLAGS.dim,
        nnode_in=nnode_in,
        nedge_in=FLAGS.dim + 1,  # relative displacement + distance
        nedge_out=FLAGS.hidden_dim,  # latent edge dimension
        latent_dim=FLAGS.hidden_dim,
        nmessage_passing_steps=FLAGS.layers,
        nmlp_layers=2,
        normalization_stats=normalization_stats,
        nparticle_types=num_particle_types,
        particle_type_embedding_size=FLAGS.particle_type_embedding_size,
        num_scales=FLAGS.num_scales,
        window_size=FLAGS.window_size,
        radius_multiplier=FLAGS.radius_multiplier,
        device=device
    )
    
    print(f"✅ MultiScaleSimulator created:")
    print(f"   - Kinematic dimensions: {FLAGS.dim}")
    print(f"   - Node input features: {nnode_in}")
    print(f"   - Edge input features: {FLAGS.dim + 1}")
    print(f"   - Latent dimension: {FLAGS.hidden_dim}")
    print(f"   - Message passing steps: {FLAGS.layers}")
    print(f"   - Multi-scale scales: {FLAGS.num_scales}")
    print(f"   - Window size: {FLAGS.window_size}")
    print(f"   - Radius multiplier: {FLAGS.radius_multiplier}")

    return simulator


def main(_):
    """Train or evaluates the model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    
    # Load metadata
    metadata_path = osp.join(FLAGS.data_path, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create simulator
    simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device)
    simulator.to(device)
    
    if FLAGS.mode == 'train':
        # Init wandb
        if FLAGS.log:
            wandb.init(project=FLAGS.project_name, name=FLAGS.run_name)
            train(simulator, metadata, device)
            wandb.finish()
        else:
            train(simulator, metadata, device)
    elif FLAGS.mode == 'valid':
        # Load model for validation
        if FLAGS.model_file is not None:
            model_path = osp.join(FLAGS.model_path, FLAGS.run_name, FLAGS.model_file)
            if os.path.exists(model_path):
                simulator.load(model_path)
                print(f"✅ Loaded model from {model_path}")
            else:
                print(f"❌ Model file not found: {model_path}")
                return
        
        # Run full validation
        val_metrics = validate_multi_scale_simulator(
            simulator=simulator,
            data_path=f"{FLAGS.data_path}valid.npz",
            metadata=metadata,
            device=device,
            input_sequence_length=FLAGS.input_sequence_length,
            inference_mode=FLAGS.inference_mode,
            num_scales=FLAGS.num_scales,
            window_size=FLAGS.window_size,
            radius_multiplier=FLAGS.radius_multiplier
        )
        
        print(f"🎯 Final validation results:")
        for key, value in val_metrics.items():
            print(f"   {key}: {value:.6f}")
    elif FLAGS.mode == 'rollout':
        predict(simulator, metadata, device)


if __name__ == '__main__':
    app.run(main)
