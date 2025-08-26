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

from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from datasets.taylor_impact_2d.taylor_impact_data_loader import (
    get_data_loader_by_samples,
    get_data_loader_by_trajectories
)
from gns import evaluate

# Meta parameters
flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'], help=(
        'Train model, validation or rollout evaluation.'))
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')

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
flags.DEFINE_enum('inference_mode', 'autoregressive', ['autoregressive', 'one_step'], 
                  help='Inference mode: "autoregressive" for rollout using predicted positions, "one_step" for pure one-step prediction using ground truth.')

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


KINEMATIC_PARTICLE_ID = -1


def predict(
        simulator: learned_simulator.LearnedSimulator,
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

    data_trajs = get_data_loader_by_trajectories(
        path=f"{FLAGS.data_path}{split}.npz")

    eval_loss = []
    with torch.no_grad():
        for example_i, data_traj in enumerate(data_trajs):
            nsteps = metadata['sequence_length'] - FLAGS.input_sequence_length
            n_particles_per_example = data_traj['n_particles_per_example'].to(device)
            positions = data_traj['positions'].to(device)
            particle_type = data_traj['particle_type'].to(device)
            strains = data_traj['strains'].to(device)
                            
            # Predict example rollout
            example_output = evaluate.rollout(simulator,
                                              positions,
                                              particle_type,
                                              n_particles_per_example,
                                              strains,
                                              nsteps,
                                              FLAGS.dim,
                                              device,
                                              FLAGS.input_sequence_length,
                                              FLAGS.inference_mode)

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
                  
                  
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
 
                        
def load_model(simulator, FLAGS, device):
    if FLAGS.model_file == "latest" and FLAGS.train_state_file == "latest":
        # find the latest model, assumes model and train_state files are in step.
        fnames = glob.glob(f"{FLAGS.model_path}*model*pt")
        max_model_number = 0
        expr = re.compile(".*model-(\d+).pt")
        for fname in fnames:
            model_num = int(expr.search(fname).groups()[0])
            if model_num > max_model_number:
                max_model_number = model_num
        # reset names to point to the latest.
        FLAGS.model_file = f"model-{max_model_number}.pt"
        FLAGS.train_state_file = f"train_state-{max_model_number}.pt"

    if os.path.exists(FLAGS.model_path + FLAGS.model_file) and os.path.exists(
        FLAGS.model_path + FLAGS.train_state_file):
        # load model
        simulator.load(FLAGS.model_path + FLAGS.model_file)

        # load train state
        train_state = torch.load(FLAGS.model_path + FLAGS.train_state_file)
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
    

def train(
        simulator: learned_simulator.LearnedSimulator,
        metadata: json,
        device: str):
    """Train the model.
  
    Args:
      simulator: Get LearnedSimulator.
    """
    optimizer = torch.optim.Adam(simulator.parameters(), lr=FLAGS.lr_init)
    step = 0
    # If model_path does not exist create new directory and begin training.
    model_path = FLAGS.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    # If model_path does exist and model_file and train_state_file exist continue training.
    if FLAGS.model_file is not None:
        simulator, step = load_model(simulator, FLAGS, device)
        
    simulator.train()
    simulator.to(device)

    data_samples = get_data_loader_by_samples(path=f"{FLAGS.data_path}train.npz",
                                            input_length_sequence=FLAGS.input_sequence_length,
                                            batch_size=FLAGS.batch_size,
                                            )
    
    # Test graph connectivity once at the beginning if debug is enabled
    if FLAGS.debug_graph:
        print("🔍 Testing graph connectivity with first batch...")
        for data_sample in data_samples:
            position = data_sample['input']['positions'].to(device)
            n_particles_per_example = data_sample['input']['n_particles_per_example'].to(device)
            
            print(f"   Position tensor shape: {position.shape}")
            print(f"   Particles per example: {n_particles_per_example}")
            
            # Extract most recent positions (last timestep) for graph testing
            # position shape: [nparticles*batch_size, input_sequence_length, dim]
            # We need: [nparticles*batch_size, dim] (most recent positions)
            most_recent_positions = position[:, -1, :]  # Last timestep
            print(f"   Most recent positions shape: {most_recent_positions.shape}")
            
            simulator.test_graph_connectivity_once(most_recent_positions, n_particles_per_example)
            break  # Only test with first batch
    
    ### =================================== Training loop ============================================
    not_reached_nsteps = True
    lowest_eval_loss = 10000
    
    # Debug mode: limit training to 5 steps
    if FLAGS.debug_graph:
        debug_max_steps = 5
        print(f"🔍 Debug mode: Training limited to {debug_max_steps} steps")
        FLAGS.ntraining_steps = min(FLAGS.ntraining_steps, debug_max_steps)
    
    try:
        while not_reached_nsteps:
            for data_sample in data_samples:
                log = {}  # wandb logging
                # position are of size (nparticles*batch_size, FLAGS.input_sequence_length, dim)
                
                position = data_sample['input']['positions'].to(device)
                particle_type = data_sample['input']['particle_type'].to(device)
                n_particles_per_example = data_sample['input']['n_particles_per_example'].to(device)
                next_position = data_sample['output']['next_position'].to(device)
                next_strain = data_sample['output']['next_strain'].to(device)
                
                # TODO (jpv): Move noise addition to data_loader
                # Sample the noise to add to the inputs to the model during training.
                sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(position,
                                                                                        noise_std_last_step=FLAGS.noise_std).to(device)
                non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device)
                sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

                # Get the predictions and target accelerations.
                pred_acc, target_acc, pred_strain = simulator.predict_accelerations(
                    next_positions=next_position,
                    position_sequence_noise=sampled_noise,
                    position_sequence=position,
                    nparticles_per_example=n_particles_per_example,
                    particle_types=particle_type
                )

                # Calculate the loss and mask out loss on kinematic particles
                loss_pos = (pred_acc - target_acc) ** 2
                loss_xy = loss_pos.mean(axis=0)  # for log purpose

                # if 1d, compute loss on x-axis only
                loss_pos = loss_pos.sum(dim=-1)

                # Calculate loss
                loss_strain = (pred_strain - next_strain) ** 2
                loss = loss_pos + loss_strain
                num_non_kinematic = non_kinematic_mask.sum()
                loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
                loss = loss.sum() / num_non_kinematic

                # Computes the gradient of loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update learning rate
                lr_new = FLAGS.lr_init * (FLAGS.lr_decay ** (step / FLAGS.lr_decay_steps)) + 1e-6
                for param in optimizer.param_groups:
                    param['lr'] = lr_new
                
                # Print training info
                if FLAGS.debug_graph:
                    remaining_steps = FLAGS.ntraining_steps - step - 1
                    print(f'🔍 Debug Training step: {step+1}/{FLAGS.ntraining_steps} (remaining: {remaining_steps}). Loss: {loss:.6f}')
                else:
                    print('Training step: {}/{}. Loss: {}.'.format(step, FLAGS.ntraining_steps, loss))
                
                # WandB logging
                log["train/loss"] = loss
                log["train/loss-x"] = loss_xy[0]
                log["train/loss-y"] = loss_xy[1]
                if FLAGS.dim == 3:
                    log["train/loss-z"] = loss_xy[2]
                log["train/loss-strain"] = loss_strain.mean()
                log["lr"] = lr_new

                ### =================================== Validation rollout ============================================
                if step != 0 and step % FLAGS.nsave_steps == 0:
                    # Validation in the training loop
                    simulator.eval()
                    # Output path
                    if not os.path.exists(FLAGS.output_path):
                        os.makedirs(FLAGS.output_path)

                    # Use `valid`` set for eval mode if not use `test`
                    split = 'test' if FLAGS.mode == 'rollout' else 'valid'
                    data_trajs = get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz")

                    eval_loss_total, eval_loss_position, eval_loss_strain, eval_loss_oneStep = [], [], [], []
                    with torch.no_grad():
                        for example_i, data_traj in enumerate(data_trajs):
                            nsteps = metadata['sequence_length'] - FLAGS.input_sequence_length
                            n_particles_per_example = data_traj['n_particles_per_example'].to(device)
                            positions = data_traj['positions'].to(device)
                            particle_type = data_traj['particle_type'].to(device)
                            strains = data_traj['strains'].to(device)
                            
                            # Predict example rollout
                            example_output = evaluate.rollout(simulator,
                                                              positions,
                                                              particle_type,
                                                              n_particles_per_example,
                                                              strains,
                                                              nsteps,
                                                              FLAGS.dim,
                                                              device,
                                                              FLAGS.input_sequence_length,
                                                              FLAGS.inference_mode)
                            
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
                                print(f"===================Saving.=============================")
                                save_dir = osp.join(model_path, FLAGS.run_name)
                                if not os.path.exists(save_dir):
                                    os.makedirs(save_dir)
                                simulator.save(osp.join(save_dir, f'model-{step:06}.pt'))
                                train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step": step})
                                #torch.save(train_state, osp.join(save_dir, f'train_state-{step:06}.pt'))

                        # log
                        log["val/loss"] = sum(eval_loss_total) / len(eval_loss_total)
                        log["val/loss-position"] = sum(eval_loss_position) / len(eval_loss_position)
                        log["val/loss-strain"] = sum(eval_loss_strain) / len(eval_loss_strain)
                        log["val/rmse-oneStep"] = sum(eval_loss_oneStep) / len(eval_loss_oneStep)
                    # =========================================================

                # Complete training
                if (step >= FLAGS.ntraining_steps):
                    not_reached_nsteps = False
                    break

                if FLAGS.log:
                    wandb.log(log, step=step)
                    
                step += 1

    except KeyboardInterrupt:
        pass

    # Debug mode completion message
    if FLAGS.debug_graph:
        print(f"\n🔍 Debug training completed after {step} steps")
        print(f"   Graph connectivity testing finished")
        print(f"   No model saved (debug mode)")
        print(f"   Use --debug_graph=False for full training")
        print(f"   Next time, run: python gns/train.py --debug_graph=False --mode=train")
    else:
        # Save model only for full training
        save_dir = osp.join(model_path, FLAGS.run_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        simulator.save(osp.join(save_dir, f'model-{step:06}.pt'))
        train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step": step})
        torch.save(train_state, osp.join(save_dir, f'train_state-{step:06}.pt'))


def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: str) -> learned_simulator.LearnedSimulator:
    """Instantiates the simulator.
  
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
    
    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=FLAGS.dim,  # xyz
        nnode_in=(FLAGS.input_sequence_length - 1) * FLAGS.dim + 1,  # (input_sequence_length-1) velocity timesteps * dim + wall distance
        nedge_in=FLAGS.dim + 1,    # input edge features, relative displacement in all dims + distance between two nodes
        latent_dim=FLAGS.hidden_dim,
        nmessage_passing_steps=FLAGS.layers,
        nmlp_layers=1,
        mlp_hidden_dim=FLAGS.hidden_dim,
        connectivity_radius=FLAGS.connection_radius,
        normalization_stats=normalization_stats,
        nparticle_types=num_particle_types,
        particle_type_embedding_size=FLAGS.particle_type_embedding_size,
        device=device)
    
    # Set debug flag if enabled
    if FLAGS.debug_graph:
        simulator._debug_graph = True
        print(f"🔍 Graph connectivity debugging enabled")

    return simulator


def main(_):
    """Train or evaluates the model.
  
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    
    # Print debug information if enabled
    if FLAGS.debug_graph:
        print(f"🔍 Graph connectivity debugging enabled")
        print(f"   - Will test graph construction during training")
        print(f"   - Will show detailed connectivity analysis")
        print(f"   - Training limited to 5 steps for quick testing")
        print(f"   - Use --debug_graph=False to disable")
    
    # Print inference mode information
    print(f"🎯 Inference mode: {FLAGS.inference_mode}")
    if FLAGS.inference_mode == 'autoregressive':
        print(f"   - Autoregressive rollout: uses predicted positions for next step")
    elif FLAGS.inference_mode == 'one_step':
        print(f"   - One-step prediction: always uses ground truth for next step")

    # Read metadata
    metadata = reading_utils.read_metadata(FLAGS.data_path)
    simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device)

    if FLAGS.mode == 'train':
        # Init wandb
        if FLAGS.log:
            wandb.init(project=FLAGS.project_name, name=FLAGS.run_name)
            train(simulator, metadata, device)
            wandb.finish()
        else:
            train(simulator, metadata, device)

    elif FLAGS.mode in ['valid', 'rollout']:
        predict(simulator, metadata, device)


if __name__ == '__main__':
    app.run(main)
