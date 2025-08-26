# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Simple matplotlib rendering of a rollout prediction against ground truth.

Usage (from parent directory):

`python -m gns.render_rollout_taylor_impact_2d --rollout_path={OUTPUT_PATH}/rollout_test_1.pkl --output_path={OUTPUT_PATH}/rollout_test_1.gif`

Where {OUTPUT_PATH} is the output path passed to `train.py` in "eval_rollout"
mode.

It may require installing Tkinter with `sudo apt-get install python3.7-tk`.

"""  # pylint: disable=line-too-long

import pickle
from pathlib import Path

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file")
flags.DEFINE_string("output_path", None, help="Path to output fig file")
flags.DEFINE_integer("step_stride", 1, help="Stride of steps to skip.")

FLAGS = flags.FLAGS

# Default normalization parameters (will be overridden by metadata if available)
MAX, MIN = np.array([100, 50]), np.array([-2.5, -50])
STRAIN_MEAN, STRAIN_STD = 150.25897834554806, 83.50737010164767  # von Mises stress stats

# Animation configuration constants
FRAMES_TO_SAVE = [30, 60, 90]  # Key timesteps to save as images
ANIMATION_INTERVAL = 50  # ms delay between frames
ANIMATION_FPS = 5
SAVE_DPI = 100
PLOT_PADDING = 5
WALL_X = -2
WALL_OFFSET = 0.4
WALL_THICKNESS = 8
WALL_SHADOW_OFFSET = 0.1
WALL_TEXTURE_OFFSET = 0.05

def load_rollout_data(rollout_path):
    """Load rollout data from pickle file."""
    if not Path(rollout_path).exists():
        raise FileNotFoundError(f"Rollout file not found: {rollout_path}")
    
    with open(rollout_path, "rb") as file:
        return pickle.load(file)

def load_metadata_config(rollout_data):
    """Load and apply metadata configuration."""
    global MAX, MIN, STRAIN_MEAN, STRAIN_STD
    
    if "metadata" in rollout_data:
        metadata = rollout_data["metadata"]
        
        # Update position bounds
        if "pos_max" in metadata and "pos_min" in metadata:
            MAX = np.array(metadata["pos_max"])
            MIN = np.array(metadata["pos_min"])
        
        # Update stress normalization
        if "stress_mean" in metadata and "stress_std" in metadata:
            STRAIN_MEAN = metadata["stress_mean"]
            STRAIN_STD = metadata["stress_std"]
        
        print(f"Loaded from metadata: MAX={MAX}, MIN={MIN}, strain_mean={STRAIN_MEAN:.2f}, strain_std={STRAIN_STD:.2f}")
    else:
        print("Using default values (no metadata found)")

def create_rigid_wall(ax, x_min, x_max, y_min, y_max):
    """Create a realistic rigid wall at x=-2."""
    if WALL_X < x_min - PLOT_PADDING or WALL_X > x_max + PLOT_PADDING:
        return
    
    # Main wall line - thick and solid (positioned so entire wall is left of x=-2)
    ax.axvline(x=WALL_X - WALL_OFFSET, color='darkgray', linewidth=WALL_THICKNESS, 
               alpha=0.9, label='Rigid Wall')
    
    # Add shadow effect for depth
    ax.axvline(x=WALL_X - WALL_OFFSET + WALL_SHADOW_OFFSET, color='lightgray', 
               linewidth=WALL_THICKNESS//2, alpha=0.5)
    
    # Add wall texture/pattern (vectorized for better performance)
    wall_y_range = np.linspace(y_min - PLOT_PADDING, y_max + PLOT_PADDING, 20)
    texture_x1 = WALL_X - WALL_OFFSET - WALL_TEXTURE_OFFSET
    texture_x2 = WALL_X - WALL_OFFSET + WALL_TEXTURE_OFFSET
    
    for y in wall_y_range:
        ax.plot([texture_x1, texture_x2], [y, y], color='black', linewidth=1, alpha=0.6)

def setup_subplot(ax, label, x_min, x_max, y_min, y_max):
    """Set up subplot with consistent styling."""
    ax.set_title(label)
    ax.set_xlim(x_min - PLOT_PADDING, x_max + PLOT_PADDING)
    ax.set_ylim(y_min - PLOT_PADDING, y_max + PLOT_PADDING)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.)

def process_trajectory_data(rollout_data, rollout_field):
    """Process trajectory data with denormalization."""
    # Combine initial positions with rollout trajectory
    trajectory = np.concatenate([
        rollout_data["initial_positions"],
        rollout_data[rollout_field]
    ], axis=0)
    
    # Denormalize positions
    trajectory = trajectory * (MAX - MIN) + MIN
    return trajectory

def process_strain_data(rollout_data, label):
    """Process strain data with denormalization."""
    # Load strain data
    if label == "LS-DYNA":
        strain = rollout_data["ground_truth_strain"]
        strain_gt = strain * STRAIN_STD + STRAIN_MEAN  # denormalize for colorbar
    elif label == "GNN":
        strain = rollout_data["predicted_strain"]
    
    # Add initial strains and denormalize
    strain = np.concatenate((rollout_data["initial_strains"], strain), axis=0)
    strain = strain * STRAIN_STD + STRAIN_MEAN  # denormalize
    
    return strain, strain_gt if label == "LS-DYNA" else None

def main(unused_argv):   
    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    
    # Load rollout data
    rollout_data = load_rollout_data(FLAGS.rollout_path)
    
    # Load metadata and override default values if available
    load_metadata_config(rollout_data)

    # Create figure with subplots: LS-DYNA, GNN, and colorbar
    fig, axes = plt.subplots(1, 3, figsize=(20, 10), gridspec_kw={"width_ratios":[10,10,0.5]})
    
    plot_info = []
    strain_gt = None  # Will be set during LS-DYNA processing
    
    for ax_i, (label, rollout_field) in enumerate([
        ("LS-DYNA", "ground_truth_rollout"),
        ("GNN", "predicted_rollout")
    ]):
    
        # Process trajectory and strain data
        trajectory = process_trajectory_data(rollout_data, rollout_field)
        strain, current_strain_gt = process_strain_data(rollout_data, label)
        
        if label == 'LS-DYNA':
            trajectory_gt = trajectory
            strain_gt = current_strain_gt
        
        # Calculate plot bounds
        x_min, y_min = trajectory_gt.min(axis=(0,1))
        x_max, y_max = trajectory_gt.max(axis=(0,1))
        
        # Set up subplot
        ax = axes[ax_i]
        setup_subplot(ax, label, x_min, x_max, y_min, y_max)
        
        # Add rigid wall
        create_rigid_wall(ax, x_min, x_max, y_min, y_max)
        
        # Set up colorbar (only once, using LS-DYNA strain bounds)
        if label == "LS-DYNA":
            cmap = matplotlib.cm.rainbow
            norm = matplotlib.colors.Normalize(vmin=strain_gt.min(axis=(0,1)), vmax=strain_gt.max(axis=(0,1)))
            cb = matplotlib.colorbar.ColorbarBase(axes[-1], cmap=cmap, norm=norm, orientation='vertical')
            cb.set_label('Von Mises Stress (MPa)', fontsize=15)
            cb.ax.tick_params(labelsize=12)
        
        # Create scatter plot for material particles (colored by stress)
        concrete_points = ax.scatter([], [], c=[], s=6, cmap="rainbow", 
                                   vmin=strain_gt.min(axis=(0,1)), vmax=strain_gt.max(axis=(0,1)))
        
        plot_info.append((trajectory, strain, concrete_points, {}))

    # Add legend for particle types and rigid wall
    axes[0].legend(loc='upper right', fontsize=10)

    num_steps = trajectory.shape[0]   
    
    def update(step_i):
        """Update animation frame."""
        outputs = []
        
        for trajectory, strain, concrete_points, other_points in plot_info:
            # Update material particle positions and colors
            concrete_points.set_offsets(trajectory[step_i, :])
            concrete_points.set_array(strain[step_i, :])
            outputs.append(concrete_points)
            
            # Update other particle type positions (if any)
            if other_points:  # Only process if there are other particle types
                for particle_type, line in other_points.items():
                    mask = rollout_data["particle_types"] == particle_type
                    line.set_data(trajectory[step_i, mask, 0], trajectory[step_i, mask, 1])
                    outputs.append(line)
        
        # Save key frames
        if step_i in FRAMES_TO_SAVE: 
            frame_path = FLAGS.output_path.replace('.gif', f'_frame{step_i}.png')
            plt.savefig(frame_path, dpi=SAVE_DPI)
        
        return outputs

    # Create and save animation
    animation_obj = animation.FuncAnimation(
        fig, update,
        frames=np.arange(0, num_steps, FLAGS.step_stride), 
        interval=ANIMATION_INTERVAL
    )

    animation_obj.save(FLAGS.output_path, dpi=SAVE_DPI, fps=ANIMATION_FPS, writer='pillow')
    print(f"Animation saved to {FLAGS.output_path}")

if __name__ == "__main__":
    app.run(main)
