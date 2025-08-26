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

Single file:
`python -m gns.render_rollout_taylor_impact_2d --rollout_path={OUTPUT_PATH}/T-20-100-170.pkl --output_path={OUTPUT_PATH}/T-20-100-170.gif`

Batch processing (entire folder):
`python -m gns.render_rollout_taylor_impact_2d --rollout_path={OUTPUT_PATH}/runrunrun/ --output_path={OUTPUT_PATH}/animations/ --batch_mode=True`

Where {OUTPUT_PATH} is the output path passed to `train.py` in "rollout" mode.
The rollout files are now named with their case identifiers (e.g., T-20-100-170.pkl)
instead of generic names like rollout_0.pkl.

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

flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file or folder containing .pkl files")
flags.DEFINE_string("output_path", None, help="Path to output fig file or folder for batch processing")
flags.DEFINE_integer("step_stride", 1, help="Stride of steps to skip.")
flags.DEFINE_bool("batch_mode", False, help="Enable batch processing of entire folder")

FLAGS = flags.FLAGS

# Default normalization parameters (will be overridden by metadata if available)
MAX, MIN = np.array([100, 50]), np.array([-2.5, -50])
STRAIN_MEAN, STRAIN_STD = 150.25897834554806, 83.50737010164767  # von Mises stress stats

# Animation configuration constants
FRAMES_TO_SAVE_RATIOS = []  # Empty list = no frames saved. Use [0.25, 0.5, 0.75] to save frames at 25%, 50%, 75% of total timesteps
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

def find_pkl_files(folder_path):
    """Find all .pkl files in a folder."""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found or not a directory: {folder_path}")
    
    pkl_files = list(folder.glob("*.pkl"))
    if not pkl_files:
        raise ValueError(f"No .pkl files found in folder: {folder_path}")
    
    return sorted(pkl_files)

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

def add_performance_info(ax, rollout_data, label):
    """Add RMSE and runtime information to the subplot."""
    # Only show performance info on GNN side
    if label != "GNN":
        return
        
    if "rmse_position" in rollout_data and "rmse_strain" in rollout_data:
        # Get final RMSE values (last timestep)
        final_rmse_pos = rollout_data["rmse_position"][-1]
        final_rmse_strain = rollout_data["rmse_strain"][-1]
        
        # Format the text
        info_text = f"RMSE Position: {final_rmse_pos:.4f}\nRMSE Strain: {final_rmse_strain:.4f}"
        
        # Add text box with performance metrics in bottom right corner
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, fontweight='bold')
    
    # Add runtime information
    if "run_time" in rollout_data:
        runtime = rollout_data["run_time"]
        runtime_text = f"Runtime: {runtime:.3f}s"
        ax.text(0.98, 0.15, runtime_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=9)

def add_case_info(fig, rollout_data):
    """Add case information to the main figure."""
    # First try to get case name from the rollout data itself
    case_name = "Unknown Case"
    
    if "case_name" in rollout_data:
        # Use the stored case name from the rollout
        case_name = rollout_data["case_name"]
    elif "metadata" in rollout_data and "file_test" in rollout_data["metadata"]:
        # Fallback: try to extract from metadata (for backward compatibility)
        try:
            case_names = rollout_data["metadata"]["file_test"]
            if len(case_names) > 0:
                case_name = case_names[0].replace('.npz', '')
        except:
            pass
    
    # Add case name as figure title
    fig.suptitle(f"Taylor Impact 2D Simulation: {case_name}", fontsize=16, fontweight='bold', y=0.95)

def process_trajectory_data(rollout_data, rollout_field):
    """Process trajectory data with denormalization."""
    # Combine initial positions with rollout trajectory
    trajectory = np.concatenate([
        rollout_data["initial_positions"],
        rollout_data[rollout_field]
    ], axis=0)
    
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

def process_single_rollout(rollout_path, output_path):
    """Process a single rollout file and create animation."""
    print(f"\n🎬 Processing: {Path(rollout_path).name}")
    
    # Load rollout data
    rollout_data = load_rollout_data(rollout_path)
    
    # Load metadata and override default values if available
    load_metadata_config(rollout_data)
    
    # Print performance summary
    if "rmse_position" in rollout_data and "rmse_strain" in rollout_data and "run_time" in rollout_data:
        final_rmse_pos = rollout_data["rmse_position"][-1]
        final_rmse_strain = rollout_data["rmse_strain"][-1]
        runtime = rollout_data["run_time"]
        print(f"📊 Performance Summary:")
        print(f"   Final RMSE Position: {final_rmse_pos:.6f}")
        print(f"   Final RMSE Strain: {final_rmse_strain:.6f}")
        print(f"   Total Runtime: {runtime:.3f} seconds")
        
        # Print case name if available
        if "case_name" in rollout_data:
            case_name = rollout_data["case_name"]
            print(f"   Case: {case_name}")
        elif "metadata" in rollout_data and "file_test" in rollout_data["metadata"]:
            try:
                case_names = rollout_data["metadata"]["file_test"]
                if len(case_names) > 0:
                    case_name = case_names[0].replace('.npz', '')
                    print(f"   Case: {case_name}")
            except:
                pass
        print()

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
        
        # Add performance information (RMSE and runtime)
        add_performance_info(ax, rollout_data, label)
        
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

    # Add case information to the main figure
    add_case_info(fig, rollout_data)
    
    # Add legend for particle types and rigid wall
    axes[0].legend(loc='upper right', fontsize=10)

    num_steps = trajectory.shape[0]   
    
    # Calculate which frames to save for this specific case
    frames_to_save = [int(ratio * num_steps) for ratio in FRAMES_TO_SAVE_RATIOS]
    print(f"📊 Total timesteps: {num_steps}")
    print(f"📸 Will save frames at: {frames_to_save}")
    
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
        
        # Save key frames at specific ratios of total timesteps
        frames_to_save = [int(ratio * num_steps) for ratio in FRAMES_TO_SAVE_RATIOS]
        if step_i in frames_to_save: 
            frame_path = str(output_path).replace('.gif', f'_frame{step_i}.png')
            plt.savefig(frame_path, dpi=SAVE_DPI)
            print(f"📸 Saved frame {step_i} to {frame_path}")
        
        return outputs

    # Create and save animation
    animation_obj = animation.FuncAnimation(
        fig, update,
        frames=np.arange(0, num_steps, FLAGS.step_stride), 
        interval=ANIMATION_INTERVAL
    )

    animation_obj.save(output_path, dpi=SAVE_DPI, fps=ANIMATION_FPS, writer='pillow')
    print(f"✅ Animation saved to {output_path}")
    
    # Close figure to free memory
    plt.close(fig)

def main(unused_argv):   
    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    
    if not FLAGS.output_path:
        raise ValueError("An `output_path` must be passed.")
    
    rollout_path = Path(FLAGS.rollout_path)
    output_path = Path(FLAGS.output_path)
    
    if FLAGS.batch_mode:
        # Batch processing: process all .pkl files in folder
        print(f"🔄 Batch mode enabled - processing all .pkl files in: {rollout_path}")
        
        try:
            pkl_files = find_pkl_files(rollout_path)
            print(f"📁 Found {len(pkl_files)} .pkl files")
            
            # Create output folder if it doesn't exist
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created output folder: {output_path}")
            
            # Process each file
            for i, pkl_file in enumerate(pkl_files, 1):
                print(f"\n{'='*60}")
                print(f"Processing file {i}/{len(pkl_files)}")
                print(f"{'='*60}")
                
                # Generate output filename
                case_name = pkl_file.stem  # Remove .pkl extension
                output_file = output_path / f"{case_name}.gif"
                
                print(f"📁 Input: {pkl_file}")
                print(f"📁 Output: {output_file}")
                
                try:
                    process_single_rollout(pkl_file, output_file)
                except Exception as e:
                    print(f"❌ Error processing {pkl_file.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"\n🎉 Batch processing complete! Processed {len(pkl_files)} files.")
            print(f"📁 Output saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            return
    
    else:
        # Single file processing
        if not rollout_path.exists():
            raise FileNotFoundError(f"Rollout file not found: {rollout_path}")
        
        if rollout_path.is_file():
            # Single file
            process_single_rollout(rollout_path, output_path)
        elif rollout_path.is_dir():
            # Folder but batch mode not enabled
            print(f"📁 Found folder: {rollout_path}")
            print(f"💡 Use --batch_mode=True to process all .pkl files in the folder")
            print(f"💡 Or specify a specific .pkl file path")
        else:
            raise ValueError(f"Invalid rollout_path: {rollout_path}")

if __name__ == "__main__":
    app.run(main)
