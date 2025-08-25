# Datasets

This folder contains dataset-specific code and configurations for the GNS project.

## Taylor Impact Bar Dataset

The Taylor Impact Bar dataset is a 2D simulation dataset for testing Graph Neural Network (GNS) models on impact dynamics.

### Files

- **`build_dataset.py`** - Main dataset builder script
- **`dataset_config.yaml`** - Configuration file with all parameters
- **`taylor_impact_data_loader.py`** - Dataset-specific PyTorch data loader
- **`data_loader_adapter.py`** - Compatibility adapter for existing training code
- **`training_config.yaml`** - Example training configuration
- **`README.md`** - This comprehensive documentation file
- **`data_processed/`** - Output directory containing processed datasets

### File Structure
```
taylor_impact_2d/
├── build_dataset.py           # Main dataset builder
├── dataset_config.yaml        # Configuration file
├── README.md                  # Documentation
└── data_processed/            # Output directory
    ├── train.npz             # Training data (199MB)
    ├── valid.npz             # Validation data (22MB)
    ├── test.npz              # Test data (22MB)
    └── metadata.json         # Dataset statistics and SPH config
```

### Raw Data Location

The raw simulation data can be organized in several ways for cross-computer compatibility:

#### Option 1: Local raw_data directory (Recommended)
```
taylor_impact_2d/
├── raw_data/                  # Your NPZ files go here
│   ├── 60-120.npz
│   ├── 60-130.npz
│   └── ... (other NPZ files)
├── build_dataset.py
├── dataset_config.yaml
└── data_processed/            # Output directory
```

#### Option 2: OneDrive symbolic link
Create a symbolic link to your OneDrive data location:
```bash
# Windows (PowerShell as Administrator)
New-Item -ItemType SymbolicLink -Path "raw_data" -Target "C:\Users\YourUsername\OneDrive - Curtin\research\civil_engineering\data\2D-Copper-Bar-Taylor-Impact\npz"

# Linux/Mac
ln -s "/path/to/your/OneDrive/data/2D-Copper-Bar-Taylor-Impact/npz" "raw_data"
```

#### Option 3: Custom path in config
Update `dataset_config.yaml` with your specific path:
```yaml
raw_data_path: "C:/Users/YourUsername/OneDrive - Curtin/research/civil_engineering/data/2D-Copper-Bar-Taylor-Impact/npz"
```

**Cross-Computer**: The build script automatically detects common OneDrive locations, so the same code works on different computers without modification!

### Quick Start

**1. Organize your data** in one of these ways:
   - Create a `raw_data/` directory with your NPZ files
   - Use symbolic links to your OneDrive location
   - Update the config file with your specific path

**2. Build the dataset**:
   ```bash
   python build_dataset.py
   ```
   
   Or with a custom config file:
   ```bash
   python build_dataset.py --config custom_config.yaml
   ```

**3. Train the model** (using the new dataset-specific data loader):
   ```bash
   # From the project root directory
   python -m gns.train --data_path=./datasets/taylor_impact_2d/data_processed/ --dim=2 --batch_size=2
   
   # Or use the provided training config
   python -m gns.train --config=./datasets/taylor_impact_2d/training_config.yaml
   ```

**Expected Output**:
```
Building Taylor Impact Bar Dataset
==================================================
Input directory: C:/Users/kylin/OneDrive - Curtin/research/civil_engineering/data/2D-Copper-Bar-Taylor-Impact/npz
Output directory: ./data_processed
Step size: 2
Total steps: 100
Random seed: 42

Found 33 NPZ files
Processing simulations...
  Training: 27 simulations → 1350 timesteps
  Validation: 3 simulations → 150 timesteps  
  Test: 3 simulations → 150 timesteps
Computing global statistics...
  Von Mises stress: mean=150.26, std=83.51
  Metadata saved → ./data_processed/metadata.json
```

**Performance**: Processing 33 simulations (~243MB raw data) typically takes 2-5 minutes on a modern system.

### Data Loader Architecture

The Taylor Impact dataset uses a **dedicated, dataset-specific data loader** instead of the unified approach:

#### **Why Dataset-Specific?**

1. **Data Structure Understanding**: Our data loader knows the exact format `(positions, particle_types, stresses)`
2. **Better Error Handling**: Clear validation and informative error messages
3. **Performance Optimization**: Optimized for the specific data patterns
4. **Maintainability**: Easy to modify for dataset-specific requirements
5. **Type Safety**: Better type hints and validation

#### **Components:**

- **`taylor_impact_data_loader.py`**: Core PyTorch datasets and data loaders
- **`data_loader_adapter.py`**: Drop-in replacement for `gns.data_loader`
- **`training_config.yaml`**: Example configuration for training

#### **Usage Modes:**

1. **Training Mode** (`get_data_loader_by_samples`): Provides individual samples with input sequences
2. **Evaluation Mode** (`get_data_loader_by_trajectories`): Provides complete trajectories for rollout

#### **Integration:**

The new data loader maintains the same interface as the original, so existing training code works without modification. Simply replace the import:

```python
# Old approach (unified)
from gns import data_loader

# New approach (dataset-specific)
from datasets.taylor_impact_2d import taylor_impact_data_loader as data_loader
```

### Configuration

The `dataset_config.yaml` file contains all configurable parameters:

- **Data paths**: Raw data and output locations
- **Dataset parameters**: Step size, total steps, random seed, random seed
- **Graph connectivity**: Default radius for neighbor search (0.6 mm)
- **SPH simulation**: Timestep, smoothing length, material properties from .k file
- **Validation/test splits**: Specific simulation identifiers (velocity-impact combinations)
- **Processing**: Boundary particle removal, von Mises stress thresholds
- **Output structure**: Directory organization for processed data

**Note**: Von Mises stress statistics (mean and standard deviation) are computed automatically from the data during processing.

**SPH Configuration**: The dataset includes complete SPH simulation parameters extracted from the .k file:
- **Smoothing Length**: 1.2 × particle spacing = 0.6 mm
- **Particle Spacing**: 0.5 mm (uniform grid)
- **Material Properties**: Copper (density: 8.9 g/cm³, shear modulus: 37.6 GPa)
- **Simulation Parameters**: 0.3s duration, 0.002s timestep, max 12 neighbors

**Important**: The NPZ files contain a field named `strains` which actually contains von Mises stress data. This naming inconsistency is preserved for compatibility with existing data files.

**Example Output**: During processing, the script will display unique particle types for each simulation:
```
Processing T-20-60-100.npz...
  Unique particle types: [1, 2, 3]
  Von Mises stress range: 0.001 to 450.2 MPa
```

### Output Structure

The processed dataset will be saved as:
```
data_processed/
├── train.npz          # Training data
├── valid.npz          # Validation data  
├── test.npz           # Test data
└── metadata.json      # Dataset statistics and metadata
```

### Dependencies

**Required Python packages**:
- `numpy` ≥ 1.19.0
- `pyyaml` ≥ 5.1
- `pathlib` (built-in, Python ≥ 3.4)

**Installation**:
```bash
pip install numpy pyyaml
```

### Notes

- The dataset builder automatically handles train/val/test splits
- All parameters are configurable via the YAML file
- The script includes comprehensive error handling and progress reporting
- Statistics are collected and saved for model training

### Troubleshooting

**Common Issues**:
1. **File not found**: Ensure the raw data path in `dataset_config.yaml` is correct
2. **Permission errors**: Check write permissions for the output directory
3. **Memory issues**: For large datasets, consider reducing `total_steps` or `step_size`
4. **YAML parsing errors**: Verify the configuration file syntax is valid

**Debug Mode**: Add `--verbose` flag for detailed processing information (if implemented)

## Numerical Model (SPH) Metadata from .k file 

```json
{
  "model": {
    "discretisation": "SPH",
    "coordinate_system": "Cartesian",
    "length_unit": "model units (likely mm)",
    "time_unit": "s"
  },
  "domain_bounds": {
    "x_min": 0.25,
    "x_max": 59.75,
    "y_min": -9.75,
    "y_max": 9.75,
    "z_min": 0.0,
    "z_max": 0.0
  },
  "particle_layout": {
    "plane": "z = 0",
    "spacing": {
      "dx": 0.5,
      "dy": 0.5,
      "dz": 0.0
    },
    "grid_points": {
      "x_first": 0.25,
      "x_step": 0.5,
      "x_last": 59.75,
      "x_count_inferred": 120,
      "y_first": -9.75,
      "y_step": 0.5,
      "y_last": 9.75,
      "y_count_inferred": 40
    },
    "particle_counts": {
      "expected_from_grid": 4800,
      "observed_in_file": 4801,
      "note": "Observed count is 1 higher than ideal lattice; likely an extra or off-grid node."
    }
  },
  "sph_parameters": {
    "cslh": 1.2,
    "hmin": 1.0,
    "hmax": 1.0,
    "max_neighbors_nmneigh": 12
  },
  "material": {
    "model": "MAT_ELASTIC_PLASTIC_HYDRO",
    "density_g_per_mm3": 0.0089,
    "shear_modulus_MPa": 37590
  },
  "runtime_controls": {
    "termination_time_s": 0.3,
    "output_interval_s": 0.002,
    "timestep_note": "No explicit CONTROL_TIMESTEP; LS-DYNA uses automatic stable Δt (CFL-based)."
  }
}
```
