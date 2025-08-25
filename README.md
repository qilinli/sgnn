# Machine Learning for Physics Simulation

## Installation

### 1. Set Up Python Environment  
Ensure you have Python 3.11 installed (tested version). Create a new Conda environment:  
```bash
conda create -n cgn python=3.11
conda activate cgn
```

### 2. Install PyTorch (Preferably with GPU & CUDA)  
Check your GPU and CUDA compatibility before installing. The following command installs PyTorch 2.5.0 with CUDA 11.8:  
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```
For different CUDA versions, refer to [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

### 3. Install PyTorch Geometric (PyG)  
Assuming PyTorch 2.5.0 with CUDA 11.8:  
```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
```
For other versions, refer to [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 4. Install Other Dependencies  
```bash
pip install -r requirements.txt
```

---

## Usage  

### 1. Download Dataset  
The **Taylor Impact 2D** dataset can be downloaded from [OneDrive](https://curtin-my.sharepoint.com/:f:/g/personal/272766h_curtin_edu_au/Egs3Uw10Ic9KqLyW2-CZkAsBMjg165h-SbB-9VQW9-DA_g).

### 2. Preprocess Data  
Run the following command to preprocess the dataset: 
```bash
python -m gns.data_preprocessing --in_dir=[PATH_TO_NPZ_DIR] \
                                 --out_dir=[PATH_TO_DATA_DIR] \
                                 --total_step=100 \
                                 --step_size=2 \
                                 --dataset=Concrete-2D-T
```

### 3. Train the Model  
```bash
python -m gns.train --mode=train \
                    --data_path=[PATH_TO_DATA_DIR] \
                    --model_path=./models/taylor/ \
                    --output_path=./rollouts/taylor/ \
                    --batch_size=1 \
                    --noise_std=0.1 \
                    --connection_radius=15 \
                    --layers=5 \
                    --hidden_dim=64 \
                    --lr_init=0.001 \
                    --ntraining_steps=30000 \
                    --lr_decay_steps=9000 \
                    --dim=2 \
                    --project_name=Segment-3D \
                    --run_name=NS01_R15_L5N64 \
                    --nsave_steps=1000 \
                    --log=False
```

### 4. Evaluate the Model  
```bash
python -m gns.train --mode=rollout \
                    --data_path=[PATH_TO_DATA_DIR] \
                    --model_path=./models/taylor/ \
                    --model_file=NS01_R15_L5N64/model-0010000.pt \
                    --output_path=./rollouts/taylor/ \
                    --batch_size=1 \
                    --noise_std=0.1 \
                    --connection_radius=15 \
                    --layers=5 \
                    --hidden_dim=64 \
                    --dim=2
```

### 5. Visualize Results  
```bash
python -m gns.render_rollout_2d_T --rollout_path={OUTPUT_PATH}/rollout_test_1.pkl \
                                  --output_path={OUTPUT_PATH}/rollout_test_1.gif
```

## Notes  
- Ensure that the dataset is properly formatted before preprocessing.  
- Adjust the hyperparameters as needed for optimal model performance.  
- GPU acceleration is recommended for training and evaluation.