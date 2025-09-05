# 🔍 Multi-Scale GNN Debugging Guide

## Problem Description
- **Autoregressive prediction**: Both position and strain prediction are poor
- **One-step prediction**: Position is perfect, but strain is still poor
- This suggests different issues for different components

## 🎯 Debugging Strategy

### 1. **Immediate Tests** (Run these first)

```bash
# Test basic model functionality
python test_model_basic.py

# Run detailed debugging analysis
python debug_multi_scale.py
```

### 2. **Key Areas to Investigate**

#### **A. Strain Prediction Issues** (Affects both modes)
- **Check strain normalization**: Is strain data properly normalized?
- **Check strain loss**: Is strain loss being computed correctly?
- **Check strain features**: Are strain features being passed to the model?
- **Check strain targets**: Are strain targets correct?

#### **B. Autoregressive Position Issues** (Only affects autoregressive mode)
- **Check position updates**: Are predicted positions being used correctly?
- **Check window shifting**: Is the sliding window working properly?
- **Check error accumulation**: Are small errors accumulating over time?

#### **C. Model Architecture Issues**
- **Check GNN layers**: Are the multi-scale GNN layers working?
- **Check edge features**: Are edge features being computed correctly?
- **Check message passing**: Is information flowing between scales?

### 3. **Specific Debugging Steps**

#### **Step 1: Check Data Quality**
```python
# Look for these in debug output:
- Are there NaN/infinite values?
- Are position/strain ranges reasonable?
- Is the data properly normalized?
```

#### **Step 2: Check Model Forward Pass**
```python
# Test if model can do basic forward pass
- Are predictions reasonable in magnitude?
- Are there any runtime errors?
- Are gradients flowing properly?
```

#### **Step 3: Check Training Convergence**
```python
# Look at training logs:
- Is loss decreasing?
- Are position and strain losses both decreasing?
- Is the model overfitting?
```

#### **Step 4: Check Inference Logic**
```python
# Compare one-step vs autoregressive:
- Are predicted positions being used correctly in autoregressive mode?
- Is the sliding window being updated properly?
- Are strain predictions consistent between modes?
```

### 4. **Common Issues and Solutions**

#### **Issue 1: Strain Normalization**
```python
# Problem: Strain values might not be normalized properly
# Solution: Check metadata.json for strain normalization stats
# Look for: strain_mean, strain_std in metadata
```

#### **Issue 2: Loss Function**
```python
# Problem: Strain loss might not be weighted correctly
# Solution: Check if strain loss is being added to total loss
# Look for: loss = loss_pos + loss_strain in training loop
```

#### **Issue 3: Feature Engineering**
```python
# Problem: Strain features might not be included in node features
# Solution: Check _build_node_features in MultiScaleSimulator
# Look for: strain features being added to node features
```

#### **Issue 4: Autoregressive Updates**
```python
# Problem: Predicted positions might not be used correctly
# Solution: Check predict_positions function
# Look for: current_positions being updated with predictions
```

### 5. **Debugging Commands**

#### **Run with Debug Output**
```bash
# Enable debug mode in validation
python gns/multi_scale/train_multi_scale.py --mode=valid --debug=True
```

#### **Check Model Weights**
```bash
# Inspect model weights
python -c "
import torch
model = torch.load('models/your_model.pt')
print('Model keys:', list(model.keys()))
print('GNN weights shape:', model['_multi_scale_gnn.g2m_block.node_fn.0.weight'].shape)
"
```

#### **Check Data Statistics**
```bash
# Check data statistics
python -c "
import numpy as np
data = np.load('datasets/taylor_impact_2d/data_processed/valid.npz')
print('Position range:', data['positions'].min(), data['positions'].max())
print('Strain range:', data['strains'].min(), data['strains'].max())
print('Position std:', data['positions'].std())
print('Strain std:', data['strains'].std())
"
```

### 6. **Expected Debug Output**

#### **Good Model Output**
```
🔍 DEBUG Step 0:
   - Target position range: [-2.500, 100.000]
   - Target strain range: [0.000, 500.000]
   - Predicted position range: [-2.450, 99.950]
   - Predicted strain range: [0.100, 480.000]
   - Position RMSE: 0.050000
   - Strain RMSE: 0.200000
```

#### **Bad Model Output**
```
🔍 DEBUG Step 0:
   - Target position range: [-2.500, 100.000]
   - Target strain range: [0.000, 500.000]
   - Predicted position range: [0.000, 0.000]  # ❌ All zeros
   - Predicted strain range: [0.000, 0.000]    # ❌ All zeros
   - Position RMSE: 50.000000                  # ❌ Very high
   - Strain RMSE: 250.000000                   # ❌ Very high
```

### 7. **Next Steps After Debugging**

1. **If strain prediction is poor**: Check strain normalization and loss function
2. **If autoregressive position is poor**: Check position update logic
3. **If both are poor**: Check model architecture and training
4. **If model won't load**: Check model file and architecture compatibility

### 8. **Quick Fixes to Try**

#### **Fix 1: Check Strain Normalization**
```python
# In MultiScaleSimulator._build_node_features
# Make sure strain is normalized properly
strain_normalized = (current_strain - self._normalization_stats['strain']['mean']) / self._normalization_stats['strain']['std']
```

#### **Fix 2: Check Loss Weighting**
```python
# In training loop
# Make sure strain loss is weighted appropriately
loss = loss_pos + 0.1 * loss_strain  # Try different weights
```

#### **Fix 3: Check Learning Rate**
```python
# Try different learning rates
--lr_init=0.0001  # Lower learning rate
--lr_init=0.01    # Higher learning rate
```

## 🚀 Running the Debug Scripts

```bash
# 1. Test basic functionality
python test_model_basic.py

# 2. Run detailed debugging
python debug_multi_scale.py

# 3. Check specific issues
python -c "
from debug_multi_scale import debug_single_trajectory
debug_single_trajectory('datasets/taylor_impact_2d/data_processed/', 'models/', 'latest')
"
```

This should help you identify the root cause of the inference issues!
