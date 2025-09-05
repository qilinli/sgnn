# MultiScaleGraph Test Suite

This directory contains comprehensive unit tests for the `MultiScaleGraph` class and related functionality.

## 📁 Test Files

- **`test_multi_scale_graph.py`** - Main test suite covering all MultiScaleGraph functionality
- **`run_tests.py`** - Test runner script that discovers and executes all tests
- **`README.md`** - This documentation file

## 🧪 Test Coverage

The test suite covers the following areas:

### 1. MultiScaleConfig Tests
- ✅ Default configuration values
- ✅ Custom configuration values
- ✅ Attribute accessibility

### 2. MultiScaleGraph Core Tests
- ✅ Initialization and setup
- ✅ Hierarchy building (grid → mesh levels)
- ✅ Sampling index chaining
- ✅ Complete edge generation

### 3. Edge Validation Tests
- ✅ Edge index validation
- ✅ Edge symmetry (grid2mesh ↔ mesh2grid)
- ✅ No self-loops
- ✅ No duplicate edges

### 4. Edge Case Tests
- ✅ Single particle handling
- ✅ Two particle handling
- ✅ Invalid configuration handling
- ✅ Empty positions handling

### 5. Configuration Variation Tests
- ✅ Large window sizes
- ✅ Many scales
- ✅ Different sampling strategies

### 6. Consistency Tests
- ✅ Multiple call consistency
- ✅ Hierarchy persistence

## 🚀 Running Tests

### Option 1: Run All Tests
```bash
cd gns/multi_scale/test
python run_tests.py
```

### Option 2: Run Specific Test File
```bash
cd gns/multi_scale/test
python -m unittest test_multi_scale_graph.py -v
```

### Option 3: Run Specific Test Class
```bash
cd gns/multi_scale/test
python -m unittest test_multi_scale_graph.TestMultiScaleConfig -v
```

### Option 4: Run Specific Test Method
```bash
cd gns/multi_scale/test
python -m unittest test_multi_scale_graph.TestMultiScaleConfig.test_default_config -v
```

## 📊 Test Output

The test runner provides detailed output including:
- Test discovery and execution
- Pass/fail status for each test
- Detailed error messages for failures
- Summary statistics
- Exit codes for CI/CD integration

## 🔧 Test Data

Tests use synthetic data including:
- **4×4 Grid**: 16 particles for basic functionality testing
- **8×8 Grid**: 64 particles for larger scale testing
- **16×16 Grid**: 256 particles for stress testing
- **Edge Cases**: Single particles, minimal configurations

## 🎯 Test Philosophy

The test suite follows these principles:
1. **Comprehensive Coverage**: Test all public methods and edge cases
2. **Fast Execution**: Use small, synthetic datasets for quick feedback
3. **Deterministic Results**: Tests produce consistent, reproducible results
4. **Clear Assertions**: Each test validates specific, well-defined behavior
5. **Isolation**: Tests don't depend on external data or state

## 🚨 Troubleshooting

### Import Errors
If you encounter import errors, ensure you're running from the correct directory:
```bash
cd gns/multi_scale/test
python run_tests.py
```

### PyTorch Issues
Ensure PyTorch is properly installed and accessible:
```bash
python -c "import torch; print(torch.__version__)"
```

### Test Failures
If tests fail, check:
1. PyTorch version compatibility
2. File permissions and paths
3. Python environment and dependencies

## 📈 Adding New Tests

To add new tests:

1. **Create Test Class**: Inherit from `unittest.TestCase`
2. **Follow Naming Convention**: Use descriptive test method names
3. **Add to Test File**: Place in appropriate test file or create new one
4. **Run Tests**: Verify new tests pass before committing

Example:
```python
def test_new_feature(self):
    """Test new feature functionality."""
    # Arrange
    expected = "expected_value"
    
    # Act
    actual = self.graph.new_feature()
    
    # Assert
    self.assertEqual(actual, expected)
```

## 🔗 Integration

These tests can be integrated with:
- **CI/CD Pipelines**: Use exit codes for pass/fail detection
- **Development Workflow**: Run before commits and merges
- **Documentation**: Tests serve as usage examples
- **Debugging**: Isolated tests help identify issues quickly
