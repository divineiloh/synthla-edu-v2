# Tests for SYNTHLA-EDU V2

This directory contains the test suite for validating SYNTHLA-EDU V2 functionality.

## Running Tests

### Install pytest
```bash
pip install pytest
```

### Run all tests
```bash
# From repository root
pytest tests/ -v
```

### Run specific test files
```bash
pytest tests/test_data_loading.py -v
pytest tests/test_synthesizers.py -v
pytest tests/test_pipeline.py -v
```

### Skip slow tests
```bash
# Fast tests only (skip CTGAN/TabDDPM training)
pytest tests/ -v -m "not slow"
```

### Run with detailed output
```bash
pytest tests/ -v --tb=short  # Short traceback
pytest tests/ -vv --tb=long  # Verbose with full traceback
```

## Test Organization

### test_data_loading.py
- **Purpose**: Validate dataset loading and schema correctness
- **Tests**:
  - OULAD dataset loads successfully
  - ASSISTments dataset aggregates to student-level
  - Required columns present
  - Target variables in valid ranges
  - No missing values in critical columns
  - Schema dictionary structure validation
- **Speed**: Fast (< 5 seconds)

### test_synthesizers.py
- **Purpose**: Smoke tests for synthesizer functionality
- **Tests**:
  - GaussianCopulaSynth initialization, fit, sample
  - CTGANSynth initialization, fit, sample (marked slow)
  - TabDDPMSynth initialization, fit, sample (marked slow)
  - sampling_patience parameter fallback
  - Data type preservation
- **Speed**: 
  - Fast tests: < 10 seconds
  - Slow tests: 1-5 minutes (short epochs for testing)

### test_pipeline.py
- **Purpose**: Integration tests for full pipeline
- **Tests**:
  - End-to-end minimal pipeline (OULAD + Gaussian Copula)
  - C2ST privacy evaluation
  - Results persistence (JSON, Parquet)
  - Visualization generation
  - Error handling (invalid datasets, empty data)
- **Speed**: 
  - Fast tests: < 10 seconds
  - Slow tests: 2-10 minutes

### conftest.py
- **Purpose**: Shared pytest configuration and fixtures
- **Contents**:
  - Custom marker definitions (slow)
  - Shared fixtures (test_data_path)

## Test Coverage

### Data Loading
- ✓ OULAD: 32,593 rows, 27 columns
- ✓ ASSISTments: Student-level aggregation
- ✓ Schema validation
- ✓ Feature engineering

### Synthesizers
- ✓ Gaussian Copula (baseline)
- ✓ CTGAN (deep learning)
- ✓ TabDDPM (diffusion model)

### Evaluations
- ✓ SDMetrics quality scores
- ✓ C2ST privacy (effective AUC)
- ✓ Results JSON structure
- ✓ Data Parquet storage

### Error Handling
- ✓ Invalid dataset names
- ✓ Missing data directories
- ✓ Empty dataframes

## Expected Test Results

All tests should pass with the following structure:

```
tests/test_data_loading.py::TestOULADDataLoading::test_oulad_loads_successfully PASSED
tests/test_data_loading.py::TestOULADDataLoading::test_oulad_expected_shape PASSED
tests/test_data_loading.py::TestOULADDataLoading::test_oulad_required_columns PASSED
tests/test_data_loading.py::TestOULADDataLoading::test_oulad_target_columns PASSED
tests/test_data_loading.py::TestOULADDataLoading::test_oulad_no_nulls_in_key_columns PASSED
tests/test_data_loading.py::TestOULADDataLoading::test_oulad_schema_structure PASSED
tests/test_data_loading.py::TestASSISTmentsDataLoading::test_assistments_loads_successfully PASSED
tests/test_data_loading.py::TestASSISTmentsDataLoading::test_assistments_aggregated_to_student_level PASSED
tests/test_data_loading.py::TestASSISTmentsDataLoading::test_assistments_required_columns PASSED
tests/test_data_loading.py::TestASSISTmentsDataLoading::test_assistments_target_ranges PASSED
tests/test_data_loading.py::TestASSISTmentsDataLoading::test_assistments_no_negative_counts PASSED
tests/test_data_loading.py::TestASSISTmentsDataLoading::test_assistments_schema_structure PASSED
tests/test_data_loading.py::TestCrossDatasetConsistency::test_both_datasets_have_targets PASSED
tests/test_data_loading.py::TestCrossDatasetConsistency::test_both_datasets_have_id_cols PASSED
tests/test_data_loading.py::TestCrossDatasetConsistency::test_schema_keys_consistent PASSED

======================== 15 passed in 45.23s ========================
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest
      - run: pytest tests/ -v -m "not slow"
```

## Adding New Tests

### Template for new test file:

```python
"""
Description of test module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from synthla_edu_v2 import ...

class TestNewFeature:
    """Test suite for new feature."""
    
    def test_feature_works(self):
        """Test that feature works correctly."""
        # Arrange
        ...
        
        # Act
        ...
        
        # Assert
        assert ...
```

## Troubleshooting

### Tests fail with "ModuleNotFoundError"
- Ensure `synthla_edu_v2.py` is in repository root
- Check `sys.path.insert()` is at top of test file

### Tests fail with "FileNotFoundError" for data
- Ensure `data/raw/oulad/` and `data/raw/assistments/` exist
- Check CSV files are present

### CTGAN/TabDDPM tests timeout
- These are marked as `@pytest.mark.slow`
- Run with `-m "not slow"` to skip
- Or increase timeout: `pytest --timeout=600`

### Memory errors on TabDDPM tests
- Reduce sample size in test
- Run tests sequentially: `pytest -n 1`

## Coverage Report

To generate coverage report:

```bash
pip install pytest-cov
pytest tests/ --cov=synthla_edu_v2 --cov-report=html
```

Open `htmlcov/index.html` to view detailed coverage.
