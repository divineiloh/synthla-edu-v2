"""
Test data loading and schema validation for SYNTHLA-EDU V2 datasets.

Tests the build_dataset() function for OULAD and ASSISTments datasets,
verifying data quality, schema correctness, and feature engineering.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from synthla_edu_v2 import build_dataset


@pytest.mark.requires_data
class TestOULADDataLoading:
    """Test suite for OULAD dataset loading and validation."""
    
    @pytest.fixture(scope="class")
    def oulad_data(self):
        """Load OULAD dataset once for all tests."""
        if not (Path("data/raw/oulad/studentInfo.csv").exists() or Path("data/raw/studentInfo.csv").exists()):
             pytest.skip("OULAD data not found")
        df, schema = build_dataset("oulad", "data/raw")
        return df, schema
    
    def test_oulad_loads_successfully(self, oulad_data):
        """Verify OULAD dataset loads without errors."""
        df, schema = oulad_data
        assert df is not None
        assert schema is not None
        assert len(df) > 0, "Dataset should not be empty"
    
    def test_oulad_expected_shape(self, oulad_data):
        """Check OULAD has expected number of rows and columns."""
        df, schema = oulad_data
        assert len(df) > 30000, f"Expected >30k rows, got {len(df)}"
        assert len(df.columns) >= 25, f"Expected >=25 columns, got {len(df.columns)}"
    
    def test_oulad_required_columns(self, oulad_data):
        """Verify OULAD contains all required columns."""
        df, schema = oulad_data
        required_cols = [
            "id_student", "code_module", "code_presentation",
            "gender", "region", "highest_education", "age_band",
            "dropout"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_oulad_target_columns(self, oulad_data):
        """Verify target columns are properly formatted."""
        df, schema = oulad_data
        
        # Check dropout binary target (pandas uses float64 for nullable handling)
        assert "dropout" in df.columns
        assert df["dropout"].dtype in [np.int64, np.int32, np.float64]
        assert set(df["dropout"].dropna().unique()) <= {0.0, 1.0}, "Dropout should be binary 0/1"
    
    def test_oulad_no_nulls_in_key_columns(self, oulad_data):
        """Check critical columns have no missing values."""
        df, schema = oulad_data
        critical_cols = ["id_student", "gender", "dropout"]
        for col in critical_cols:
            assert df[col].notna().all(), f"Column {col} has missing values"
    
    def test_oulad_schema_structure(self, oulad_data):
        """Validate schema dictionary structure."""
        df, schema = oulad_data
        
        required_schema_keys = ["id_cols", "target_cols", "categorical_cols", "group_col"]
        for key in required_schema_keys:
            assert key in schema, f"Schema missing key: {key}"
        
        # Check id_cols
        assert "id_student" in schema["id_cols"]
        
        # Check target_cols
        assert "dropout" in schema["target_cols"]
        
        # Check group_col for GroupShuffleSplit
        assert schema["group_col"] == "id_student"
        
        # Check categorical separation
        assert len(schema["categorical_cols"]) > 0


@pytest.mark.requires_data
class TestASSISTmentsDataLoading:
    """Test suite for ASSISTments dataset loading and validation."""
    
    @pytest.fixture(scope="class")
    def assistments_data(self):
        """Load ASSISTments dataset once for all tests."""
        if not (Path("data/raw/assistments/assistments_2009_2010.csv").exists() or Path("data/raw/assistments_2009_2010.csv").exists()):
             pytest.skip("ASSISTments data not found")
        df, schema = build_dataset("assistments", "data/raw")
        return df, schema
    
    def test_assistments_loads_successfully(self, assistments_data):
        """Verify ASSISTments dataset loads without errors."""
        df, schema = assistments_data
        assert df is not None
        assert schema is not None
        assert len(df) > 0, "Dataset should not be empty"
    
    def test_assistments_aggregated_to_student_level(self, assistments_data):
        """Verify data is aggregated to student-level (no duplicate user_ids)."""
        df, schema = assistments_data
        assert "user_id" in df.columns
        assert df["user_id"].is_unique, "user_id should be unique (student-level data)"
    
    def test_assistments_required_columns(self, assistments_data):
        """Verify ASSISTments contains required aggregated columns."""
        df, schema = assistments_data
        required_cols = [
            "user_id",
            "n_interactions",
            "student_pct_correct",
            "high_accuracy"  # Binary target added
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_assistments_target_ranges(self, assistments_data):
        """Verify target variables are in valid ranges."""
        df, schema = assistments_data
        
        # Check student_pct_correct is 0-1 (fraction, not percentage)
        if "student_pct_correct" in df.columns:
            assert df["student_pct_correct"].min() >= 0
            assert df["student_pct_correct"].max() <= 1.0, "student_pct_correct should be 0-1 fraction"
    
    def test_assistments_no_negative_counts(self, assistments_data):
        """Check count features have no negative values."""
        df, schema = assistments_data
        count_cols = [col for col in df.columns if "count" in col.lower() or "total" in col.lower()]
        for col in count_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                assert (df[col] >= 0).all(), f"Column {col} has negative values"
    
    def test_assistments_schema_structure(self, assistments_data):
        """Validate schema dictionary structure."""
        df, schema = assistments_data
        
        required_schema_keys = ["id_cols", "target_cols", "categorical_cols"]
        for key in required_schema_keys:
            assert key in schema, f"Schema missing key: {key}"
        
        # Check user_id in id_cols
        assert "user_id" in schema["id_cols"]
        
        # Check both targets present
        assert "high_accuracy" in schema["target_cols"]
        assert "student_pct_correct" in schema["target_cols"]


@pytest.mark.requires_data
class TestCrossDatasetConsistency:
    """Test consistency between OULAD and ASSISTments datasets."""
    
    @pytest.fixture(scope="class")
    def both_datasets(self):
        """Load both datasets."""
        oulad_exists = (Path("data/raw/oulad/studentInfo.csv").exists() or Path("data/raw/studentInfo.csv").exists())
        assist_exists = (Path("data/raw/assistments/assistments_2009_2010.csv").exists() or Path("data/raw/assistments_2009_2010.csv").exists())
        
        if not (oulad_exists and assist_exists):
             pytest.skip("One or both datasets not found")

        oulad_df, oulad_schema = build_dataset("oulad", "data/raw")
        assist_df, assist_schema = build_dataset("assistments", "data/raw")
        return (oulad_df, oulad_schema), (assist_df, assist_schema)
    
    def test_both_datasets_have_targets(self, both_datasets):
        """Verify both datasets have target columns defined."""
        (oulad_df, oulad_schema), (assist_df, assist_schema) = both_datasets
        
        assert len(oulad_schema["target_cols"]) > 0
        assert len(assist_schema["target_cols"]) > 0
    
    def test_both_datasets_have_id_cols(self, both_datasets):
        """Verify both datasets have id columns defined."""
        (oulad_df, oulad_schema), (assist_df, assist_schema) = both_datasets
        
        assert len(oulad_schema["id_cols"]) > 0
        assert len(assist_schema["id_cols"]) > 0
    
    def test_schema_keys_consistent(self, both_datasets):
        """Verify schema dictionaries have consistent structure."""
        (oulad_df, oulad_schema), (assist_df, assist_schema) = both_datasets
        
        common_keys = {"id_cols", "target_cols"}
        assert common_keys.issubset(oulad_schema.keys())
        assert common_keys.issubset(assist_schema.keys())


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
