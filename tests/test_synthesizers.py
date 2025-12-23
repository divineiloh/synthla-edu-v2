"""
Smoke tests for SYNTHLA-EDU V2 synthesizers.

Tests basic functionality of GaussianCopulaSynth, CTGANSynth, and TabDDPMSynth
to ensure they can fit and sample synthetic data without errors.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from synthla_edu_v2 import (
    build_dataset,
    GaussianCopulaSynth,
    CTGANSynth,
    TabDDPMSynth
)


@pytest.fixture(scope="module")
def small_oulad_sample():
    """Load a small sample of OULAD for fast testing."""
    df, schema = build_dataset("oulad", "data/raw")
    # Use small sample for speed
    sample = df.sample(n=min(500, len(df)), random_state=42)
    return sample, schema


class TestGaussianCopulaSynth:
    """Test GaussianCopulaSynth synthesizer."""
    
    def test_gaussian_copula_initialization(self, small_oulad_sample):
        """Test synthesizer initializes without errors."""
        df, schema = small_oulad_sample
        synth = GaussianCopulaSynth(schema)
        assert synth is not None
    
    def test_gaussian_copula_fit(self, small_oulad_sample):
        """Test synthesizer can fit data."""
        df, schema = small_oulad_sample
        synth = GaussianCopulaSynth(schema)
        synth.fit(df)
        # If no exception raised, test passes
    
    def test_gaussian_copula_sample(self, small_oulad_sample):
        """Test synthesizer can generate samples."""
        df, schema = small_oulad_sample
        synth = GaussianCopulaSynth(schema)
        synth.fit(df)
        
        synthetic = synth.sample(n=100)
        
        assert synthetic is not None
        assert len(synthetic) == 100
        assert set(synthetic.columns) == set(df.columns)
    
    def test_gaussian_copula_preserves_dtypes(self, small_oulad_sample):
        """Test synthesizer preserves column data types."""
        df, schema = small_oulad_sample
        synth = GaussianCopulaSynth(schema)
        synth.fit(df)
        
        synthetic = synth.sample(n=50)
        
        # Check categorical columns remain categorical
        for col in schema["categorical_cols"]:
            if col in synthetic.columns:
                # Should be object or category dtype
                assert synthetic[col].dtype in [object, 'category']


class TestCTGANSynth:
    """Test CTGANSynth deep learning synthesizer."""
    
    def test_ctgan_initialization(self, small_oulad_sample):
        """Test CTGAN initializes without errors."""
        df, schema = small_oulad_sample
        synth = CTGANSynth(schema, epochs=10)  # Very short for testing
        assert synth is not None
        assert synth.epochs == 10
    
    def test_ctgan_fit(self, small_oulad_sample):
        """Test CTGAN can fit data (short training)."""
        df, schema = small_oulad_sample
        synth = CTGANSynth(schema, epochs=10, batch_size=64)
        synth.fit(df)
        # If no exception raised, test passes
    
    @pytest.mark.slow
    def test_ctgan_sample(self, small_oulad_sample):
        """Test CTGAN can generate samples."""
        df, schema = small_oulad_sample
        synth = CTGANSynth(schema, epochs=10)
        synth.fit(df)
        
        synthetic = synth.sample(n=100)
        
        assert synthetic is not None
        assert len(synthetic) == 100
        assert set(synthetic.columns) == set(df.columns)


class TestTabDDPMSynth:
    """Test TabDDPMSynth diffusion model synthesizer."""
    
    def test_tabddpm_initialization(self, small_oulad_sample):
        """Test TabDDPM initializes without errors."""
        df, schema = small_oulad_sample
        synth = TabDDPMSynth(schema, num_timesteps=100)  # Very short for testing
        assert synth is not None
    
    @pytest.mark.slow
    def test_tabddpm_fit(self, small_oulad_sample):
        """Test TabDDPM can fit data (short training)."""
        df, schema = small_oulad_sample
        synth = TabDDPMSynth(schema, num_timesteps=100)
        synth.fit(df)
        # If no exception raised, test passes
    
    @pytest.mark.slow
    def test_tabddpm_sample(self, small_oulad_sample):
        """Test TabDDPM can generate samples."""
        df, schema = small_oulad_sample
        synth = TabDDPMSynth(schema, num_timesteps=100)
        synth.fit(df)
        
        synthetic = synth.sample(n=50)  # Smaller sample for speed
        
        assert synthetic is not None
        assert len(synthetic) > 0  # TabDDPM may return slightly fewer samples
        assert set(synthetic.columns) == set(df.columns)
    
    def test_tabddpm_sampling_patience_fallback(self, small_oulad_sample):
        """Test TabDDPM handles sampling_patience parameter correctly."""
        df, schema = small_oulad_sample
        synth = TabDDPMSynth(schema, num_timesteps=100)
        synth.fit(df)
        
        # This should not raise an error even if sampling_patience is unsupported
        try:
            synthetic = synth.sample(n=50)
            assert synthetic is not None
        except TypeError as e:
            # If error still occurs, ensure it's not related to sampling_patience
            assert "sampling_patience" not in str(e).lower()


class TestSynthesizerConsistency:
    """Test consistency across all synthesizers."""
    
    @pytest.mark.parametrize("synth_class,kwargs", [
        (GaussianCopulaSynth, {}),
        (CTGANSynth, {"epochs": 10}),
        (TabDDPMSynth, {"num_timesteps": 100}),
    ])
    def test_all_synthesizers_return_dataframes(self, small_oulad_sample, synth_class, kwargs):
        """Test all synthesizers return pandas DataFrames."""
        df, schema = small_oulad_sample
        synth = synth_class(schema, **kwargs)
        synth.fit(df)
        synthetic = synth.sample(n=50)
        
        assert isinstance(synthetic, pd.DataFrame)
    
    @pytest.mark.parametrize("synth_class,kwargs", [
        (GaussianCopulaSynth, {}),
    ])  # Only test fast synthesizer for full column check
    def test_synthesizers_preserve_all_columns(self, small_oulad_sample, synth_class, kwargs):
        """Test synthesizers preserve all columns from original data."""
        df, schema = small_oulad_sample
        synth = synth_class(schema, **kwargs)
        synth.fit(df)
        synthetic = synth.sample(n=50)
        
        # Remove id columns for comparison (may be regenerated)
        real_cols = set(df.columns) - set(schema.get("id_cols", []))
        synth_cols = set(synthetic.columns) - set(schema.get("id_cols", []))
        
        assert synth_cols >= real_cols, f"Missing columns: {real_cols - synth_cols}"


if __name__ == "__main__":
    # Run tests with verbose output
    # Use -m "not slow" to skip slow tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
