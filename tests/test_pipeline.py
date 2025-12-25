"""
Integration tests for SYNTHLA-EDU V2 pipeline.

Tests the full end-to-end pipeline including data loading, synthesis,
evaluation, and visualization generation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import json
import pandas as pd
from pathlib import Path
from synthla_edu_v2 import (
    build_dataset,
    GaussianCopulaSynth,
    sdmetrics_quality,
    c2st_effective_auc,
    create_cross_dataset_visualizations
)


def _make_minimal_cross_dataset_viz_inputs() -> tuple[dict, dict, dict]:
    """Create minimal, valid inputs for create_cross_dataset_visualizations.

    The plotting function expects a fairly rich nested results structure.
    This helper provides just enough shape/keys to generate figures quickly.
    """

    def synth_result(*, overall_score: float, eff_auc: float, mia_eff_auc: float, rf_auc: float, ridge_mae: float) -> dict:
        return {
            "sdmetrics": {"overall_score": overall_score},
            "c2st": {"effective_auc": eff_auc},
            "mia": {
                "worst_case_effective_auc": mia_eff_auc,
                "attackers": {
                    "logistic_regression": {"effective_auc": mia_eff_auc},
                    "random_forest": {"effective_auc": min(1.0, mia_eff_auc + 0.02)},
                },
            },
            "utility": {
                "classification": {
                    "trtr_rf_auc": 0.80,
                    "rf_auc": rf_auc,
                    "rf_auc_ci": {"ci_low": max(0.5, rf_auc - 0.02), "ci_high": min(1.0, rf_auc + 0.02)},
                },
                "regression": {
                    "trtr_ridge_mae": 12.0,
                    "ridge_mae": ridge_mae,
                    "ridge_mae_ci": {"ci_low": max(0.0, ridge_mae - 1.0), "ci_high": ridge_mae + 1.0},
                },
            },
            "timing": {"fit_seconds": 1.0, "sample_seconds": 1.0},
        }

    all_results = {
        "oulad": {
            "dataset": "oulad",
            "synthesizers": {
                "gaussian_copula": synth_result(overall_score=0.82, eff_auc=0.62, mia_eff_auc=0.53, rf_auc=0.78, ridge_mae=11.0),
                "ctgan": synth_result(overall_score=0.78, eff_auc=0.66, mia_eff_auc=0.55, rf_auc=0.76, ridge_mae=11.5),
                "tabddpm": synth_result(overall_score=0.80, eff_auc=0.64, mia_eff_auc=0.54, rf_auc=0.77, ridge_mae=11.2),
            },
        },
        "assistments": {
            "dataset": "assistments",
            "synthesizers": {
                "gaussian_copula": synth_result(overall_score=0.81, eff_auc=0.63, mia_eff_auc=0.52, rf_auc=0.74, ridge_mae=10.5),
                "ctgan": synth_result(overall_score=0.77, eff_auc=0.68, mia_eff_auc=0.56, rf_auc=0.73, ridge_mae=10.9),
                "tabddpm": synth_result(overall_score=0.79, eff_auc=0.65, mia_eff_auc=0.55, rf_auc=0.735, ridge_mae=10.7),
            },
        },
    }

    base_train = pd.DataFrame(
        {
            "num_feature_1": list(range(100)),
            "num_feature_2": [x * 2 for x in range(100)],
            "cat_feature": pd.Categorical(["a", "b"] * 50),
        }
    )

    all_train_data = {"oulad": base_train.copy(), "assistments": base_train.copy()}

    def synth_df(seed: int) -> pd.DataFrame:
        # Keep the same numeric columns so correlation/distribution plots can run.
        df = base_train.copy()
        df["num_feature_1"] = (df["num_feature_1"] + seed) % 100
        df["num_feature_2"] = (df["num_feature_2"] + 3 * seed) % 200
        return df

    all_synthetic_data = {
        "oulad": {
            "gaussian_copula": synth_df(1),
            "ctgan": synth_df(2),
            "tabddpm": synth_df(3),
        },
        "assistments": {
            "gaussian_copula": synth_df(4),
            "ctgan": synth_df(5),
            "tabddpm": synth_df(6),
        },
    }

    return all_results, all_train_data, all_synthetic_data


@pytest.fixture(scope="module")
def temp_output_dir(tmp_path_factory):
    """Create temporary output directory for test runs."""
    return tmp_path_factory.mktemp("test_output")


class TestEndToEndPipeline:
    """Test full pipeline execution."""
    
    @pytest.mark.slow
    def test_minimal_pipeline_oulad(self, temp_output_dir):
        """Test minimal pipeline on OULAD with Gaussian Copula."""
        # Load data
        df, schema = build_dataset("oulad", "data/raw")
        df_small = df.sample(n=500, random_state=42)
        
        # Train synthesizer
        synth = GaussianCopulaSynth()
        synth.fit(df_small)
        
        # Generate synthetic data
        df_synth = synth.sample(n=500)
        
        # Run evaluation
        quality_results = sdmetrics_quality(df_small, df_synth)
        
        # Verify results (function now returns overall_score + optional details)
        assert quality_results is not None
        assert "overall_score" in quality_results
        assert 0.0 <= quality_results["overall_score"] <= 1.0
    
    @pytest.mark.slow
    def test_c2st_evaluation(self, temp_output_dir):
        """Test C2ST privacy evaluation."""
        # Load small sample
        df, schema = build_dataset("oulad", "data/raw")
        df_small = df.sample(n=500, random_state=42)
        
        # Split train/test
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df_small, test_size=0.3, random_state=42)
        
        # Generate synthetic
        synth = GaussianCopulaSynth()
        synth.fit(train)
        df_synth = synth.sample(n=len(train))
        
        # Run C2ST (signature is: real_test, synthetic_train)
        c2st_results = c2st_effective_auc(test, df_synth, test_size=0.3, seed=42)
        
        # Verify results structure
        assert "effective_auc" in c2st_results
        assert "n_per_class" in c2st_results
        
        # Check values are reasonable (effective_auc is max(auc, 1-auc) so always >= 0.5)
        assert 0.5 <= c2st_results["effective_auc"] <= 1.0


class TestResultsPersistence:
    """Test saving and loading results."""
    
    def test_results_json_structure(self, temp_output_dir):
        """Test results.json has expected structure."""
        # Create mock results
        results = {
            "dataset": "oulad",
            "synthesizers": {
                "gaussian_copula": {
                    "sdmetrics": {"Column Shapes": 0.85},
                    "c2st": {"effective_auc_train": 0.15},
                    "mia": {"worst_case_auc": 0.52},
                    "tstr": {"r2_synthetic": 0.45}
                }
            }
        }
        
        # Save to temp dir
        output_file = temp_output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Load and verify
        with open(output_file, "r") as f:
            loaded = json.load(f)
        
        assert loaded["dataset"] == "oulad"
        assert "gaussian_copula" in loaded["synthesizers"]
        assert "sdmetrics" in loaded["synthesizers"]["gaussian_copula"]
    
    def test_data_parquet_persistence(self, temp_output_dir):
        """Test data.parquet saves and loads correctly."""
        # Create mock data
        df = pd.DataFrame({
            "id": range(100),
            "source": ["real"] * 50 + ["gaussian_copula"] * 50,
            "feature1": range(100),
            "feature2": [x * 2 for x in range(100)]
        })
        
        # Save to temp dir
        output_file = temp_output_dir / "data.parquet"
        df.to_parquet(output_file, index=False)
        
        # Load and verify
        loaded = pd.read_parquet(output_file)
        assert len(loaded) == 100
        assert "source" in loaded.columns
        assert set(loaded["source"].unique()) == {"real", "gaussian_copula"}


class TestVisualizationGeneration:
    """Test visualization creation."""
    
    @pytest.mark.slow
    def test_cross_dataset_visualizations(self, temp_output_dir):
        """Test cross-dataset visualization generation."""
        import matplotlib
        matplotlib.use("Agg", force=True)

        figures_dir = temp_output_dir / "figures"
        all_results, all_train_data, all_synthetic_data = _make_minimal_cross_dataset_viz_inputs()

        paths = create_cross_dataset_visualizations(
            figures_dir,
            all_results,
            all_train_data,
            all_synthetic_data,
        )

        assert isinstance(paths, list)
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)
    
    def test_visualization_files_created(self, temp_output_dir):
        """Test that visualization files are created."""
        import matplotlib
        matplotlib.use("Agg", force=True)

        figures_dir = temp_output_dir / "figures_files_created"
        all_results, all_train_data, all_synthetic_data = _make_minimal_cross_dataset_viz_inputs()
        create_cross_dataset_visualizations(figures_dir, all_results, all_train_data, all_synthetic_data)

        pngs = list(figures_dir.glob("*.png"))
        assert len(pngs) >= 1


class TestErrorHandling:
    """Test error handling in pipeline."""
    
    def test_invalid_dataset_name(self):
        """Test build_dataset handles invalid dataset names."""
        with pytest.raises((ValueError, FileNotFoundError, KeyError)):
            build_dataset("invalid_dataset", "data/raw")
    
    def test_missing_data_directory(self):
        """Test build_dataset handles missing data directory."""
        with pytest.raises(FileNotFoundError):
            build_dataset("oulad", "/nonexistent/path")
    
    def test_synthesizer_with_empty_data(self):
        """Test synthesizers handle empty dataframes gracefully."""
        empty_df = pd.DataFrame()
        schema = {"categorical_cols": [], "numeric_cols": []}
        
        synth = GaussianCopulaSynth()
        with pytest.raises((ValueError, IndexError, KeyError)):
            synth.fit(empty_df)


if __name__ == "__main__":
    # Run tests with verbose output
    # Use -m "not slow" to skip slow tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
