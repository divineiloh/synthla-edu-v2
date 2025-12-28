"""
pytest configuration and shared fixtures for SYNTHLA-EDU V2 tests.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require local data files"
    )


# Global test configuration
@pytest.fixture(scope="session")
def test_data_path():
    """Return path to test data directory."""
    return "data/raw"


from pathlib import Path

@pytest.fixture(autouse=True)
def skip_if_data_missing(request):
    """Skip tests marked with 'requires_data' if data is missing."""
    if request.node.get_closest_marker("requires_data"):
        # Check for OULAD or ASSISTments data
        oulad_exists = (Path("data/raw/oulad/studentInfo.csv").exists() or 
                       Path("data/raw/studentInfo.csv").exists())
        assist_exists = (Path("data/raw/assistments/assistments_2009_2010.csv").exists() or 
                        Path("data/raw/assistments_2009_2010.csv").exists())
        
        if not (oulad_exists or assist_exists):
            pytest.skip("Test data not found in data/raw")

