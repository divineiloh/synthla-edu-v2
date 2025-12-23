"""
pytest configuration and shared fixtures for SYNTHLA-EDU V2 tests.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Global test configuration
@pytest.fixture(scope="session")
def test_data_path():
    """Return path to test data directory."""
    return "data/raw"
