import pytest
import os
from src.app import app
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    test_dir = "data/test"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # Cleanup will be handled by the cleanup fixture in test_app.py 