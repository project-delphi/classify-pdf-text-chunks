import os
import json
import shutil
import tempfile
import logging
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from fastapi.testclient import TestClient
from src.app import (
    extract_text,
    text_splitter,
    initialize_vector_store,
    save_vector_store,
    load_vector_store,
    predict_class,
    VectorStoreConfig,
    vector_store,
    metadata,
    app,
)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = TestClient(app)


@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing."""
    from reportlab.pdfgen import canvas

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        c = canvas.Canvas(tmp.name)
        c.drawString(100, 750, "This is a test PDF document.")
        c.drawString(100, 700, "It contains multiple lines of text.")
        c.drawString(100, 650, "This will be used for testing.")
        c.save()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def vector_store_config():
    """Mock config for vector store."""
    config = VectorStoreConfig(
        vector_store_path="data/test_vector_store",
        metadata_path="data/test_metadata.json",
        similarity_search_k=2,
        chunk_size=100,
        chunk_overlap=20,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    os.makedirs(os.path.dirname(config.vector_store_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.metadata_path), exist_ok=True)
    return config


@pytest.fixture(autouse=True)
def clean_vector_store(vector_store_config):
    """Clean up vector store before and after each test."""
    # Clean up before test
    if os.path.exists(vector_store_config.vector_store_path):
        shutil.rmtree(vector_store_config.vector_store_path)
    if os.path.exists(vector_store_config.metadata_path):
        os.remove(vector_store_config.metadata_path)

    # Initialize vector store
    initialize_vector_store()

    yield

    # Clean up after test
    if os.path.exists(vector_store_config.vector_store_path):
        shutil.rmtree(vector_store_config.vector_store_path)
    if os.path.exists(vector_store_config.metadata_path):
        os.remove(vector_store_config.metadata_path)


@pytest.fixture
def mock_vector_store():
    """Mock vector store operations."""
    with patch("src.app.vector_store") as mock_vs, patch(
        "src.app.metadata"
    ) as mock_metadata:
        mock_vs.similarity_search.return_value = [
            MagicMock(page_content="test content", metadata={"doc_id": "test1"})
        ]
        mock_metadata.get.return_value = {"label": "good"}
        yield mock_vs


@pytest.fixture
def mock_openai():
    """Mock OpenAI API responses."""
    with patch("openai.OpenAI") as mock:
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [
            type(
                "Choice", (), {"message": type("Message", (), {"content": "good"})()}
            )()
        ]
        mock_client.chat.completions.create.return_value = mock_completion
        mock.return_value = mock_client
        yield mock


def test_extract_text(sample_pdf):
    """Test text extraction from PDF."""
    text = extract_text(sample_pdf)
    assert "test PDF document" in text
    assert "multiple lines" in text


def test_text_splitter(sample_pdf):
    """Test text splitting functionality."""
    text = extract_text(sample_pdf)
    chunks = text_splitter.split_text(text)
    assert len(chunks) > 0
    assert all(len(chunk) <= text_splitter._chunk_size for chunk in chunks)


def test_vector_store_operations(vector_store_config):
    """Test vector store operations."""
    test_text = "This is a test document for the vector store."
    test_metadata = {"doc_id": "test1", "label": "good"}
    embeddings = HuggingFaceEmbeddings(model_name=vector_store_config.embedding_model)
    vs = FAISS.from_texts([test_text], embeddings, metadatas=[test_metadata])
    vs.save_local(vector_store_config.vector_store_path)
    with open(vector_store_config.metadata_path, "w") as f:
        json.dump({"test1": {"label": "good"}}, f)
    loaded_vs = FAISS.load_local(vector_store_config.vector_store_path, embeddings)
    results = loaded_vs.similarity_search("test document", k=1)
    assert "test document" in results[0].page_content
    with open(vector_store_config.metadata_path) as f:
        metadata = json.load(f)
    assert metadata["test1"]["label"] == "good"


def test_train_endpoint(sample_pdf, vector_store_config):
    """Test the train endpoint."""
    try:
        with open(sample_pdf, "rb") as f1, open(sample_pdf, "rb") as f2:
            response = client.post(
                "/train",
                data={"labels": "good,bad"},
                files=[
                    ("pdfs", ("test1.pdf", f1, "application/pdf")),
                    ("pdfs", ("test2.pdf", f2, "application/pdf")),
                ],
            )
            logger.debug(
                f"Train response: {response.json() if response.status_code == 200 else response.content}"
            )
        assert (
            response.status_code == 200
        ), f"Train failed with status {response.status_code}: {response.content}"
        assert (
            response.json()["status"] == "model trained"
        ), f"Unexpected response: {response.json()}"
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
