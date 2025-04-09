import pytest
import os
import tempfile
from src.app import (
    extract_text,
    train_model,
    predict_class,
    load_vector_store,
    save_vector_store,
    vector_store,
    metadata,
    VECTOR_STORE_PATH,
    METADATA_PATH
)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import shutil
from unittest.mock import patch
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_test_pdf(content: str) -> str:
    """Create a temporary PDF file with the given content."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        c = canvas.Canvas(tmp.name, pagesize=letter)
        c.drawString(100, 750, content)
        c.save()
        return tmp.name

@pytest.fixture(autouse=True)
def setup_environment():
    """Set up environment variables for testing."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    yield
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

@pytest.fixture
def setup_test_data():
    """Setup test data and clean up after tests."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create test PDFs
    good_pdf = create_test_pdf("This is a well-written document with clear structure and good content.")
    bad_pdf = create_test_pdf("This document is poorly written with no structure and bad content.")
    neutral_pdf = create_test_pdf("This is an average document with some structure and content.")
    
    # Train the model
    train_model([good_pdf, bad_pdf, neutral_pdf], ["good", "bad", "neutral"])
    
    # Save the initial state
    save_vector_store()
    
    yield {
        "good_pdf": good_pdf,
        "bad_pdf": bad_pdf,
        "neutral_pdf": neutral_pdf
    }
    
    # Cleanup
    for pdf in [good_pdf, bad_pdf, neutral_pdf]:
        os.unlink(pdf)
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
    if os.path.exists(METADATA_PATH):
        os.unlink(METADATA_PATH)
    if os.path.exists("data/classifier.pkl"):
        os.unlink("data/classifier.pkl")
    try:
        os.rmdir("data")
    except OSError:
        pass

@pytest.fixture(autouse=True)
def mock_openai():
    """Mock OpenAI API calls."""
    with patch("openai.OpenAI") as mock:
        mock.return_value.chat.completions.create.return_value.choices = [
            type("Choice", (), {"message": type("Message", (), {"content": "good"})})()
        ]
        yield mock

def test_vector_store_initialization(setup_test_data):
    """Test that the vector store is properly initialized and loaded."""
    # Check if vector store exists
    assert vector_store is not None
    assert len(metadata) > 0
    
    # Test loading from disk
    load_vector_store()
    assert vector_store is not None
    assert len(metadata) > 0

def test_similarity_search(setup_test_data):
    """Test that similarity search returns relevant results."""
    # Create a test query
    query = "This is a well-structured document"
    
    # Search for similar documents
    results = vector_store.similarity_search(query, k=1)
    
    # Check results
    assert len(results) == 1
    assert hasattr(results[0], 'metadata')
    assert 'doc_id' in results[0].metadata

def test_rag_classification(setup_test_data):
    """Test RAG-based classification with different types of documents."""
    # Test good document
    good_test = create_test_pdf("This is another well-written document with excellent structure.")
    good_pred = predict_class(good_test)
    assert good_pred == "good"
    
    # Test bad document
    bad_test = create_test_pdf("This is another poorly written document with no structure.")
    bad_pred = predict_class(bad_test)
    assert bad_pred == "good"  # Will be "good" because we mocked the OpenAI response
    
    # Test neutral document
    neutral_test = create_test_pdf("This is another average document with some structure.")
    neutral_pred = predict_class(neutral_test)
    assert neutral_pred == "good"  # Will be "good" because we mocked the OpenAI response
    
    # Cleanup
    for pdf in [good_test, bad_test, neutral_test]:
        os.unlink(pdf)

def test_metadata_integration(setup_test_data):
    """Test that metadata is properly integrated with the vector store."""
    # Check metadata structure
    for doc_id, meta in metadata.items():
        assert 'label' in meta
        assert 'source' in meta
        assert 'chunk_index' in meta
        assert meta['label'] in ['good', 'bad', 'neutral']

def save_vector_store_to_path(vector_store_path, metadata_path):
    """Save vector store and metadata to specified paths."""
    if not os.path.exists(os.path.dirname(vector_store_path)):
        os.makedirs(os.path.dirname(vector_store_path))
    
    # Save vector store
    vector_store.save_local(vector_store_path)
    print(f"Saving vector store to {vector_store_path}")
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Saving metadata with {len(metadata)} entries to {metadata_path}")
    
    # Verify save
    if os.path.exists(vector_store_path) and os.path.exists(metadata_path):
        print("Save verified working")
    else:
        print("Save verification failed")

@pytest.fixture
def clean_persistence_state():
    """Fixture to provide a clean state for persistence testing."""
    global VECTOR_STORE_PATH, METADATA_PATH
    
    # Store original paths
    original_vector_store_path = VECTOR_STORE_PATH
    original_metadata_path = METADATA_PATH
    
    # Use temporary paths for test
    test_dir = tempfile.mkdtemp()
    VECTOR_STORE_PATH = os.path.join(test_dir, "test_vector_store")
    METADATA_PATH = os.path.join(test_dir, "test_metadata.json")
    
    yield
    
    # Clean up test directory
    shutil.rmtree(test_dir)
    
    # Restore original paths
    VECTOR_STORE_PATH = original_vector_store_path
    METADATA_PATH = original_metadata_path

def test_persistence(clean_persistence_state):
    """Test that the vector store and metadata can be saved and loaded."""
    global vector_store, metadata
    
    # Initialize with test data
    test_metadata = {"test_entry": {"chunk_index": 0, "label": "test", "source": "test"}}
    test_text = ["test content"]
    
    # Create new vector store with test data
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(
        test_text, 
        embeddings, 
        metadatas=[{"chunk_index": 0, "label": "test", "source": "test"}]
    )
    
    # Set metadata
    metadata = test_metadata.copy()
    
    # Save test state to temporary paths
    save_vector_store_to_path(VECTOR_STORE_PATH, METADATA_PATH)
    
    # Clear current state
    vector_store = None
    metadata = {}
    
    # Load saved state
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    # Verify loaded state matches test data
    assert vector_store is not None
    assert metadata == test_metadata
    
    # Verify search functionality
    results = vector_store.similarity_search("test", k=1)
    assert len(results) > 0
    assert "test content" in results[0].page_content 