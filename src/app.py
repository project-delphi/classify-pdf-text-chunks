# Import required libraries
from pypdf import PdfReader  # For reading PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks
from sentence_transformers import SentenceTransformer  # For generating text embeddings
from sklearn.linear_model import LogisticRegression  # For classification
import pickle  # For saving/loading the trained model
import numpy as np  # For numerical operations
from fastapi import FastAPI, UploadFile, File, Form  # For creating the API
from tempfile import NamedTemporaryFile  # For handling uploaded files
import shutil  # For file operations
from typing import List, Any, Optional, Dict
import openai  # For LLM-based classification
import os  # For file path operations
import time
from threading import Lock
import logging
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import json
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for vector store and metadata
vector_store = None
metadata = {}

class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    pass

@dataclass
class VectorStoreConfig:
    """Configuration for vector store operations."""
    similarity_search_k: int = 3  # Number of similar documents to retrieve
    chunk_size: int = 1000  # Size of text chunks
    chunk_overlap: int = 200  # Overlap between chunks
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model to use
    vector_store_path: str = "data/vector_store"  # Path to store vector data
    metadata_path: str = "data/metadata.json"  # Path to store metadata

# Initialize configuration
config = VectorStoreConfig()

# Update paths to use config
VECTOR_STORE_PATH = config.vector_store_path
METADATA_PATH = config.metadata_path
# Initialize text splitter with chunk size of 1000 characters and 200 character overlap
# This helps in breaking down large documents into manageable pieces
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap
)

# Initialize the sentence transformer model for generating embeddings
# This model converts text chunks into numerical vectors
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Define mapping between text labels and numerical values for the classifier
label_map = {'bad': 0, 'neutral': 1, 'good': 2}
# Create reverse mapping for converting predictions back to text labels
reverse_label_map = {v: k for k, v in label_map.items()}
# Path where the trained model will be saved
MODEL_PATH = os.path.join("data", "classifier.pkl")
# Rate limiting configuration
RATE_LIMIT_TOKENS = 3  # Number of tokens in the bucket
RATE_LIMIT_REFILL_RATE = 1.0  # Tokens per second
RATE_LIMIT_LAST_REFILL = time.time()
RATE_LIMIT_CURRENT_TOKENS = float(RATE_LIMIT_TOKENS)  # Use float for fractional tokens
RATE_LIMIT_LOCK = Lock()

def refill_tokens():
    """Refill the token bucket based on time elapsed."""
    global RATE_LIMIT_LAST_REFILL, RATE_LIMIT_CURRENT_TOKENS
    with RATE_LIMIT_LOCK:
        now = time.time()
        time_passed = now - RATE_LIMIT_LAST_REFILL
        new_tokens = time_passed * RATE_LIMIT_REFILL_RATE
        RATE_LIMIT_CURRENT_TOKENS = min(
            float(RATE_LIMIT_TOKENS),
            RATE_LIMIT_CURRENT_TOKENS + new_tokens
        )
        RATE_LIMIT_LAST_REFILL = now
        return RATE_LIMIT_CURRENT_TOKENS

def wait_for_token():
    """Wait until a token is available."""
    global RATE_LIMIT_CURRENT_TOKENS, RATE_LIMIT_LAST_REFILL
    while True:
        with RATE_LIMIT_LOCK:
            current_tokens = refill_tokens()
            if current_tokens >= 1.0:
                RATE_LIMIT_CURRENT_TOKENS -= 1.0
                return
        time.sleep(0.1)  # Sleep outside the lock to allow other threads to proceed

def extract_text(path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        Concatenated text from all pages of the PDF
    """
    reader = PdfReader(path)
    return "\n".join(p.extract_text() or '' for p in reader.pages)

def train_model(paths: list[str], labels: list[str]):
    """
    Train a classifier model using PDF documents and their labels, and build a vector store.
    
    Args:
        paths: List of paths to PDF files
        labels: List of labels ('bad', 'neutral', 'good') corresponding to the PDFs
    """
    global vector_store, metadata
    X, y = [], []  # Lists to store features and labels
    all_chunks = []  # List to store all text chunks
    all_metadata = []  # List to store chunk metadata
    new_metadata = {}  # Store new metadata
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    for path, label in zip(paths, labels):
        # Extract text from PDF
        text = extract_text(path)
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        # Generate embeddings for each chunk
        chunk_embeddings = embedder.encode(chunks)
        
        # Add chunks to metadata
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            doc_id = f"{os.path.basename(path)}_{i}"
            new_metadata[doc_id] = {
                "label": label,
                "source": path,
                "chunk_index": i
            }
            all_chunks.append(chunk)
            all_metadata.append({"doc_id": doc_id})
        
        # Add embeddings and corresponding labels to training data
        X.extend(chunk_embeddings)
        y.extend([label_map[label]] * len(chunks))
    
    # Create or update vector store
    if not all_chunks:
        all_chunks = [""]  # Add empty string to initialize store
        all_metadata = [{"doc_id": "empty"}]
        new_metadata = {"empty": {"label": "neutral", "source": "none", "chunk_index": 0}}
    
    vector_store = FAISS.from_texts(all_chunks, embeddings, metadatas=all_metadata)
    metadata = new_metadata  # Update metadata after successful vector store creation
    
    # Train a logistic regression classifier
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    
    # Save the trained model and vector store
    os.makedirs("data", exist_ok=True)
    pickle.dump(clf, open(MODEL_PATH, "wb"))
    save_vector_store()

def classify_chunk_with_llm(chunk: str, similar_docs: List[Any]) -> str:
    """
    Classify a text chunk using OpenAI's GPT-4 model with RAG context.
    
    Args:
        chunk: Text chunk to classify
        similar_docs: List of similar documents from vector store
        
    Returns:
        Classification label ('bad', 'neutral', 'good')
    """
    # Create a prompt with similar examples
    examples = "\n".join([
        f"Example {i+1}:\n{doc.page_content}\nLabel: {metadata[doc.metadata['doc_id']]['label']}"
        for i, doc in enumerate(similar_docs)
    ])
    
    prompt = f"""Classify this text as 'bad', 'neutral', or 'good' based on the following examples:

{examples}

Text to classify:
---
{chunk}
---
Label:"""
    # Wait for rate limit token
    wait_for_token()
    
    # Get classification from OpenAI API
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower()

def predict_class(path: str) -> str:
    """
    Predict the class of a PDF document using RAG-based classification.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        Predicted class ('bad', 'neutral', 'good')
    """
    # Extract text from PDF
    text = extract_text(path)
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    # Get predictions for each chunk using RAG
    predictions = []
    for chunk in chunks:
        # Retrieve similar chunks from vector store
        similar_docs = vector_store.similarity_search(chunk, k=config.similarity_search_k)        
        # Get labels of similar chunks
        similar_labels = [metadata[doc.metadata["doc_id"]]["label"] for doc in similar_docs]
        
        # Use LLM to classify based on the chunk and similar examples
        prediction = classify_chunk_with_llm(chunk, similar_docs)
        predictions.append(label_map[prediction])
    
    # Calculate the average prediction and round to nearest integer
    mean_pred = int(round(np.mean(predictions)))
    return reverse_label_map[mean_pred]

# Initialize FastAPI application
app = FastAPI(
    title="PDF Document Classifier",
    description="A FastAPI service for classifying PDF documents using RAG and LLMs",
    version="1.0.0"
)

@app.get("/")
async def root():
    """
    Root endpoint that provides basic information about the API.
    """
    return {
        "message": "Welcome to the PDF Document Classifier API",
        "endpoints": {
            "/train": "POST endpoint to train the classifier with PDFs and labels",
            "/predict": "POST endpoint to predict the class of a PDF",
            "/docs": "Interactive API documentation (Swagger UI)",
            "/redoc": "Alternative API documentation (ReDoc)"
        }
    }

@app.post("/train")
async def train(pdfs: List[UploadFile] = File(...), labels: str = Form(...)):
    """
    API endpoint to train the classifier with uploaded PDFs and their labels.
    
    Args:
        pdfs: List of uploaded PDF files
        labels: Comma-separated list of labels corresponding to the PDFs
        
    Returns:
        Status message indicating successful training
    """
    paths = []
    # Save uploaded PDFs to temporary files
    for pdf in pdfs:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(pdf.file, tmp)
            paths.append(tmp.name)
    # Split labels into a list
    label_list = [label.strip() for label in labels.split(',')]
    # Train the model with the uploaded PDFs
    train_model(paths, label_list)
    return {"status": "model trained"}

@app.post("/predict")
async def predict(pdf: UploadFile = File(...)):
    """
    API endpoint to predict the class of an uploaded PDF.
    
    Args:
        pdf: Uploaded PDF file
        
    Returns:
        Predicted class of the PDF
    """
    # Save uploaded PDF to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await pdf.read()  # Read the entire file
        tmp.write(content)  # Write it to the temporary file
        tmp.flush()  # Ensure it's written to disk
        # Predict the class of the PDF
        predicted = predict_class(tmp.name)
    return {"predicted_class": predicted}

def initialize_vector_store() -> None:
    """Initialize an empty vector store with error handling."""
    global vector_store, metadata
    try:
        logger.info("Initializing vector store")
        embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
        vector_store = FAISS.from_texts([""], embeddings, metadatas=[{"doc_id": "empty"}])
        metadata = {"empty": {"label": "neutral", "source": "none", "chunk_index": 0}}
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise VectorStoreError(f"Vector store initialization failed: {str(e)}")

def load_vector_store() -> None:
    """Load the vector store from disk with enhanced error handling."""
    global vector_store, metadata
    try:
        logger.info(f"Attempting to load vector store from {VECTOR_STORE_PATH}")
        if not (os.path.exists(VECTOR_STORE_PATH) and os.path.exists(METADATA_PATH)):
            logger.info("No existing vector store found, initializing new one")
            initialize_vector_store()
            return

        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
            
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata with {len(metadata)} entries")
            
            # Verify vector store is working
            try:
                vector_store.similarity_search("test", k=1)
                logger.info("Vector store verified working")
            except Exception as e:
                logger.error(f"Vector store verification failed: {str(e)}")
                raise VectorStoreError("Vector store verification failed")
                
        except Exception as e:
            logger.error(f"Error loading existing vector store: {str(e)}")
            raise VectorStoreError(f"Failed to load vector store: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in load_vector_store: {str(e)}")
        initialize_vector_store()

def save_vector_store() -> None:
    """Save the vector store and metadata to disk with enhanced error handling."""
    if not (vector_store and metadata):
        logger.warning("No vector store or metadata to save")
        return
        
    try:
        os.makedirs("data", exist_ok=True)
        logger.info(f"Saving vector store to {VECTOR_STORE_PATH}")
        
        if os.path.exists(VECTOR_STORE_PATH):
            logger.info("Removing existing vector store")
            shutil.rmtree(VECTOR_STORE_PATH)
            
        vector_store.save_local(VECTOR_STORE_PATH)
        logger.info(f"Saving metadata with {len(metadata)} entries to {METADATA_PATH}")
        
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f)
        
        # Verify save was successful
        try:
            test_store = FAISS.load_local(
                VECTOR_STORE_PATH, 
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )
            test_store.similarity_search("test", k=1)
            logger.info("Save verified working")
        except Exception as e:
            logger.error(f"Save verification failed: {str(e)}")
            raise VectorStoreError("Save verification failed")
            
    except Exception as e:
        logger.error(f"Error saving vector store: {str(e)}")
        raise VectorStoreError(f"Failed to save vector store: {str(e)}") 