# Import required libraries
from pypdf import PdfReader  # For reading PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks
from sentence_transformers import SentenceTransformer  # For generating text embeddings
from sklearn.linear_model import LogisticRegression  # For classification
import pickle  # For saving/loading the trained model
import numpy as np  # For numerical operations
from fastapi import FastAPI, UploadFile, File  # For creating the API
from tempfile import NamedTemporaryFile  # For handling uploaded files
import shutil  # For file operations
from typing import List  # For type hints
import openai  # For LLM-based classification
import os  # For file path operations

# Initialize text splitter with chunk size of 1000 characters and 200 character overlap
# This helps in breaking down large documents into manageable pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Initialize the sentence transformer model for generating embeddings
# This model converts text chunks into numerical vectors
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Define mapping between text labels and numerical values for the classifier
label_map = {'bad': 0, 'neutral': 1, 'good': 2}
# Create reverse mapping for converting predictions back to text labels
reverse_label_map = {v: k for k, v in label_map.items()}
# Path where the trained model will be saved
MODEL_PATH = os.path.join("data", "classifier.pkl")

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
    Train a classifier model using PDF documents and their labels.
    
    Args:
        paths: List of paths to PDF files
        labels: List of labels ('bad', 'neutral', 'good') corresponding to the PDFs
    """
    X, y = [], []  # Lists to store features and labels
    for path, label in zip(paths, labels):
        # Extract text from PDF
        text = extract_text(path)
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        # Generate embeddings for each chunk
        embeddings = embedder.encode(chunks)
        # Add embeddings and corresponding labels to training data
        X.extend(embeddings)
        y.extend([label_map[label]] * len(embeddings))
    # Train a logistic regression classifier
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    # Save the trained model to disk
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    pickle.dump(clf, open(MODEL_PATH, "wb"))

def classify_chunk_with_llm(chunk: str) -> str:
    """
    Classify a text chunk using OpenAI's GPT-4 model.
    
    Args:
        chunk: Text chunk to classify
        
    Returns:
        Classification label ('bad', 'neutral', 'good')
    """
    # Create a prompt for the LLM
    prompt = f"Classify this text as 'bad', 'neutral', or 'good':\n---\n{chunk}\n---\nLabel:"
    # Get classification from OpenAI API
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    # Extract and return the classification label
    return response.choices[0].message.content.strip().lower()

def predict_class(path: str) -> str:
    """
    Predict the class of a PDF document using the trained model.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        Predicted class ('bad', 'neutral', 'good')
    """
    # Extract text from PDF
    text = extract_text(path)
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    # Load the trained classifier
    clf = pickle.load(open(MODEL_PATH, "rb"))
    # Generate embeddings for each chunk
    embeddings = embedder.encode(chunks)
    # Get predictions for each chunk
    preds = clf.predict(embeddings)
    # Calculate the average prediction and round to nearest integer
    mean_pred = int(round(np.mean(preds)))
    # Convert numerical prediction back to text label
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
async def train(pdfs: List[UploadFile] = File(...), labels: List[str] = File(...)):
    """
    API endpoint to train the classifier with uploaded PDFs and their labels.
    
    Args:
        pdfs: List of uploaded PDF files
        labels: List of labels corresponding to the PDFs
        
    Returns:
        Status message indicating successful training
    """
    paths = []
    # Save uploaded PDFs to temporary files
    for pdf in pdfs:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(pdf.file, tmp)
            paths.append(tmp.name)
    # Train the model with the uploaded PDFs
    train_model(paths, labels)
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