# PDF Document Classifier

This application provides a FastAPI service for classifying PDF documents based on their content quality. It uses sentence embeddings, machine learning, and Retrieval-Augmented Generation (RAG) to assign one of three ordinal labels — `bad`, `neutral`, `good` — to documents based on their writing quality, structure, and content.

## Project Structure

```
.
├── README.md
├── data/               # Data directory
│   ├── test/          # Test data
│   └── training/      # Training data
├── src/               # Source code
│   ├── app.py         # Main application
│   └── create_example_pdfs.py
├── tests/             # Test scripts
│   ├── conftest.py    # Test configuration and fixtures
│   ├── test_app.py    # API and functionality tests
│   └── test_model.py
├── requirements.txt
└── setup.sh           # Setup and run script
```

## How It Works

The classifier uses a three-stage approach:

1. **Text Processing**:
   - Extracts text from PDF documents
   - Splits text into manageable chunks
   - Generates embeddings using SentenceTransformer

2. **Vector Store and RAG**:
   - Stores document chunks in a FAISS vector store
   - Uses RAG to retrieve relevant examples for classification
   - Maintains metadata about document quality and sources

3. **Classification**:
   - Uses a Logistic Regression model trained on labeled examples
   - Optionally uses GPT-4 with RAG context for more nuanced classification
   - Aggregates predictions across chunks to determine final document quality

## Setup

### Quick Start

The easiest way to get started is to use the provided setup script:

```bash
# Make the script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

This script will:
1. Check for Python 3 installation
2. Create a virtual environment if it doesn't exist
3. Install all required dependencies
4. Check for OpenAI API key
5. Initialize the vector store
6. Start the FastAPI server

### Manual Setup

If you prefer to set up manually:

1. Create a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key (if using LLM classification):
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

4. Initialize the vector store:
   ```bash
   python -c "from src.app import initialize_vector_store; initialize_vector_store()"
   ```

## Running the Application

### Using the Setup Script

Simply run:
```bash
./setup.sh
```

### Manual Start

Start the FastAPI server:
```bash
uvicorn src.app:app --reload
```

The server will start at `http://localhost:8000`.

## API Endpoints

### 1. Train the Classifier

- **Endpoint**: `/train`
- **Method**: POST
- **Description**: Train the classifier with labeled PDF documents
- **Input**:
  - `pdfs`: List of PDF files
  - `labels`: Comma-separated list of labels ('bad', 'neutral', 'good')
- **Output**: Status message

### 2. Predict Document Quality

- **Endpoint**: `/predict`
- **Method**: POST
- **Description**: Predict the quality of a PDF document using RAG-enhanced classification
- **Input**: PDF file
- **Output**: Predicted quality label

### 3. Health Check

- **Endpoint**: `/`
- **Method**: GET
- **Description**: Check if the service is running
- **Output**: Service status

## Testing

The application includes a comprehensive test suite that covers:
- Text extraction from PDFs
- Text splitting functionality
- Vector store operations
- RAG-based classification
- API endpoints

Run the tests with:
```bash
python -m pytest tests/ -v
```

## Vector Store Management

The application uses FAISS for efficient vector storage and retrieval. The vector store is automatically initialized when the application starts and persists between sessions. It stores:
- Document chunks and their embeddings
- Metadata about document quality
- Source information for RAG context

## Rate Limiting

The application includes rate limiting for API calls to prevent overload:
- Token bucket algorithm implementation
- Configurable rate limits
- Automatic token refill

## API Documentation

Once the server is running, you can access the interactive API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Generating Example PDFs

To generate example PDFs for training or testing:

1. Training PDFs:
   ```bash
   python src/create_example_pdfs.py
   ```

2. Test PDFs:
   ```bash
   python src/create_test_pdfs.py
   ```
