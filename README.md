# PDF Document Classifier

This application provides a FastAPI service for classifying PDF documents using RAG (Retrieval-Augmented Generation) and LLMs. It assigns one of three ordinal labels — `bad`, `neutral`, `good` — to documents based on their contents.

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
│   ├── test_app.py
│   └── test_model.py
├── requirements.txt
└── setup.sh           # Setup and run script
```

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
5. Start the FastAPI server

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
- **Input**: 
  - `pdfs`: List of PDF files
  - `labels`: List of labels ('bad', 'neutral', 'good')
- **Output**: Status message

### 2. Predict Document Class

- **Endpoint**: `/predict`
- **Method**: POST
- **Input**: 
  - `pdf`: Single PDF file
- **Output**: Predicted class ('bad', 'neutral', 'good')

## API Documentation

Once the server is running, you can access the interactive API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

The project includes several test scripts:

1. Basic API tests:
   ```bash
   pytest tests/test_app.py
   ```

2. Model training and prediction tests:
   ```bash
   python tests/test_model.py
   ```

3. Test with new PDFs:
   ```bash
   python tests/test_new_pdfs.py
   ```

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
