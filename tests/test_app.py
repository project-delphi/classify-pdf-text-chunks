import pytest
import os
import tempfile
from fastapi.testclient import TestClient
from app import app, extract_text, train_model, predict_class, classify_chunk_with_llm
import numpy as np
from unittest.mock import patch, MagicMock
from reportlab.pdfgen import canvas
from io import BytesIO

# Create a test client
client = TestClient(app)

def create_test_pdf(text: str) -> BytesIO:
    """Helper function to create a PDF file in memory"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, text)
    c.showPage()  # Ensure the page is complete
    c.save()
    # Get the value of the BytesIO buffer and create a fresh buffer
    pdf_bytes = buffer.getvalue()
    buffer.close()
    new_buffer = BytesIO(pdf_bytes)
    return new_buffer

def create_pdf_bytes(text: str) -> bytes:
    """Create a PDF in memory and return its bytes"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, text)
    c.showPage()
    c.save()
    return buffer.getvalue()

def test_extract_text():
    # Create a temporary PDF file with some text
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        # Create a PDF using reportlab
        c = canvas.Canvas(tmp.name)
        c.drawString(100, 750, "Hello World")
        c.save()
        tmp_path = tmp.name

    try:
        # Test the extract_text function
        text = extract_text(tmp_path)
        assert "Hello World" in text
    finally:
        # Clean up
        os.unlink(tmp_path)

def test_train_model():
    # Create temporary PDF files
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp1, \
         tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp2:
        # Create PDFs using reportlab
        c1 = canvas.Canvas(tmp1.name)
        c1.drawString(100, 750, "Good document")
        c1.save()
        
        c2 = canvas.Canvas(tmp2.name)
        c2.drawString(100, 750, "Bad document")
        c2.save()
        
        paths = [tmp1.name, tmp2.name]

    try:
        # Test the train_model function
        train_model(paths, ['good', 'bad'])
        # Verify that the model file was created
        assert os.path.exists('classifier.pkl')
    finally:
        # Clean up
        for path in paths:
            os.unlink(path)
        if os.path.exists('classifier.pkl'):
            os.unlink('classifier.pkl')

def test_train_endpoint():
    # Create sample PDFs
    pdf1 = create_test_pdf("This is a good document")
    pdf2 = create_test_pdf("This is a bad document")
    
    # Create form data
    files = [
        ('pdfs', ('doc1.pdf', pdf1, 'application/pdf')),
        ('pdfs', ('doc2.pdf', pdf2, 'application/pdf')),
    ]
    data = {
        'labels': ('good', 'bad')
    }
    
    response = client.post('/train', files=files, data=data)
    assert response.status_code == 200
    assert response.json()['status'] == 'model trained'

def test_predict_endpoint():
    # Create PDF files in memory
    pdf1_bytes = create_pdf_bytes("This is a good document")
    pdf2_bytes = create_pdf_bytes("This is a bad document")
    test_pdf_bytes = create_pdf_bytes("This is a test document")
    
    # Train the model
    train_files = [
        ('pdfs', ('doc1.pdf', BytesIO(pdf1_bytes), 'application/pdf')),
        ('pdfs', ('doc2.pdf', BytesIO(pdf2_bytes), 'application/pdf'))
    ]
    train_data = {'labels': ('good', 'bad')}
    response = client.post('/train', files=train_files, data=train_data)
    assert response.status_code == 200
    
    # Test prediction
    test_file = BytesIO(test_pdf_bytes)
    test_file.seek(0)  # Ensure we're at the start of the file
    response = client.post(
        '/predict',
        files={'pdf': ('test.pdf', test_file, 'application/pdf')}
    )
    assert response.status_code == 200
    assert 'predicted_class' in response.json()
    assert response.json()['predicted_class'] in ['good', 'neutral', 'bad']

@patch('openai.OpenAI')
def test_classify_chunk_with_llm(mock_openai_class):
    # Mock the OpenAI client instance
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Mock the chat completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "good"
    mock_client.chat.completions.create.return_value = mock_response

    # Test the function
    result = classify_chunk_with_llm("This is a test document")
    
    # Assert the result
    assert result == "good"
    
    # Verify the API was called with correct parameters
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args['model'] == 'gpt-4'
    assert len(call_args['messages']) == 1
    assert call_args['messages'][0]['role'] == 'user'
    assert 'This is a test document' in call_args['messages'][0]['content'] 