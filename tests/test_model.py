import requests
import os
from glob import glob

# Base URL for the API
BASE_URL = "http://localhost:8000"

def train_model():
    """Train the model using example PDFs."""
    # Get all PDF files from the training_data directory
    pdf_files = glob("data/training/*.pdf")
    
    # Create corresponding labels based on filenames
    labels = []
    for pdf_file in pdf_files:
        if "good" in pdf_file:
            labels.append("good")
        elif "neutral" in pdf_file:
            labels.append("neutral")
        else:
            labels.append("bad")
    
    # Prepare files for upload
    files = []
    for pdf_file in pdf_files:
        files.append(('pdfs', (os.path.basename(pdf_file), open(pdf_file, 'rb'), 'application/pdf')))
    
    # Add labels as form data
    data = [('labels', label) for label in labels]
    
    # Send training request
    response = requests.post(f"{BASE_URL}/train", files=files, data=data)
    
    # Close all file handles
    for _, (_, file, _) in files:
        file.close()
    
    return response.json()

def test_prediction(pdf_file):
    """Test prediction on a single PDF file."""
    # Prepare file for upload
    files = {'pdf': (os.path.basename(pdf_file), open(pdf_file, 'rb'), 'application/pdf')}
    
    # Send prediction request
    response = requests.post(f"{BASE_URL}/predict", files=files)
    
    # Close file handle
    files['pdf'][1].close()
    
    return response.json()

def main():
    print("Training the model...")
    train_result = train_model()
    print(f"Training result: {train_result}\n")
    
    print("Testing predictions on all PDFs:")
    print("-" * 50)
    
    # Test predictions on all PDFs
    for pdf_file in glob("data/training/*.pdf"):
        expected_label = "good" if "good" in pdf_file else "neutral" if "neutral" in pdf_file else "bad"
        result = test_prediction(pdf_file)
        print(f"File: {os.path.basename(pdf_file)}")
        print(f"Expected label: {expected_label}")
        print(f"Predicted label: {result['predicted_class']}")
        print("-" * 50)

if __name__ == "__main__":
    main() 