import requests
import os
from glob import glob

# Base URL for the API
BASE_URL = "http://localhost:8000"

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
    print("Testing predictions on new PDFs:")
    print("-" * 50)
    
    # Test predictions on all PDFs in the test directory
    for pdf_file in glob("data/test/*.pdf"):
        # Get expected label based on filename
        expected_label = {
            "tech_paper.pdf": "good",
            "blog_post.pdf": "neutral",
            "business_proposal.pdf": "good",
            "product_desc.pdf": "bad"
        }.get(os.path.basename(pdf_file), "unknown")
        
        result = test_prediction(pdf_file)
        print(f"File: {os.path.basename(pdf_file)}")
        print(f"Expected label: {expected_label}")
        print(f"Predicted label: {result['predicted_class']}")
        print("-" * 50)

if __name__ == "__main__":
    main() 