from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os


def create_pdf(filename, content):
    """Create a PDF file with the given content."""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Set font and size
    c.setFont("Helvetica", 12)

    # Split content into lines and write to PDF
    lines = content.split("\n")
    y = height - 50  # Start from top of page

    for line in lines:
        if y < 50:  # If we're near the bottom of the page
            c.showPage()  # Create a new page
            y = height - 50  # Reset y position
            c.setFont("Helvetica", 12)

        c.drawString(50, y, line)
        y -= 20  # Move down for next line

    c.save()


# Create test data directory if it doesn't exist
os.makedirs("data/test", exist_ok=True)

# Test content examples with different styles and topics
test_content_1 = """A technical research paper on machine learning applications.
The methodology section is well-documented with clear experimental setup.
Results are presented with appropriate statistical analysis.
The discussion provides valuable insights into the findings.
Future work directions are clearly outlined."""

test_content_2 = """A casual blog post about recent technology trends.
The writing style is informal and conversational.
Some interesting points are made but lack depth.
The content is engaging but not particularly informative.
The conclusion is somewhat abrupt."""

test_content_3 = """A business proposal for a new project.
The executive summary is concise and clear.
Financial projections are well-supported with data.
The implementation timeline is realistic and detailed.
Risk assessment is thorough and mitigation strategies are provided."""

test_content_4 = """A poorly formatted product description.
the content jumps between topics without clear transitions
some important details are missing or unclear
the language is inconsistent and contains errors
the overall structure needs improvement"""

# Create PDF files
create_pdf("data/test/tech_paper.pdf", test_content_1)
create_pdf("data/test/blog_post.pdf", test_content_2)
create_pdf("data/test/business_proposal.pdf", test_content_3)
create_pdf("data/test/product_desc.pdf", test_content_4)

print("Created test PDF files in the data/test directory:")
print("- tech_paper.pdf (expected: good)")
print("- blog_post.pdf (expected: neutral)")
print("- business_proposal.pdf (expected: good)")
print("- product_desc.pdf (expected: bad)")
