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


# Create training data directory if it doesn't exist
os.makedirs("data/training", exist_ok=True)

# Good content examples
good_content_1 = """This is an excellent document that demonstrates strong writing skills.
The content is well-structured and provides valuable information.
The ideas are clearly presented and supported with evidence.
The language used is professional and appropriate for the context.
The document maintains a positive and constructive tone throughout."""

good_content_2 = """A comprehensive analysis of the current market trends.
The report includes detailed data visualization and clear explanations.
Recommendations are well-supported by research and analysis.
The document follows a logical structure and is easy to follow.
The conclusions are well-reasoned and actionable."""

# Neutral content examples
neutral_content_1 = """This is a standard document with average content quality.
The information is presented in a straightforward manner.
The document contains basic facts and figures without much analysis.
The language used is simple and direct.
The content serves its purpose without being exceptional."""

neutral_content_2 = """A basic overview of the project status.
The document includes standard updates and progress reports.
Information is presented in a chronological order.
The content is factual and objective.
The document follows standard formatting guidelines."""

# Bad content examples
bad_content_1 = """This document is poorly written and lacks structure.
The content is confusing and difficult to follow.
There are numerous spelling and grammar errors.
The ideas are not well-developed or supported.
The language used is inappropriate for the context."""

bad_content_2 = """A poorly organized report with inconsistent formatting.
The content lacks clear direction or purpose.
Important information is missing or incomplete.
The document contains irrelevant details and tangents.
The conclusions are not supported by the presented data."""

# Create PDF files
create_pdf("data/training/good_doc1.pdf", good_content_1)
create_pdf("data/training/good_doc2.pdf", good_content_2)
create_pdf("data/training/neutral_doc1.pdf", neutral_content_1)
create_pdf("data/training/neutral_doc2.pdf", neutral_content_2)
create_pdf("data/training/bad_doc1.pdf", bad_content_1)
create_pdf("data/training/bad_doc2.pdf", bad_content_2)

print("Created example PDF files in the data/training directory:")
print("- good_doc1.pdf and good_doc2.pdf (positive examples)")
print("- neutral_doc1.pdf and neutral_doc2.pdf (neutral examples)")
print("- bad_doc1.pdf and bad_doc2.pdf (negative examples)")
