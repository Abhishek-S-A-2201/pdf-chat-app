import pytest
import fitz  # PyMuPDF
from pathlib import Path
from src.parser import extract_text
import pytesseract

# Define a constant for the text we'll use in our test PDFs.
# This avoids "magic strings" and makes tests easier to read.
TEST_TEXT = "This is a test document for our PDF parser."


@pytest.fixture(scope="module")
def native_pdf(tmp_path_factory) -> Path:
    """
    Pytest fixture to create a temporary NATIVE PDF file for testing.
    This PDF contains selectable text.
    The fixture yields the file path and is cleaned up automatically by pytest.
    'scope="module"' means this is created only once for all tests in this file.
    """
    # tmp_path_factory is a pytest fixture that provides a temporary directory
    # unique to the test module.
    pdf_path = tmp_path_factory.mktemp("data") / "sample_text_document.pdf"
    doc = fitz.open()  # Create a new, empty PDF
    page = doc.new_page()
    # Insert text into the page. This makes it a "native" PDF.
    page.insert_text((50, 72), TEST_TEXT, fontsize=12)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture(scope="module")
def scanned_pdf(tmp_path_factory) -> Path:
    """
    Pytest fixture to create a temporary SCANNED (image-based) PDF file.
    This simulates a scanned document by rendering text to an image and
    inserting that image into the PDF, so there is no selectable text layer.
    """
    pdf_path = tmp_path_factory.mktemp("data") / "sample_scanned_document.pdf"
    
    # Step 1: Create a temporary document with text to render
    text_doc = fitz.open()
    text_page = text_doc.new_page()
    text_page.insert_text((50, 72), TEST_TEXT, fontsize=12)
    
    # Step 2: Store the dimensions before closing the document
    page_width = text_page.rect.width
    page_height = text_page.rect.height
    
    # Step 3: Render the page to a pixmap (image)
    pix = text_page.get_pixmap(dpi=200)
    text_doc.close()

    # Step 4: Create the final PDF and insert the image
    img_doc = fitz.open()
    # Corrected line: use width and height to set the page size
    img_page = img_doc.new_page(width=page_width, height=page_height) 
    img_page.insert_image(img_page.rect, pixmap=pix)
    img_doc.save(pdf_path)
    img_doc.close()
    
    return pdf_path


@pytest.fixture(scope="module")
def empty_pdf(tmp_path_factory) -> Path:
    """
    Pytest fixture to create a temporary, completely empty PDF file.
    """
    pdf_path = tmp_path_factory.mktemp("data") / "sample_empty_document.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_extract_text_from_native_pdf(native_pdf):
    """
    Tests the primary path: successfully extracting text from a native PDF.
    """
    extracted = extract_text(native_pdf)
    assert TEST_TEXT in extracted
    print(f"\nNative PDF Extraction PASSED. Found '{TEST_TEXT}'")


def test_extract_text_from_scanned_pdf(scanned_pdf):
    """
    Tests the fallback path: successfully extracting text from a scanned PDF via OCR.
    Note: This test is slower because it invokes the Tesseract OCR engine.
    """
    extracted = extract_text(scanned_pdf)
    # OCR might have minor inaccuracies (e.g., periods, spacing), so we check
    # for a significant portion of the text, not an exact match.
    assert "This is a test document" in extracted
    print(f"\nScanned PDF Extraction PASSED. Found text via OCR.")


def test_handles_empty_pdf(empty_pdf):
    """
    Tests that an empty PDF returns an empty string.
    """
    extracted = extract_text(empty_pdf)
    assert extracted == ""
    print("\nEmpty PDF Handling PASSED.")


def test_handles_file_not_found(caplog):
    """
    Tests graceful failure when the file path does not exist.
    'caplog' is a pytest fixture to capture logging output.
    """
    non_existent_path = Path("non_existent_file.pdf")
    extracted = extract_text(non_existent_path)
    assert extracted == ""
    # Check that a specific error message was logged
    assert f"Invalid file path or not a PDF: {non_existent_path}" in caplog.text
    print("\nFile Not Found Handling PASSED.")


def test_handles_non_pdf_file(tmp_path):
    """
    Tests graceful failure when the file is not a PDF.
    """
    not_a_pdf = tmp_path / "test.txt"
    not_a_pdf.write_text("This is not a pdf.")
    extracted = extract_text(not_a_pdf)
    assert extracted == ""
    print("\nNon-PDF File Handling PASSED.")


def test_ocr_fallback_is_triggered(scanned_pdf, mocker, caplog):
    """
    Verifies that the OCR fallback logic is correctly triggered.
    """
    # Spy on the internal functions
    pymupdf_spy = mocker.spy(fitz, "open")
    ocr_spy = mocker.spy(pytesseract, "image_to_string")

    # Run the extraction on a scanned PDF
    extract_text(scanned_pdf)

    # Assertions
    # fitz.open is called for both methods, so at least twice
    assert pymupdf_spy.call_count >= 2
    # The crucial check: was OCR actually used?
    assert ocr_spy.called
    
    # Corrected assertion to match the actual log message
    assert "Falling back to OCR" in caplog.text 
    
    print("\nOCR Fallback Trigger Logic PASSED.")