import fitz
import pytesseract
import logging
from pathlib import Path
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _extract_text_with_pymupdf(pdf_path: Path) -> str:
    """
    Extracts text from a PDF using PyMuPDF.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        The extracted text as a single string.
    """
    logging.info(f"Attempting to extract text with PyMuPDF from '{pdf_path.name}'...")
    try:
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
        return text.strip()
    except Exception as e:
        logging.error(f"PyMuPDF failed on '{pdf_path.name}': {e}")
        return ""


def _extract_text_with_ocr(pdf_path: Path) -> str:
    """
    Extracts text from a PDF using OCR (Tesseract). Renders each page as an
    image and then performs OCR.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        The extracted text as a single string.
    """
    logging.warning(f"Falling back to OCR for '{pdf_path.name}'. This may be slow.")
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                logging.info(f"OCR processing page {page_num + 1}/{len(doc)}...")
                # Render page to a pixmap (image)
                pix = page.get_pixmap(dpi=300)  # Higher DPI for better OCR
                img_data = pix.tobytes("png")
                image = Image.open(BytesIO(img_data))
                
                # Perform OCR
                page_text = pytesseract.image_to_string(image, lang='eng')
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logging.error(f"OCR processing failed for '{pdf_path.name}': {e}")
        return ""


def extract_text(pdf_path: str | Path) -> str:
    """
    Extracts text from a PDF file using a robust two-step approach.

    First, it tries the fast and accurate PyMuPDF method. If the extracted
    text is empty or too short (indicating a scanned/image-based PDF),
    it falls back to a more resource-intensive OCR method.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        The extracted text, or an empty string if both methods fail.
    """
    path = Path(pdf_path)
    if not path.is_file() or path.suffix.lower() != '.pdf':
        logging.error(f"Invalid file path or not a PDF: {path}")
        return ""

    # 1. Primary method: PyMuPDF
    extracted_text = _extract_text_with_pymupdf(path)

    # 2. Check if the primary method failed (e.g., for a scanned PDF)
    # Heuristic: If the text is very short, it's likely an image-based PDF.
    if len(extracted_text) < 100:  # Threshold can be adjusted
        logging.info("PyMuPDF extracted little or no text. Triggering OCR fallback.")
        extracted_text = _extract_text_with_ocr(path)

    if not extracted_text:
        logging.warning(f"Both extraction methods failed to retrieve text from '{path.name}'.")

    logging.info(f"Successfully finished processing '{path.name}'.")
    return extracted_text