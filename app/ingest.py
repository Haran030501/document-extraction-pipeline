"""PDF -> text. Uses the embedded text layer, falling back to Tesseract OCR per page."""

import hashlib
import io
import logging
from dataclasses import dataclass

import pdfplumber

from app.config import get_settings

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    text: str
    page_count: int
    ocr_pages: list[int]
    sha256: str

    @property
    def ocr_used(self) -> bool:
        return bool(self.ocr_pages)


def _ocr_page(pdf_bytes: bytes, page_number: int) -> str:
    # Imported lazily: OCR needs the tesseract + poppler system binaries (installed in Docker).
    import pytesseract
    from pdf2image import convert_from_bytes

    images = convert_from_bytes(pdf_bytes, dpi=300, first_page=page_number, last_page=page_number)
    return pytesseract.image_to_string(images[0]) if images else ""


def extract_text(pdf_bytes: bytes) -> IngestResult:
    min_chars = get_settings().ocr_min_chars
    pages: list[str] = []
    ocr_pages: list[int] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) < min_chars:
                try:
                    ocr_text = _ocr_page(pdf_bytes, i)
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        ocr_pages.append(i)
                except Exception as e:  # tesseract/poppler missing or failed
                    log.warning("OCR failed on page %d: %s", i, e)
            pages.append(f"[Page {i}]\n{text.strip()}")
        page_count = len(pdf.pages)
    return IngestResult(
        text="\n\n".join(pages),
        page_count=page_count,
        ocr_pages=ocr_pages,
        sha256=hashlib.sha256(pdf_bytes).hexdigest(),
    )
