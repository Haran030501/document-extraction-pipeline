"""Create image-only ("scanned") copies of PDFs to exercise the OCR path.

    python -m scripts.make_scanned ca2_22_282 cal_s239777

Each page is rasterized, slightly rotated and blurred, and re-saved as an image-only
PDF at data/pdfs/<id>_scanned.pdf, so there is no text layer to extract.
"""

import random
import sys
from pathlib import Path

from pdf2image import convert_from_path
from PIL import ImageFilter

PDFS = Path(__file__).resolve().parent.parent / "data" / "pdfs"


def make_scanned(doc_id: str, dpi: int = 150) -> Path:
    rng = random.Random(doc_id)
    pages = convert_from_path(PDFS / f"{doc_id}.pdf", dpi=dpi, grayscale=True)
    degraded = [
        p.rotate(rng.uniform(-1.2, 1.2), expand=False, fillcolor=255).filter(ImageFilter.GaussianBlur(0.6))
        for p in pages
    ]
    out = PDFS / f"{doc_id}_scanned.pdf"
    degraded[0].save(out, save_all=True, append_images=degraded[1:], resolution=dpi)
    return out


if __name__ == "__main__":
    for doc_id in sys.argv[1:]:
        print(make_scanned(doc_id))
