import fitz  # PyMuPDF
from pathlib import Path
from typing import List

def extract_page_screenshots(pdf_path: Path, output_dir: Path, dpi: int = 150) -> List[dict]:
    """
    Extracts high-resolution PNG for every page in the PDF.
    Returns list of metadata dicts: [{'page_number': 1, 'image_path': Path}]
    """
    doc = fitz.open(pdf_path)
    output_images = []
    zoom = dpi / 72  # standard PDF point size is 72 dpi
    matrix = fitz.Matrix(zoom, zoom)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text").strip()
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_path = output_dir / f"{pdf_path.stem}_page_{page_idx + 1}.png"
        pix.save(str(img_path))
        
        output_images.append({
            "page_number": page_idx + 1,
            "image_path": img_path,
            "text": text,
            "total_pages": len(doc)
        })
    doc.close()
    return output_images
