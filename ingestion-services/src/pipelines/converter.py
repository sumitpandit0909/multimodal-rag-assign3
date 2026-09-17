import subprocess
import os
import tempfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def convert_to_pdf(input_file_path:Path,output_dir:Path) -> Path:
    """
    converts DOCX/PPT/PPTX to pdf using headless LibreOffice with an isolated user profile.
    """
    if input_file_path.suffix.lower()==".pdf":
        return input_file_path

    with tempfile.TemporaryDirectory() as user_profile_dir:
        cmd = [
            "libreoffice",
            "--headless",
            f"-env:UserInstallation=file://{user_profile_dir}",
            "--convert-to", "pdf",
            "--outdir", str(output_dir),
            str(input_file_path)
        ]

        logger.info(f"Converting {input_file_path.name} to PDF...")

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)

        if result.returncode!=0:
            logger.error(f"PDF conversion failed for {input_file_path}")
            raise RuntimeError(f"LibreOffice conversion failed:\n{result.stderr}")
            
    expected_pdf = output_dir / f"{input_file_path.stem}.pdf"
    if not expected_pdf.exists():
        raise FileNotFoundError(f"Expected PDF output not found at {expected_pdf}")
        
    return expected_pdf
