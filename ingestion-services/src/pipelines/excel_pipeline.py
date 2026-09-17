import openpyxl
import pandas as pd
from pathlib import Path
from typing import List, Dict

def parse_excel_file(file_path: Path) -> List[Dict]:
    """
    Reads an Excel file, iterates through every sheet, and converts
    the data into clean markdown tables with metadata.
    """
    excel_file = pd.ExcelFile(file_path)
    chunks = []

    for sheet_name in excel_file.sheet_names:
        df = excel_file.parse(sheet_name)
        df = df.dropna(how="all")  # Clean empty rows
        
        if df.empty:
            continue

        # Convert dataframe to markdown representation
        markdown_table = df.to_markdown(index=False)
        
        content = (
            f"### Document: {file_path.name}\n"
            f"### Sheet: {sheet_name}\n\n"
            f"{markdown_table}"
        )

        chunks.append({
            "file_name": file_path.name,
            "source_type": "tabular",
            "page_number": None,
            "sheet_name": sheet_name,
            "screenshot_url": None,
            "raw_data": df.head(100).to_dict(orient="records"),  # For Data Card preview
            "text_content": content
        })

    return chunks
