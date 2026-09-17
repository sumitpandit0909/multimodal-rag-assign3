import math
import openpyxl
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def sanitize_value(val: Any) -> Any:
    """Ensure all cell values are JSON & BSON serializable without NaNs or raw timestamps."""
    if val is None:
        return ""
    if isinstance(val, (float, int)):
        if math.isnan(val) or math.isinf(val):
            return ""
        return val
    if hasattr(val, "isoformat"):
        return val.isoformat()
    # Check for pandas NaT or missing types
    if pd.isna(val):
        return ""
    return str(val)

def sanitize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean_records = []
    for row in records:
        clean_row = {str(k): sanitize_value(v) for k, v in row.items()}
        clean_records.append(clean_row)
    return clean_records

def parse_excel_file(file_path: Path) -> List[Dict]:
    """
    Reads an Excel file, iterates through every sheet, and converts
    the data into clean markdown tables with structured metadata.
    Chunks sheets logically into row batches (strictly no screenshots).
    """
    excel_file = pd.ExcelFile(file_path)
    chunks = []
    chunk_row_size = 20

    for sheet_name in excel_file.sheet_names:
        df = excel_file.parse(sheet_name)
        df = df.dropna(how="all")  # Clean empty rows
        
        if df.empty:
            continue

        total_rows = len(df)
        
        # Process in batches of rows to stay well within embedding token limits
        for i in range(0, total_rows, chunk_row_size):
            sub_df = df.iloc[i : i + chunk_row_size]
            row_start = i + 1
            row_end = min(i + chunk_row_size, total_rows)

            # Sanitize for markdown and raw records
            clean_sub_df = sub_df.copy()
            clean_sub_df = clean_sub_df.fillna("")

            try:
                markdown_table = clean_sub_df.to_markdown(index=False)
            except Exception:
                markdown_table = clean_sub_df.to_string(index=False)

            content = (
                f"### Document: {file_path.name}\n"
                f"### Sheet: {sheet_name} (Rows {row_start} to {row_end} of {total_rows})\n\n"
                f"{markdown_table}"
            )

            raw_dict_list = clean_sub_df.to_dict(orient="records")
            sanitized_raw_data = sanitize_records(raw_dict_list)

            chunks.append({
                "file_name": file_path.name,
                "source_type": "tabular",
                "page_number": None,
                "sheet_name": sheet_name,
                "screenshot_url": None,
                "raw_data": sanitized_raw_data,  # Clean BSON for Data Card preview
                "text_content": content
            })

    return chunks
