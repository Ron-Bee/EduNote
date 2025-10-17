# extractors.py
from pdfminer.high_level import extract_text as extract_text_from_pdf
from docx import Document
import tempfile
from pathlib import Path
from typing import Tuple
import os

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)

def extract_text_from_pdf_or_fallback(path: str) -> str:
    try:
        return extract_text_from_pdf(path)
    except Exception as e:
        # On complex PDFs, recommend LibreOffice or external OCR (Tesseract) as next step.
        raise

def extract_text_from_uploaded_file(tmp_path: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf_or_fallback(tmp_path)
    elif ext == ".docx":
        return extract_text_from_docx(tmp_path)
    elif ext in (".txt",):
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX or TXT.")
