"""
Atlas AI Financial Assistant - PDF & Document Parser
Extracts structured text, financial sections, and page-indexed chunks from PDF filings.
"""

import io
from typing import Dict, Any, List
from pypdf import PdfReader
from app.core.logger import logger


class PDFParserService:
    """Parses PDF documents into page-indexed text and semantic chunks."""

    @classmethod
    def parse_pdf_bytes(cls, pdf_bytes: bytes, file_name: str = "document.pdf") -> Dict[str, Any]:
        """Extracts text from PDF bytes and breaks into clean page-indexed segments."""
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(reader.pages)
            pages_text: List[Dict[str, Any]] = []
            full_text_list: List[str] = []

            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                text = page.extract_text() or ""
                clean_text = " ".join(text.split())
                if clean_text:
                    pages_text.append({
                        "page_number": page_num,
                        "text": clean_text
                    })
                    full_text_list.append(f"[Page {page_num}]: {clean_text}")

            combined_full_text = "\n\n".join(full_text_list)
            
            # Simple chunking (approx 1500 chars per chunk with 200 char overlap)
            chunks = cls._create_chunks(combined_full_text, chunk_size=1500, overlap=200)

            # Executive summary preview (first 1000 characters)
            preview_summary = combined_full_text[:1200] + ("..." if len(combined_full_text) > 1200 else "")

            return {
                "success": True,
                "file_name": file_name,
                "num_pages": num_pages,
                "pages": pages_text,
                "full_text": combined_full_text,
                "chunks": chunks,
                "preview_summary": preview_summary
            }
        except Exception as e:
            logger.error(f"Error parsing PDF document {file_name}: {e}")
            return {
                "success": False,
                "file_name": file_name,
                "error": f"Failed to extract PDF content: {str(e)}"
            }

    @staticmethod
    def _create_chunks(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
        """Splits long text into overlapping chunks for retrieval."""
        if not text:
            return []
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start += (chunk_size - overlap)
        return chunks
