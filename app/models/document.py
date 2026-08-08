"""
Atlas AI Financial Assistant - Document Model for Financial Filings / PDFs
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """Stores uploaded document metadata, text summaries, and extracted content."""
    telegram_id: int = Field(..., description="Telegram user ID who uploaded the document")
    document_id: str = Field(..., description="Unique document or file identifier")
    file_name: str = Field(..., description="Original file name")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    extracted_text_summary: str = Field(
        default="",
        description="High-level executive summary of the document"
    )
    full_text_chunks: List[str] = Field(
        default_factory=list,
        description="Text chunks extracted from the document for contextual search"
    )
    upload_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Upload timestamp"
    )

    def to_mongo(self) -> Dict[str, Any]:
        """Converts model to MongoDB document."""
        return self.model_dump()

    @classmethod
    def from_mongo(cls, data: Dict[str, Any]) -> "DocumentRecord":
        """Builds instance from MongoDB record."""
        if "_id" in data:
            data.pop("_id")
        return cls(**data)
