"""
Atlas AI Financial Assistant - Conversation Message Model
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """Stores interaction messages for conversational context."""
    telegram_id: int = Field(..., description="Telegram user ID")
    role: MessageRole = Field(..., description="Role of message author (user / assistant)")
    content: str = Field(..., description="Message text content")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the message"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional metadata such as tool calls, voice transcription flags, etc."
    )

    def to_mongo(self) -> Dict[str, Any]:
        """Converts model to MongoDB document."""
        data = self.model_dump()
        data["role"] = self.role.value
        return data

    @classmethod
    def from_mongo(cls, data: Dict[str, Any]) -> "ConversationMessage":
        """Builds instance from MongoDB record."""
        if "_id" in data:
            data.pop("_id")
        return cls(**data)
