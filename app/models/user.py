"""
Atlas AI Financial Assistant - User Model
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class NotificationSchedule(BaseModel):
    """User preferences for proactive notifications."""
    brief_time: str = Field(default="08:00", description="Morning Brief time in HH:MM format")
    timezone: str = Field(default="UTC", description="User's local timezone (e.g., 'America/New_York', 'UTC')")
    morning_brief_enabled: bool = Field(default=True, description="Whether to send daily morning brief")
    market_alerts_enabled: bool = Field(default=True, description="Whether to send high-volatility price alerts (>5%)")
    last_brief_sent_at: Optional[datetime] = None
    last_alert_sent_at: Optional[datetime] = None


class UserProfile(BaseModel):
    """User schema for MongoDB storage and session state."""
    telegram_id: int = Field(..., description="Unique Telegram user ID")
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[str] = Field(
        default=None,
        description="Professional role (e.g. Equity Analyst, Portfolio Manager, Retail Trader, CFO)"
    )
    industries_followed: List[str] = Field(
        default_factory=list,
        description="Industries followed (e.g. Semiconductors, Clean Energy, SaaS, Biotech)"
    )
    watchlists: List[str] = Field(
        default_factory=list,
        description="Stock tickers followed (e.g. AAPL, NVDA, MSFT, TSLA)"
    )
    notification_schedule: NotificationSchedule = Field(
        default_factory=NotificationSchedule,
        description="Notification schedule and alert preferences"
    )
    integration_tokens: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional API integration tokens"
    )
    is_onboarded: bool = Field(
        default=False,
        description="Whether the user has completed natural conversational onboarding"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Account creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last profile update timestamp"
    )

    def to_mongo(self) -> Dict[str, Any]:
        """Converts model to MongoDB document dict."""
        return self.model_dump()

    @classmethod
    def from_mongo(cls, data: Dict[str, Any]) -> "UserProfile":
        """Builds model instance from MongoDB document."""
        if "_id" in data:
            data.pop("_id")
        return cls(**data)
