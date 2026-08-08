"""
Atlas AI Financial Assistant - Configuration Settings
"""

from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "Atlas AI Financial Assistant"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Telegram Bot
    TELEGRAM_BOT_TOKEN: str = Field(
        default="",
        description="Telegram bot token obtained from @BotFather"
    )
    TELEGRAM_MODE: Literal["polling", "webhook"] = "polling"
    TELEGRAM_WEBHOOK_URL: str = Field(
        default="",
        description="Public URL for Telegram webhook (e.g. https://domain.com/api/v1/webhook)"
    )
    TELEGRAM_WEBHOOK_SECRET: str = Field(
        default="",
        description="Secret token for Telegram webhook validation"
    )

    # Groq AI Engine (Fast Inference, Whisper Transcription & Vision)
    GROQ_API_KEY: str = Field(
        default="",
        description="Groq API key from https://console.groq.com/"
    )
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_VISION_MODEL: str = "qwen/qwen3.6-27b"
    GROQ_AUDIO_MODEL: str = "whisper-large-v3"

    # Google Gemini AI (Optional / Multimodal Fallback)
    GEMINI_API_KEY: str = Field(
        default="",
        description="Google Gemini API key (optional fallback)"
    )
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # MongoDB Connection
    MONGODB_URI: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection string"
    )
    MONGODB_DB_NAME: str = "atlas_financial_assistant"

    # Financial Data
    FINNHUB_API_KEY: str = Field(
        default="",
        description="Finnhub API key for real-time news and financial data"
    )
    SEC_USER_AGENT: str = "AtlasFinancialAssistant/1.0 (contact@atlasfinance.ai)"

    # Scheduler and Proactive Alerts
    DEFAULT_BRIEF_TIME: str = "08:00"
    DEFAULT_TIMEZONE: str = "UTC"
    MARKET_ALERT_THRESHOLD_PCT: float = 5.0  # 5% movement triggers alert

    # Context window limits
    MAX_CONVERSATION_HISTORY: int = 15  # Injected history messages for context
    MAX_DB_CONVERSATION_LIMIT: int = 50  # Total stored interactions per user limit

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True
    )


settings = Settings()
