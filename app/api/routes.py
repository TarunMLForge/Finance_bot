"""
Atlas AI Financial Assistant - REST API Endpoints
Provides health monitoring, Telegram webhook ingestion, and diagnostic triggers.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Request, HTTPException, Header, status, UploadFile, File, Form
from telegram import Update

from app.core.config import settings
from app.core.logger import logger
from app.core.database import DatabaseManager, get_users_collection
from app.models.user import UserProfile
from app.services.financial.yfinance_service import YFinanceService
from app.services.financial.finnhub_service import FinnhubService
from app.services.ai.groq_service import GroqService
from app.services.scheduler.jobs import SchedulerService
from app.telegram.bot import TelegramBotManager

router = APIRouter()


@router.get("/", tags=["General"])
async def root():
    """Returns application name and operational state."""
    return {
        "app": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "status": "operational",
        "docs_url": "/docs"
    }


@router.get("/health", tags=["Health & Diagnostics"])
async def health_check():
    """Detailed health check of all integrated components."""
    mongo_status = "connected" if DatabaseManager.db is not None else "disconnected"
    bot_status = "active" if TelegramBotManager.app is not None else "uninitialized"
    scheduler_status = "running" if (SchedulerService.scheduler and SchedulerService.scheduler.running) else "inactive"
    groq_status = "configured" if settings.GROQ_API_KEY else "unconfigured"

    return {
        "status": "healthy" if mongo_status == "connected" else "degraded",
        "components": {
            "database_mongodb": mongo_status,
            "telegram_bot": bot_status,
            "apscheduler": scheduler_status,
            "ai_groq": groq_status,
            "telegram_mode": settings.TELEGRAM_MODE
        }
    }


@router.post("/api/v1/webhook", tags=["Telegram Webhook"])
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str = Header(default=None)
):
    """Processes incoming Telegram updates via Webhook in production."""
    if settings.TELEGRAM_WEBHOOK_SECRET and x_telegram_bot_api_secret_token != settings.TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid secret token")

    if TelegramBotManager.app is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Bot not initialized")

    try:
        data = await request.json()
        update = Update.de_json(data, TelegramBotManager.app.bot)
        await TelegramBotManager.app.process_update(update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error processing Telegram webhook update: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/api/v1/users", tags=["Users & Intelligence"])
async def list_users():
    """Lists registered users and their active watchlists."""
    users_col = get_users_collection()
    if users_col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    cursor = users_col.find({})
    users = []
    async for doc in cursor:
        users.append(UserProfile.from_mongo(doc).model_dump(mode="json"))
    return {"total": len(users), "users": users}


@router.post("/api/v1/trigger-brief/{telegram_id}", tags=["Manual Triggers"])
async def trigger_morning_brief(telegram_id: int):
    """Manually triggers generation and push of a morning brief to a user."""
    users_col = get_users_collection()
    if users_col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    doc = await users_col.find_one({"telegram_id": telegram_id})
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")

    user = UserProfile.from_mongo(doc)
    await SchedulerService.generate_and_send_user_brief(user)
    return {"status": "success", "message": f"Morning brief sent to {telegram_id}"}


@router.post("/api/v1/trigger-volatility-scan", tags=["Manual Triggers"])
async def trigger_volatility_scan():
    """Manually triggers the market volatility monitor."""
    await SchedulerService.check_market_volatility_alerts()
    return {"status": "success", "message": "Market volatility scanner triggered"}


@router.get("/api/v1/financial/quote/{ticker}", tags=["Financial Data Test"])
async def test_quote(ticker: str):
    """Live quote test endpoint."""
    return await YFinanceService.get_stock_price(ticker)


@router.get("/api/v1/financial/summary/{ticker}", tags=["Financial Data Test"])
async def test_summary(ticker: str):
    """Fundamental valuation test endpoint."""
    return await YFinanceService.get_financial_summary(ticker)


@router.get("/api/v1/financial/news/{ticker}", tags=["Financial Data Test"])
async def test_news(ticker: str, limit: int = 5):
    """Company news test endpoint."""
    return await FinnhubService.get_company_news(ticker, limit=limit)


@router.post("/api/v1/financial/analyze-image", tags=["Multimodal Vision Test"])
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    caption: Optional[str] = Form(default=None)
):
    """
    Direct endpoint for analyzing stock charts, financial statements, or screenshots.
    Accepts PNG, JPG, JPEG, WEBP, BMP.
    """
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"
    
    analysis = await GroqService.analyze_image(
        image_bytes=image_bytes,
        caption=caption,
        mime_type=mime_type
    )
    return {
        "status": "success",
        "filename": file.filename,
        "content_type": mime_type,
        "caption": caption,
        "analysis": analysis
    }

