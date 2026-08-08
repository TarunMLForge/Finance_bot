"""
Atlas AI Financial Assistant - Main FastAPI Application
Entrypoint for server, background scheduler, and Telegram bot lifecycle.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.core.logger import logger
from app.core.database import DatabaseManager
from app.services.scheduler.jobs import SchedulerService
from app.telegram.bot import TelegramBotManager
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager handling startup and graceful shutdown."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} in [{settings.ENVIRONMENT.upper()}] mode")
    logger.info("=" * 60)

    # 1. Connect to MongoDB and create indexes
    await DatabaseManager.connect()

    # 2. Start APScheduler for Proactive Intelligence & Alerts
    SchedulerService.start()

    # 3. Start Telegram Bot (Polling or Webhook)
    await TelegramBotManager.start_bot()

    yield

    # Shutdown sequence
    logger.info("Initiating graceful application shutdown...")
    await TelegramBotManager.stop_bot()
    SchedulerService.shutdown()
    await DatabaseManager.disconnect()
    logger.info("All services shut down gracefully.")


# FastAPI Application Instance
app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Production-grade, zero-command Telegram AI Agent for finance professionals. "
        "Delivers real-time quotes, fundamental equity valuation, SEC filing intelligence, "
        "and proactive daily morning briefs with zero infrastructure cost."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routes
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
