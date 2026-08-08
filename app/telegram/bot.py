"""
Atlas AI Financial Assistant - Telegram Bot Application Manager
"""

import asyncio
from typing import Optional
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from app.core.config import settings
from app.core.logger import logger
from app.services.scheduler.jobs import SchedulerService
from app.telegram.handlers import (
    start_handler,
    text_message_handler,
    voice_message_handler,
    photo_message_handler,
    document_upload_handler
)


async def global_telegram_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Logs uncaught exceptions in Telegram handlers."""
    logger.error(f"Telegram handler exception on update {update}: {context.error}", exc_info=context.error)


class TelegramBotManager:
    """Manages Telegram Bot lifecycle, handler registration, and polling/webhook mode."""

    app: Optional[Application] = None
    _polling_task: Optional[asyncio.Task] = None

    @classmethod
    def build_application(cls) -> Optional[Application]:
        """Initializes python-telegram-bot application and registers handlers."""
        if not settings.TELEGRAM_BOT_TOKEN or settings.TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
            logger.warning(
                "TELEGRAM_BOT_TOKEN is not configured. Telegram bot features will remain idle until token is set."
            )
            return None

        try:
            builder = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN)
            cls.app = builder.build()

            # Register Global Error Handler
            cls.app.add_error_handler(global_telegram_error_handler)

            # Register Handlers
            # Mandatory /start onboarding handler
            cls.app.add_handler(CommandHandler("start", start_handler))
            
            # Conversational Text Handler (Zero-Command UI)
            cls.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
            
            # Multimodal Voice Note Handler
            cls.app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_message_handler))
            
            # Multimodal Photo & Chart Handler
            cls.app.add_handler(MessageHandler(filters.PHOTO, photo_message_handler))

            # Document / PDF / Image Document Handler
            cls.app.add_handler(MessageHandler(filters.Document.ALL, document_upload_handler))

            # Connect bot instance to scheduler for proactive message pushes
            SchedulerService.set_bot_instance(cls.app.bot)
            logger.info("Telegram Bot application successfully built and handlers registered.")
            return cls.app
        except Exception as e:
            logger.error(f"Error building Telegram bot application: {e}")
            return None

    @classmethod
    async def start_bot(cls) -> None:
        """Starts the bot in either polling or webhook mode."""
        if cls.app is None:
            cls.build_application()

        if cls.app is None:
            logger.warning("Skipping Telegram Bot startup: Application not built.")
            return

        try:
            # Clear any preexisting webhook to ensure polling receives messages
            try:
                await cls.app.bot.delete_webhook(drop_pending_updates=False)
            except Exception as w_err:
                logger.warning(f"Note on webhook check: {w_err}")

            await cls.app.initialize()
            
            if settings.TELEGRAM_MODE == "polling":
                logger.info("Starting Telegram Bot in POLLING mode...")
                await cls.app.start()
                await cls.app.updater.start_polling(drop_pending_updates=False)
                logger.info("Telegram Bot polling started successfully and listening for messages.")
            
            elif settings.TELEGRAM_MODE == "webhook":
                logger.info(f"Setting Telegram Webhook to {settings.TELEGRAM_WEBHOOK_URL}...")
                await cls.app.bot.set_webhook(
                    url=settings.TELEGRAM_WEBHOOK_URL,
                    secret_token=settings.TELEGRAM_WEBHOOK_SECRET if settings.TELEGRAM_WEBHOOK_SECRET else None
                )
                await cls.app.start()
                logger.info("Telegram Webhook registered successfully.")

        except Exception as e:
            logger.error(f"Error starting Telegram bot: {e}", exc_info=True)

    @classmethod
    async def stop_bot(cls) -> None:
        """Stops the bot gracefully."""
        if cls.app is not None:
            logger.info("Stopping Telegram Bot...")
            try:
                if cls.app.updater and cls.app.updater.running:
                    await cls.app.updater.stop()
                if cls.app.running:
                    await cls.app.stop()
                await cls.app.shutdown()
                logger.info("Telegram Bot shutdown complete.")
            except Exception as e:
                logger.error(f"Error during Telegram bot shutdown: {e}")
