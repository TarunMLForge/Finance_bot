"""
Atlas AI Financial Assistant - Background Jobs & Proactive Intelligence
Manages scheduled Morning Briefs and real-time market movement alerts adhering to 'Silence is Golden'.
"""

import asyncio
from datetime import datetime, timezone
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from app.core.config import settings
from app.core.logger import logger
from app.core.database import get_users_collection
from app.models.user import UserProfile
from app.services.financial.yfinance_service import YFinanceService
from app.services.financial.finnhub_service import FinnhubService
from app.services.ai.groq_service import GroqService


class SchedulerService:
    """Async background scheduler for proactive financial intelligence."""

    scheduler: AsyncIOScheduler = None
    _bot_instance = None  # Reference to Telegram bot for proactive message pushes

    @classmethod
    def set_bot_instance(cls, bot_instance) -> None:
        """Sets the Telegram bot instance used for pushing notifications."""
        cls._bot_instance = bot_instance

    @classmethod
    def start(cls) -> None:
        """Initializes and starts the APScheduler jobs."""
        if cls.scheduler is None:
            cls.scheduler = AsyncIOScheduler(timezone=settings.DEFAULT_TIMEZONE)
            
            # Job 1: Check every 15 minutes for users due for their Morning Brief
            cls.scheduler.add_job(
                cls.check_and_dispatch_morning_briefs,
                trigger=IntervalTrigger(minutes=15),
                id="morning_brief_checker",
                name="Check and dispatch morning briefs to users",
                replace_existing=True
            )
            
            # Job 2: Check hourly for significant market volatility (>5% movement)
            cls.scheduler.add_job(
                cls.check_market_volatility_alerts,
                trigger=IntervalTrigger(hours=1),
                id="market_volatility_checker",
                name="Check watchlist volatility and alert if >5%",
                replace_existing=True
            )
            
            cls.scheduler.start()
            logger.info("APScheduler started with Morning Brief and Market Volatility jobs.")

    @classmethod
    def shutdown(cls) -> None:
        """Stops the scheduler gracefully."""
        if cls.scheduler and cls.scheduler.running:
            cls.scheduler.shutdown(wait=False)
            logger.info("APScheduler shutdown complete.")

    @classmethod
    async def send_telegram_message(cls, telegram_id: int, text: str) -> bool:
        """Pushes a proactive message to a user via Telegram."""
        if cls._bot_instance is None:
            logger.warning(f"Cannot send proactive message to {telegram_id}: Bot instance not attached.")
            return False

        try:
            await cls._bot_instance.send_message(
                chat_id=telegram_id,
                text=text,
                parse_mode="Markdown"
            )
            logger.info(f"Proactive message successfully sent to Telegram ID {telegram_id}.")
            return True
        except Exception as e:
            # Fallback to plain text if Markdown parsing fails
            try:
                await cls._bot_instance.send_message(
                    chat_id=telegram_id,
                    text=text
                )
                return True
            except Exception as inner_e:
                logger.error(f"Failed to send Telegram message to {telegram_id}: {inner_e}")
                return False

    @classmethod
    async def check_and_dispatch_morning_briefs(cls) -> None:
        """Evaluates all users and sends morning brief if user's local time matches brief_time."""
        users_col = get_users_collection()
        if users_col is None:
            return

        now_utc = datetime.now(timezone.utc)
        logger.info("Evaluating scheduled morning briefs...")

        try:
            cursor = users_col.find({"notification_schedule.morning_brief_enabled": True})
            async for doc in cursor:
                user = UserProfile.from_mongo(doc)
                user_tz_str = user.notification_schedule.timezone or "UTC"
                try:
                    user_tz = pytz.timezone(user_tz_str)
                except Exception:
                    user_tz = pytz.UTC

                user_now = now_utc.astimezone(user_tz)
                current_time_str = user_now.strftime("%H:%M")
                target_brief_time = user.notification_schedule.brief_time or "08:00"

                # Check if already sent today
                last_sent = user.notification_schedule.last_brief_sent_at
                already_sent_today = False
                if last_sent:
                    last_sent_user_tz = last_sent.astimezone(user_tz) if last_sent.tzinfo else user_tz.localize(last_sent)
                    if last_sent_user_tz.date() == user_now.date():
                        already_sent_today = True

                # Match within a 20-minute window
                target_hour, target_min = map(int, target_brief_time.split(":"))
                if not already_sent_today and user_now.hour == target_hour and abs(user_now.minute - target_min) < 20:
                    logger.info(f"Triggering Morning Brief for user {user.telegram_id} ({user.role}) at local time {current_time_str}.")
                    await cls.generate_and_send_user_brief(user)

        except Exception as e:
            logger.error(f"Error checking morning briefs: {e}", exc_info=True)

    @classmethod
    async def generate_and_send_user_brief(cls, user: UserProfile) -> None:
        """Fetches watchlist data and pushes synthesized morning brief."""
        watchlist = user.watchlists if user.watchlists else ["SPY", "QQQ", "AAPL"]
        
        # 1. Fetch live quotes for watchlist
        quotes_summary = []
        for ticker in watchlist[:6]:
            quote = await YFinanceService.get_stock_price(ticker)
            if quote.get("success"):
                change_sign = "+" if (quote.get("change_pct") or 0) >= 0 else ""
                quotes_summary.append(
                    f"• {ticker}: ${quote.get('current_price')} ({change_sign}{quote.get('change_pct')}%)"
                )

        market_data_text = "\n".join(quotes_summary) if quotes_summary else "Market data unavailable."

        # 2. Fetch top news
        news_summary = []
        top_ticker = watchlist[0] if watchlist else "SPY"
        news_res = await FinnhubService.get_company_news(top_ticker, limit=3)
        if news_res.get("success") and news_res.get("articles"):
            for art in news_res["articles"][:3]:
                news_summary.append(f"- {art.get('headline')} ({art.get('source')})")

        news_data_text = "\n".join(news_summary) if news_summary else "No major catalysts reported overnight."

        # 3. Synthesize via Groq
        brief_text = await GroqService.generate_morning_brief(user, market_data_text, news_data_text)

        # 4. Dispatch Telegram Message
        sent = await cls.send_telegram_message(user.telegram_id, brief_text)
        if sent:
            users_col = get_users_collection()
            if users_col is not None:
                await users_col.update_one(
                    {"telegram_id": user.telegram_id},
                    {"$set": {"notification_schedule.last_brief_sent_at": datetime.now(timezone.utc)}}
                )

    @classmethod
    async def check_market_volatility_alerts(cls) -> None:
        """
        Hourly Market Volatility Monitor.
        'Silence is Golden': Only sends an alert if a stock moves > 5%.
        """
        users_col = get_users_collection()
        if users_col is None:
            return

        logger.info("Running market volatility scanner (Silence is Golden)...")
        threshold = settings.MARKET_ALERT_THRESHOLD_PCT

        try:
            cursor = users_col.find({"notification_schedule.market_alerts_enabled": True})
            async for doc in cursor:
                user = UserProfile.from_mongo(doc)
                if not user.watchlists:
                    continue

                for ticker in user.watchlists:
                    quote = await YFinanceService.get_stock_price(ticker)
                    if not quote.get("success"):
                        continue

                    change_pct = quote.get("change_pct")
                    if change_pct is not None and abs(change_pct) >= threshold:
                        # Major swing detected!
                        logger.info(f"Volatility trigger: {ticker} moved {change_pct}% for user {user.telegram_id}.")
                        
                        # Fetch catalyst
                        news_res = await FinnhubService.get_company_news(ticker, limit=1)
                        top_headline = "High market trading volume and volatility."
                        if news_res.get("success") and news_res.get("articles"):
                            top_headline = news_res["articles"][0].get("headline", top_headline)

                        alert_msg = await GroqService.generate_market_alert(
                            user_profile=user,
                            ticker=ticker,
                            current_price=quote.get("current_price", 0.0),
                            currency=quote.get("currency", "USD"),
                            change_pct=change_pct,
                            headline=top_headline
                        )

                        await cls.send_telegram_message(user.telegram_id, alert_msg)
                    else:
                        # Silence is Golden: No alert sent if within threshold
                        pass

        except Exception as e:
            logger.error(f"Error during market volatility scan: {e}")
