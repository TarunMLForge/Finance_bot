"""
Atlas AI Financial Assistant - Finnhub & Market News Service
Provides live market and company news using Finnhub with graceful Yahoo Finance fallback.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import httpx
import yfinance as yf
from app.core.config import settings
from app.core.logger import logger


class FinnhubService:
    """Async news aggregator utilizing Finnhub API with Yahoo Finance fallback."""

    @classmethod
    async def get_company_news(cls, ticker: str, limit: int = 5) -> Dict[str, Any]:
        """Fetches company-specific news from Finnhub or Yahoo Finance."""
        ticker = ticker.strip().upper()
        
        # 1. Try Finnhub if API key is provided
        if settings.FINNHUB_API_KEY and settings.FINNHUB_API_KEY != "your_finnhub_api_key_here":
            try:
                today = datetime.now(timezone.utc)
                from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
                to_date = today.strftime("%Y-%m-%d")
                
                url = "https://finnhub.io/api/v1/company-news"
                params = {
                    "symbol": ticker,
                    "from": from_date,
                    "to": to_date,
                    "token": settings.FINNHUB_API_KEY
                }
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        articles = resp.json()
                        if articles and isinstance(articles, list):
                            formatted_news = []
                            for item in articles[:limit]:
                                formatted_news.append({
                                    "headline": item.get("headline", "No title"),
                                    "summary": item.get("summary", ""),
                                    "source": item.get("source", "Finnhub"),
                                    "url": item.get("url", ""),
                                    "datetime": datetime.fromtimestamp(
                                        item.get("datetime", 0), tz=timezone.utc
                                    ).strftime("%Y-%m-%d %H:%M UTC") if item.get("datetime") else "Recent"
                                })
                            return {
                                "success": True,
                                "ticker": ticker,
                                "source": "Finnhub",
                                "articles": formatted_news
                            }
            except Exception as e:
                logger.warning(f"Finnhub API request failed for {ticker}: {e}. Falling back to Yahoo Finance.")

        # 2. Fallback to Yahoo Finance news
        try:
            return await asyncio.to_thread(cls._fetch_yahoo_news_sync, ticker, limit)
        except Exception as e:
            logger.error(f"Failed to fetch news for {ticker}: {e}")
            return {
                "success": False,
                "ticker": ticker,
                "error": f"Could not retrieve news for {ticker}: {str(e)}",
                "articles": []
            }

    @staticmethod
    def _fetch_yahoo_news_sync(ticker: str, limit: int = 5) -> Dict[str, Any]:
        """Synchronously extracts news from yfinance ticker object."""
        t = yf.Ticker(ticker)
        raw_news = t.news or []
        formatted = []
        for item in raw_news[:limit]:
            # Yahoo Finance news structure handling (supports both old and new structure)
            content = item.get("content", {}) if "content" in item else item
            title = content.get("title") or item.get("title") or "No title"
            summary = content.get("summary") or item.get("summary") or ""
            publisher = (
                content.get("provider", {}).get("displayName")
                or item.get("publisher")
                or "Yahoo Finance"
            )
            link = (
                content.get("canonicalUrl", {}).get("url")
                or item.get("link")
                or ""
            )
            formatted.append({
                "headline": title,
                "summary": summary,
                "source": publisher,
                "url": link,
                "datetime": "Recent"
            })
            
        return {
            "success": True,
            "ticker": ticker,
            "source": "Yahoo Finance",
            "articles": formatted
        }

    @classmethod
    async def get_market_news(cls, category: str = "general", limit: int = 5) -> Dict[str, Any]:
        """Fetches top macroeconomic and general market news."""
        if settings.FINNHUB_API_KEY and settings.FINNHUB_API_KEY != "your_finnhub_api_key_here":
            try:
                url = "https://finnhub.io/api/v1/news"
                params = {
                    "category": category,
                    "token": settings.FINNHUB_API_KEY
                }
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        articles = resp.json()
                        formatted_news = []
                        for item in articles[:limit]:
                            formatted_news.append({
                                "headline": item.get("headline", ""),
                                "summary": item.get("summary", ""),
                                "source": item.get("source", "Finnhub"),
                                "url": item.get("url", "")
                            })
                        return {
                            "success": True,
                            "category": category,
                            "articles": formatted_news
                        }
            except Exception as e:
                logger.warning(f"Error fetching market news from Finnhub: {e}")

        # Fallback to major index news (e.g. SPY)
        return await cls.get_company_news("SPY", limit=limit)
