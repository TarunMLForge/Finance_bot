"""
Atlas AI Financial Assistant - Yahoo Finance Service
Provides live and historical stock data, valuation metrics, and financial summaries.
"""

import asyncio
from typing import Dict, Any, Optional
import yfinance as yf
from app.core.logger import logger


class YFinanceService:
    """Async wrapper for Yahoo Finance market data."""

    @staticmethod
    def _fetch_stock_price_sync(ticker: str) -> Dict[str, Any]:
        """Synchronously fetches live stock price and daily performance."""
        ticker = ticker.strip().upper()
        t = yf.Ticker(ticker)
        info = t.info or {}
        
        # Determine price (handle different market sessions)
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("ask")
            or info.get("previousClose")
        )
        previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        
        if current_price is None:
            # Fallback to history
            hist = t.history(period="2d")
            if not hist.empty:
                current_price = float(hist["Close"].iloc[-1])
                if len(hist) > 1:
                    previous_close = float(hist["Close"].iloc[-2])
                else:
                    previous_close = float(hist["Open"].iloc[-1])
                    
        if current_price is None:
            return {
                "success": False,
                "ticker": ticker,
                "error": f"No market data found for ticker '{ticker}'."
            }

        change_abs = None
        change_pct = None
        if previous_close and previous_close > 0:
            change_abs = round(current_price - previous_close, 2)
            change_pct = round(((current_price - previous_close) / previous_close) * 100, 2)

        return {
            "success": True,
            "ticker": ticker,
            "company_name": info.get("shortName") or info.get("longName") or ticker,
            "currency": info.get("currency", "USD"),
            "current_price": round(float(current_price), 2),
            "previous_close": round(float(previous_close), 2) if previous_close else None,
            "change_abs": change_abs,
            "change_pct": change_pct,
            "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
            "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "market_state": info.get("marketState", "REGULAR"),
        }

    @classmethod
    async def get_stock_price(cls, ticker: str) -> Dict[str, Any]:
        """Async fetch of live stock price."""
        try:
            return await asyncio.to_thread(cls._fetch_stock_price_sync, ticker)
        except Exception as e:
            logger.error(f"Error fetching stock price for {ticker}: {e}")
            return {
                "success": False,
                "ticker": ticker.upper(),
                "error": f"Failed to retrieve price for {ticker}: {str(e)}"
            }

    @staticmethod
    def _fetch_financial_summary_sync(ticker: str) -> Dict[str, Any]:
        """Fetches fundamental financial metrics and valuation ratios."""
        ticker = ticker.strip().upper()
        t = yf.Ticker(ticker)
        info = t.info or {}

        def format_currency_number(val: Optional[float]) -> Optional[str]:
            if val is None:
                return None
            if abs(val) >= 1e12:
                return f"${val / 1e12:.2f}T"
            if abs(val) >= 1e9:
                return f"${val / 1e9:.2f}B"
            if abs(val) >= 1e6:
                return f"${val / 1e6:.2f}M"
            return f"${val:,.2f}"

        market_cap = info.get("marketCap")
        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        peg_ratio = info.get("pegRatio")
        revenue = info.get("totalRevenue")
        revenue_growth = info.get("revenueGrowth")
        profit_margins = info.get("profitMargins")
        debt_to_equity = info.get("debtToEquity")
        fifty_two_week_high = info.get("fiftyTwoWeekHigh")
        fifty_two_week_low = info.get("fiftyTwoWeekLow")
        target_mean_price = info.get("targetMeanPrice")
        recommendation_key = info.get("recommendationKey")

        return {
            "success": True,
            "ticker": ticker,
            "company_name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": format_currency_number(market_cap),
            "market_cap_raw": market_cap,
            "trailing_pe": round(trailing_pe, 2) if trailing_pe else None,
            "forward_pe": round(forward_pe, 2) if forward_pe else None,
            "peg_ratio": round(peg_ratio, 2) if peg_ratio else None,
            "total_revenue": format_currency_number(revenue),
            "revenue_growth_pct": round(revenue_growth * 100, 2) if revenue_growth is not None else None,
            "profit_margin_pct": round(profit_margins * 100, 2) if profit_margins is not None else None,
            "debt_to_equity": debt_to_equity,
            "52_week_high": fifty_two_week_high,
            "52_week_low": fifty_two_week_low,
            "analyst_target_mean": target_mean_price,
            "analyst_rating": recommendation_key.replace("_", " ").title() if recommendation_key else None,
            "summary": info.get("longBusinessSummary", "")[:300] + "..." if info.get("longBusinessSummary") else ""
        }

    @classmethod
    async def get_financial_summary(cls, ticker: str) -> Dict[str, Any]:
        """Async fetch of fundamental valuation & growth summary."""
        try:
            return await asyncio.to_thread(cls._fetch_financial_summary_sync, ticker)
        except Exception as e:
            logger.error(f"Error fetching financial summary for {ticker}: {e}")
            return {
                "success": False,
                "ticker": ticker.upper(),
                "error": f"Failed to retrieve financial summary for {ticker}: {str(e)}"
            }

    @staticmethod
    def _fetch_historical_movement_sync(ticker: str, period: str = "5d") -> Dict[str, Any]:
        """Fetches historical price change over a given period."""
        ticker = ticker.strip().upper()
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty:
            return {"success": False, "ticker": ticker, "error": "No historical data available."}
        
        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        change_pct = round(((end_price - start_price) / start_price) * 100, 2)

        return {
            "success": True,
            "ticker": ticker,
            "period": period,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "change_pct": change_pct,
            "highest": round(float(hist["High"].max()), 2),
            "lowest": round(float(hist["Low"].min()), 2),
        }

    @classmethod
    async def get_historical_movement(cls, ticker: str, period: str = "5d") -> Dict[str, Any]:
        """Async fetch of historical price trends."""
        try:
            return await asyncio.to_thread(cls._fetch_historical_movement_sync, ticker, period)
        except Exception as e:
            logger.error(f"Error fetching history for {ticker}: {e}")
            return {"success": False, "ticker": ticker.upper(), "error": str(e)}
