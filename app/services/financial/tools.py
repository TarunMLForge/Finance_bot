"""
Atlas AI Financial Assistant - Tool Registry & Execution Bindings
Registers executable tools for Groq function/tool calling.
"""

from typing import Dict, Any, List, Optional
from app.services.financial.yfinance_service import YFinanceService
from app.services.financial.finnhub_service import FinnhubService
from app.services.financial.sec_service import SECService
from app.core.database import get_users_collection
from app.core.logger import logger


# Tool Definitions for Groq Function Calling (OpenAI Tool Standard)
GROQ_FINANCIAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Fetch real-time stock quote, daily price change, volume, and day range for a given stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. AAPL, NVDA, TSLA, MSFT)."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_summary",
            "description": "Fetch comprehensive fundamental financial valuation metrics, including Market Cap, Trailing and Forward P/E, Revenue, Revenue Growth, Profit Margins, Debt-to-Equity, 52-Week Range, and Analyst Consensus Rating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. NVDA, AMZN, GOOGL)."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_news",
            "description": "Retrieve the latest news articles, major headlines, and press coverage for a given company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. AAPL, TSLA)."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of news articles to retrieve (default 5)."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_news",
            "description": "Fetch top macroeconomic and general financial market headlines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Market news category (e.g. 'general', 'forex', 'crypto', 'merger')."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sec_filings",
            "description": "Query official SEC EDGAR 10-K (Annual Report), 10-Q (Quarterly Report), and 8-K (Current Event) regulatory filings for a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. AAPL, META)."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_watchlist",
            "description": "Add or remove a stock ticker from the user's personal watchlist for daily morning briefs and price movement tracking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove"],
                        "description": "'add' to add a ticker, or 'remove' to remove a ticker."
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. NVDA, AAPL)."
                    }
                },
                "required": ["action", "ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_preferences",
            "description": "Update user's profile details such as their professional role, industries followed, or morning brief delivery time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "description": "User's financial role (e.g., Equity Analyst, Portfolio Manager, Retail Trader, CFO)."
                    },
                    "industries_followed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of sectors/industries of interest (e.g., ['Semiconductors', 'AI Infrastructure', 'Energy'])."
                    },
                    "brief_time": {
                        "type": "string",
                        "description": "Preferred morning brief time in 24-hr format (e.g. '08:30')."
                    },
                    "timezone": {
                        "type": "string",
                        "description": "User's timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Kolkata', 'UTC')."
                    }
                }
            }
        }
    }
]


class FinancialToolExecutor:
    """Dispatches tool execution calls from Groq to appropriate services."""

    @classmethod
    async def execute(cls, tool_name: str, args: Dict[str, Any], telegram_id: int) -> Dict[str, Any]:
        """Executes a named tool and returns the structured result."""
        logger.info(f"Executing tool '{tool_name}' for telegram_id={telegram_id} with args={args}")
        try:
            if tool_name == "get_stock_price":
                ticker = args.get("ticker", "")
                return await YFinanceService.get_stock_price(ticker)

            elif tool_name == "get_financial_summary":
                ticker = args.get("ticker", "")
                return await YFinanceService.get_financial_summary(ticker)

            elif tool_name == "get_company_news":
                ticker = args.get("ticker", "")
                limit = int(args.get("limit", 5))
                return await FinnhubService.get_company_news(ticker, limit=limit)

            elif tool_name == "get_market_news":
                category = args.get("category", "general")
                return await FinnhubService.get_market_news(category=category)

            elif tool_name == "get_sec_filings":
                ticker = args.get("ticker", "")
                return await SECService.get_recent_filings(ticker)

            elif tool_name == "manage_watchlist":
                action = args.get("action", "").lower()
                ticker = args.get("ticker", "").upper().strip()
                return await cls._manage_watchlist(telegram_id, action, ticker)

            elif tool_name == "update_user_preferences":
                return await cls._update_user_preferences(telegram_id, args)

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": f"Failed to execute {tool_name}: {str(e)}"}

    @staticmethod
    async def _manage_watchlist(telegram_id: int, action: str, ticker: str) -> Dict[str, Any]:
        """Adds or removes a ticker from the user's watchlist in MongoDB or in-memory fallback."""
        users_col = get_users_collection()
        if users_col is not None:
            try:
                if action == "add":
                    await users_col.update_one(
                        {"telegram_id": telegram_id},
                        {"$addToSet": {"watchlists": ticker}},
                        upsert=True
                    )
                    user = await users_col.find_one({"telegram_id": telegram_id})
                    return {
                        "success": True,
                        "action": "added",
                        "ticker": ticker,
                        "current_watchlist": user.get("watchlists", []) if user else [ticker]
                    }
                elif action == "remove":
                    await users_col.update_one(
                        {"telegram_id": telegram_id},
                        {"$pull": {"watchlists": ticker}}
                    )
                    user = await users_col.find_one({"telegram_id": telegram_id})
                    return {
                        "success": True,
                        "action": "removed",
                        "ticker": ticker,
                        "current_watchlist": user.get("watchlists", []) if user else []
                    }
            except Exception as e:
                logger.warning(f"Failed to update MongoDB watchlist ({e}), using in-memory store.")

        # In-memory fallback
        from app.telegram.handlers import IN_MEMORY_USERS
        user = IN_MEMORY_USERS.get(telegram_id)
        if user:
            if action == "add" and ticker not in user.watchlists:
                user.watchlists.append(ticker)
            elif action == "remove" and ticker in user.watchlists:
                user.watchlists.remove(ticker)
            return {
                "success": True,
                "action": action,
                "ticker": ticker,
                "current_watchlist": user.watchlists
            }
        return {"success": True, "action": action, "ticker": ticker, "current_watchlist": [ticker] if action == "add" else []}

    @staticmethod
    async def _update_user_preferences(telegram_id: int, args: Dict[str, Any]) -> Dict[str, Any]:
        """Updates user profile preferences in MongoDB or in-memory fallback."""
        users_col = get_users_collection()
        update_fields: Dict[str, Any] = {"is_onboarded": True}
        if "role" in args and args["role"]:
            update_fields["role"] = args["role"]
        if "industries_followed" in args and isinstance(args["industries_followed"], list):
            update_fields["industries_followed"] = args["industries_followed"]
        if "brief_time" in args and args["brief_time"]:
            update_fields["notification_schedule.brief_time"] = args["brief_time"]
        if "timezone" in args and args["timezone"]:
            update_fields["notification_schedule.timezone"] = args["timezone"]

        if users_col is not None:
            try:
                await users_col.update_one(
                    {"telegram_id": telegram_id},
                    {"$set": update_fields},
                    upsert=True
                )
                return {"success": True, "updated": update_fields}
            except Exception as e:
                logger.warning(f"Failed to update MongoDB preferences ({e}), updating in-memory.")

        # In-memory fallback
        from app.telegram.handlers import IN_MEMORY_USERS
        user = IN_MEMORY_USERS.get(telegram_id)
        if user:
            if "role" in args and args["role"]:
                user.role = args["role"]
            if "industries_followed" in args and isinstance(args["industries_followed"], list):
                user.industries_followed = args["industries_followed"]
            if "brief_time" in args and args["brief_time"]:
                user.notification_schedule.brief_time = args["brief_time"]
            if "timezone" in args and args["timezone"]:
                user.notification_schedule.timezone = args["timezone"]
            user.is_onboarded = True

        return {"success": True, "updated": update_fields}
