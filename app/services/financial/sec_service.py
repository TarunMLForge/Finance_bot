"""
Atlas AI Financial Assistant - SEC EDGAR Regulatory Filings Service
Provides free access to SEC 10-K, 10-Q, and 8-K filings for public companies.
"""

from typing import Dict, Any, List
import httpx
from app.core.config import settings
from app.core.logger import logger


class SECService:
    """Async client for querying public SEC EDGAR regulatory submissions."""

    BASE_URL = "https://data.sec.gov"
    CIK_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

    _cik_cache: Dict[str, str] = {}

    @classmethod
    async def _get_cik_for_ticker(cls, ticker: str) -> str:
        """Resolves a stock ticker to a 10-digit zero-padded SEC CIK number."""
        ticker = ticker.strip().upper()
        if ticker in cls._cik_cache:
            return cls._cik_cache[ticker]

        headers = {"User-Agent": settings.SEC_USER_AGENT}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(cls.CIK_TICKER_URL, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.values():
                        sym = str(item.get("ticker", "")).upper()
                        cik = str(item.get("cik_str", "")).zfill(10)
                        cls._cik_cache[sym] = cik
                    return cls._cik_cache.get(ticker, "")
        except Exception as e:
            logger.warning(f"Failed to fetch SEC CIK ticker mapping: {e}")
        return ""

    @classmethod
    async def get_recent_filings(cls, ticker: str, filing_types: List[str] = None, limit: int = 5) -> Dict[str, Any]:
        """Fetches recent SEC filings (e.g. 10-K, 10-Q, 8-K) for a given ticker."""
        if filing_types is None:
            filing_types = ["10-K", "10-Q", "8-K"]

        ticker = ticker.strip().upper()
        cik = await cls._get_cik_for_ticker(ticker)
        if not cik:
            return {
                "success": False,
                "ticker": ticker,
                "error": f"Could not find SEC CIK identifier for ticker '{ticker}'."
            }

        headers = {"User-Agent": settings.SEC_USER_AGENT}
        submissions_url = f"{cls.BASE_URL}/submissions/CIK{cik}.json"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(submissions_url, headers=headers)
                if resp.status_code != 200:
                    return {
                        "success": False,
                        "ticker": ticker,
                        "error": f"SEC EDGAR returned status code {resp.status_code}."
                    }

                data = resp.json()
                recent = data.get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                filing_dates = recent.get("filingDate", [])
                accession_numbers = recent.get("accessionNumber", [])
                primary_docs = recent.get("primaryDocument", [])
                descriptions = recent.get("primaryDocDescription", [])

                matching_filings = []
                for i in range(len(forms)):
                    form = forms[i]
                    if any(ft in form for ft in filing_types):
                        acc_no_clean = accession_numbers[i].replace("-", "")
                        filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_clean}/{primary_docs[i]}"
                        matching_filings.append({
                            "form": form,
                            "filing_date": filing_dates[i],
                            "description": descriptions[i] or f"{form} filing",
                            "filing_url": filing_url
                        })
                        if len(matching_filings) >= limit:
                            break

                return {
                    "success": True,
                    "ticker": ticker,
                    "cik": cik,
                    "company_name": data.get("name", ticker),
                    "filings": matching_filings
                }
        except Exception as e:
            logger.error(f"Error fetching SEC filings for {ticker}: {e}")
            return {
                "success": False,
                "ticker": ticker,
                "error": f"Failed to retrieve SEC filings: {str(e)}"
            }
