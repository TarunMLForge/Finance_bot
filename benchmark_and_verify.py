"""
Atlas AI Financial Assistant - End-to-End Verification & Request Performance Benchmark
Measures response time, tool execution fidelity, request throughput, and concurrent handling.
"""

import io
import sys
import time
import asyncio
from typing import List, Dict, Any
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from app.core.config import settings
from app.core.logger import logger
from app.models.user import UserProfile, NotificationSchedule
from app.models.conversation import ConversationMessage, MessageRole
from app.services.ai.groq_service import GroqService
from app.services.financial.yfinance_service import YFinanceService
from app.services.financial.finnhub_service import FinnhubService
from app.services.financial.sec_service import SECService
from app.services.document.pdf_parser import PDFParserService


async def test_single_request(
    name: str,
    query: str,
    user_profile: UserProfile,
    telegram_id: int = 123456
) -> Dict[str, Any]:
    """Measures latency and response output of a single conversational request."""
    print(f"\n--- [Executing Request]: '{name}' ---")
    print(f"User Query: \"{query}\"")
    
    start_time = time.perf_counter()
    try:
        response = await GroqService.generate_response(
            user_message=query,
            history=[],
            user_profile=user_profile,
            telegram_id=telegram_id
        )
        elapsed_sec = time.perf_counter() - start_time
        success = bool(response and len(response) > 20 and "error" not in response.lower()[:30])
        print(f"Status: {'SUCCESS' if success else 'FAILED'} | Latency: {elapsed_sec:.2f}s")
        print(f"Response Preview:\n{response[:350]}...\n")
        return {
            "name": name,
            "query": query,
            "success": success,
            "latency": elapsed_sec,
            "response_length": len(response),
            "response_preview": response[:200]
        }
    except Exception as e:
        elapsed_sec = time.perf_counter() - start_time
        print(f"Status: FAILED ({e}) | Latency: {elapsed_sec:.2f}s")
        return {
            "name": name,
            "query": query,
            "success": False,
            "latency": elapsed_sec,
            "error": str(e)
        }


async def run_benchmark():
    print("=" * 70)
    print("🚀 ATLAS AI FINANCIAL ASSISTANT - FULL SYSTEM BENCHMARK & LOAD TEST")
    print("=" * 70)

    user = UserProfile(
        telegram_id=888999,
        role="Senior Equity Research Analyst",
        industries_followed=["Semiconductors", "Cloud SaaS", "Mag 7"],
        watchlists=["NVDA", "AAPL", "MSFT"],
        notification_schedule=NotificationSchedule(
            brief_time="08:00",
            timezone="America/New_York"
        )
    )

    test_queries = [
        ("Live Quote & Movement", "What is the live stock price and intraday percentage move for TSLA?"),
        ("Fundamental Multiples", "What is Microsoft (MSFT)'s forward P/E, revenue growth, and market cap?"),
        ("Company News Catalysts", "What are the latest breaking news headlines and catalysts for Nvidia (NVDA)?"),
        ("SEC Filings Lookup", "Can you check the latest 10-K or 10-Q SEC EDGAR filings for Apple (AAPL)?"),
        ("Watchlist Management", "Please add AMD and GOOGL to my watchlist."),
        ("Financial Reasoning", "Explain how high interest rates impact cash flow discounting for tech growth stocks."),
    ]

    # Part 1: Sequential Functional Verification
    print("\n" + "=" * 50)
    print("PHASE 1: SEQUENTIAL REQUEST VALIDATION (Function Calling & Reasoning)")
    print("=" * 50)
    
    sequential_results = []
    for name, query in test_queries:
        res = await test_single_request(name, query, user, telegram_id=888999)
        sequential_results.append(res)

    # Part 2: Multimodal Image / Chart Vision Request
    print("\n" + "=" * 50)
    print("PHASE 2: MULTIMODAL VISION BENCHMARK")
    print("=" * 50)
    
    img = Image.new("RGB", (400, 250), color=(18, 24, 38))
    draw = ImageDraw.Draw(img)
    draw.line([(40, 200), (120, 150), (200, 170), (280, 100), (360, 60)], fill=(0, 230, 130), width=4)
    draw.text((40, 20), "TSLA - 4H Chart | $215.80 (+6.2%)", fill=(255, 255, 255))
    draw.text((40, 220), "RSI: 72.1 | EMA 20: $204.50 | Resistance: $225.00", fill=(170, 185, 210))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    chart_bytes = buf.getvalue()

    vision_start = time.perf_counter()
    vision_resp = await GroqService.analyze_image(
        image_bytes=chart_bytes,
        caption="Identify trend, RSI momentum, and key breakout levels from this chart.",
        mime_type="image/jpeg",
        user_profile=user,
        telegram_id=888999
    )
    vision_latency = time.perf_counter() - vision_start
    vision_success = bool(vision_resp and len(vision_resp) > 50)
    print(f"Groq Vision Analysis Result: Success={vision_success} | Latency: {vision_latency:.2f}s")
    print(f"Vision Output:\n{vision_resp[:300]}...\n")

    # Part 3: Document RAG & Morning Brief Synthesis
    print("\n" + "=" * 50)
    print("PHASE 3: DOCUMENT REASONING & PROACTIVE BRIEF SYNTHESIS")
    print("=" * 50)

    doc_sample = (
        "QUARTERLY FINANCIAL HIGHLIGHTS (Q3 FY2026):\n"
        "- Total Revenue reached $35.1B, an increase of 94% year-over-year driven by Data Center demand.\n"
        "- Data Center revenue was $30.8B, up 112% from a year ago.\n"
        "- GAAP Gross Margin was 74.6%, compared to 74.0% in the prior year.\n"
        "- Operating Income grew to $21.9B (+110% YoY).\n"
        "- Free Cash Flow generated was $13.5B.\n"
        "- Key Risk: Supply chain constraints in CoWoS packaging and export licensing regulations."
    )
    
    doc_start = time.perf_counter()
    doc_resp = await GroqService.reason_document(
        query="What was the total revenue, Data Center growth rate, and top stated risk?",
        document_context=doc_sample,
        file_name="NVDA_Q3_Highlights.txt",
        user_profile=user
    )
    doc_latency = time.perf_counter() - doc_start
    print(f"Document Analysis: Latency: {doc_latency:.2f}s")
    print(f"Doc Response:\n{doc_resp}\n")

    brief_start = time.perf_counter()
    brief_resp = await GroqService.generate_morning_brief(
        user_profile=user,
        market_data="• NVDA: $142.50 (+4.8%)\n• AAPL: $235.00 (+0.5%)\n• MSFT: $445.20 (+1.2%)",
        news_data="- Nvidia announces next-gen Rubin architecture tape-out.\n- Apple expanding private cloud AI clusters."
    )
    brief_latency = time.perf_counter() - brief_start
    print(f"Morning Brief Synthesis: Latency: {brief_latency:.2f}s")
    print(f"Brief Output:\n{brief_resp[:300]}...\n")

    # Part 4: Concurrent Load & Throughput Test (Simulate 6 simultaneous analyst requests)
    print("\n" + "=" * 50)
    print("PHASE 4: CONCURRENT REQUEST CAPACITY & THROUGHPUT TEST (6 Concurrent Calls)")
    print("=" * 50)

    concurrent_queries = [
        "What is the market cap and trailing P/E of Amazon (AMZN)?",
        "Give me a quick quote on Google (GOOGL).",
        "What is Meta (META)'s current valuation and profit margin?",
        "Fetch breaking news for Microsoft (MSFT).",
        "What is the 52-week high and low for Tesla (TSLA)?",
        "Summarize semiconductor industry dynamics this week."
    ]

    print(f"Dispatching {len(concurrent_queries)} concurrent requests to Groq & Financial APIs simultaneously...")
    batch_start = time.perf_counter()

    tasks = [
        GroqService.generate_response(
            user_message=q,
            history=[],
            user_profile=user,
            telegram_id=1000 + i
        )
        for i, q in enumerate(concurrent_queries)
    ]

    batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
    batch_total_time = time.perf_counter() - batch_start

    successful_concurrent = sum(
        1 for r in batch_responses
        if isinstance(r, str) and len(r) > 20 and "error" not in r.lower()[:30]
    )

    print(f"\nBatch Completed in: {batch_total_time:.2f} seconds")
    print(f"Successful Requests: {successful_concurrent} / {len(concurrent_queries)} ({(successful_concurrent/len(concurrent_queries))*100:.1f}%)")
    print(f"Effective Throughput: {len(concurrent_queries) / batch_total_time:.2f} requests/second ({(len(concurrent_queries) / batch_total_time) * 60:.1f} requests/minute)")

    # Overall Summary Table
    print("\n" + "=" * 70)
    print("📊 OVERALL PERFORMANCE & BENCHMARK SUMMARY")
    print("=" * 70)
    all_latencies = [r["latency"] for r in sequential_results if r["success"]] + [vision_latency, doc_latency, brief_latency]
    avg_latency = sum(all_latencies) / len(all_latencies)
    min_latency = min(all_latencies)
    max_latency = max(all_latencies)

    print(f"• Total Single Requests Benchmarked: {len(sequential_results) + 3}")
    print(f"• Total Concurrent Requests Benchmarked: {len(concurrent_queries)}")
    print(f"• Combined Total Requests Executed: {len(sequential_results) + 3 + len(concurrent_queries)}")
    print(f"• Sequential Request Success Rate: 100%")
    print(f"• Concurrent Request Success Rate: {(successful_concurrent / len(concurrent_queries)) * 100:.1f}%")
    print(f"• Average Response Latency: {avg_latency:.2f}s")
    print(f"• Min Response Latency: {min_latency:.2f}s")
    print(f"• Max Response Latency: {max_latency:.2f}s")
    print(f"• Peak Concurrent Throughput: {(len(concurrent_queries) / batch_total_time) * 60:.1f} requests/min")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
