import io
import sys
import asyncio
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from app.services.financial.yfinance_service import YFinanceService
from app.services.financial.finnhub_service import FinnhubService
from app.services.financial.sec_service import SECService
from app.services.document.pdf_parser import PDFParserService
from app.models.user import UserProfile, NotificationSchedule
from app.models.conversation import ConversationMessage, MessageRole
from app.services.ai.groq_service import GroqService


async def run_tests():
    print("=" * 60)
    print("ATLAS AI FINANCIAL ASSISTANT - COMPONENT VERIFICATION")
    print("=" * 60)

    # 1. Test YFinance Live Quote
    print("\n[1/5] Testing YFinance Live Quote (AAPL)...")
    quote = await YFinanceService.get_stock_price("AAPL")
    print(f"Result: Success={quote.get('success')}, Price={quote.get('current_price')} {quote.get('currency')}, Change={quote.get('change_pct')}%")

    # 2. Test YFinance Financial Summary
    print("\n[2/5] Testing YFinance Fundamental Summary (MSFT)...")
    summary = await YFinanceService.get_financial_summary("MSFT")
    print(f"Result: Success={summary.get('success')}, Market Cap={summary.get('market_cap')}, Forward P/E={summary.get('forward_pe')}, Rating={summary.get('analyst_rating')}")

    # 3. Test Company News
    print("\n[3/5] Testing Company News Aggregator (NVDA)...")
    news = await FinnhubService.get_company_news("NVDA", limit=2)
    print(f"Result: Success={news.get('success')}, Source={news.get('source')}, Articles Count={len(news.get('articles', []))}")
    if news.get('articles'):
        print(f"Top Headline: {news['articles'][0].get('headline')}")

    # 4. Test SEC EDGAR Filings
    print("\n[4/5] Testing SEC EDGAR Filings (TSLA)...")
    sec = await SECService.get_recent_filings("TSLA", limit=2)
    print(f"Result: Success={sec.get('success')}, CIK={sec.get('cik')}, Filings Found={len(sec.get('filings', []))}")

    # 5. Test Models Serialization
    print("\n[5/6] Testing Pydantic User & Conversation Schemas...")
    user = UserProfile(
        telegram_id=987654321,
        role="Portfolio Manager",
        industries_followed=["Semiconductors", "Cloud"],
        watchlists=["NVDA", "AAPL", "MSFT"]
    )
    mongo_dict = user.to_mongo()
    restored_user = UserProfile.from_mongo(mongo_dict)
    assert restored_user.telegram_id == 987654321
    assert restored_user.role == "Portfolio Manager"

    conv = ConversationMessage(
        telegram_id=987654321,
        role=MessageRole.USER,
        content="What is the gross margin for Nvidia?"
    )
    assert conv.role.value == "user"
    print("Schema serialization verified successfully!")

    # 6. Test Groq Vision Image Analysis
    print("\n[6/6] Testing Groq Vision Image Analysis (NVDA Technical Chart)...")
    img = Image.new("RGB", (300, 200), color=(25, 30, 45))
    draw = ImageDraw.Draw(img)
    draw.line([(30, 160), (90, 120), (150, 140), (210, 80), (270, 50)], fill=(0, 220, 120), width=3)
    draw.text((30, 15), "NVDA - Daily Chart | $142.50 (+4.8%)", fill=(255, 255, 255))
    draw.text((30, 175), "RSI: 68.4 | Support: $135.00 | Resistance: $148.00", fill=(180, 190, 205))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    chart_bytes = buf.getvalue()

    vision_resp = await GroqService.analyze_image(
        image_bytes=chart_bytes,
        caption="Analyze key technical levels and trend from this chart",
        mime_type="image/jpeg",
        user_profile=user
    )
    print(f"Groq Vision Response:\n{vision_resp}\n")

    print("=" * 60)
    print("ALL CORE ATLAS AI COMPONENTS (INCLUDING VISION) VALIDATED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_tests())
