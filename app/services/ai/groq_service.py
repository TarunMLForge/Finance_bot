"""
Atlas AI Financial Assistant - Groq AI Service
Integrates Groq ultra-fast LLM inference (Llama-3.3-70b-versatile) and Whisper (whisper-large-v3)
for natural language reasoning, function calling, audio transcription, and document analysis.
"""

import io
import re
import json
import base64
import asyncio
from typing import List, Dict, Any, Optional
from PIL import Image
from groq import AsyncGroq

from app.core.config import settings
from app.core.logger import logger
from app.models.user import UserProfile
from app.models.conversation import ConversationMessage
from app.services.ai.prompts import (
    ATLAS_SYSTEM_PROMPT,
    MORNING_BRIEF_PROMPT_TEMPLATE,
    MARKET_ALERT_PROMPT_TEMPLATE,
    FINANCIAL_IMAGE_SYSTEM_INSTRUCTION,
    DEFAULT_IMAGE_ANALYSIS_QUERY
)
from app.services.financial.tools import GROQ_FINANCIAL_TOOLS, FinancialToolExecutor


class GroqService:
    """Core AI engine managing Groq reasoning, tool dispatch, Whisper audio notes, and documents."""

    _client: Optional[AsyncGroq] = None

    @classmethod
    def get_client(cls) -> Optional[AsyncGroq]:
        """Initializes and returns the AsyncGroq client."""
        if cls._client is None and settings.GROQ_API_KEY and settings.GROQ_API_KEY != "your_groq_api_key_here":
            try:
                cls._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
                logger.info("Groq AI Client initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing Groq Client: {e}")
        return cls._client

    @classmethod
    async def _create_chat_completion_with_retry(
        cls,
        client: AsyncGroq,
        max_retries: int = 3,
        **kwargs
    ) -> Any:
        """Executes a Groq chat completion with automatic backoff and seamless failover to llama-3.1-8b-instant."""
        last_exception = None
        current_kwargs = dict(kwargs)
        
        for attempt in range(max_retries + 1):
            try:
                return await client.chat.completions.create(**current_kwargs)
            except Exception as e:
                last_exception = e
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    # Check if tokens per day or long wait
                    is_tpd = "tokens per day" in err_str.lower() or "tpd" in err_str.lower()
                    match = re.search(r"try again in (\d+\.?\d*)s", err_str, re.IGNORECASE)
                    wait_sec = float(match.group(1)) if match else 5.0

                    # If wait is too long (> 10s) or TPD exceeded, failover immediately to secondary model
                    if (is_tpd or wait_sec > 10.0 or attempt >= 2) and current_kwargs.get("model") != "llama-3.1-8b-instant":
                        logger.warning(
                            f"Primary Groq model rate limited/exhausted ({err_str[:120]}...). "
                            "Failing over to lightning-fast 'llama-3.1-8b-instant'..."
                        )
                        current_kwargs["model"] = "llama-3.1-8b-instant"
                        continue
                    
                    if attempt < max_retries:
                        sleep_time = min(wait_sec + 0.6, 10.0)
                        logger.warning(
                            f"Groq rate limit hit (attempt {attempt+1}/{max_retries+1}). "
                            f"Waiting {sleep_time:.2f}s before retrying..."
                        )
                        await asyncio.sleep(sleep_time)
                        continue
                        
                # On non-429 function call parsing error, also try fallback model if not already using it
                if "tool_use_failed" in err_str and current_kwargs.get("model") != "llama-3.1-8b-instant":
                    current_kwargs["model"] = "llama-3.1-8b-instant"
                    continue

                raise last_exception
        raise last_exception

    @classmethod
    def _build_system_instruction(cls, user_profile: Optional[UserProfile] = None) -> str:
        """Constructs system instruction with user profile context."""
        instruction = ATLAS_SYSTEM_PROMPT
        if user_profile:
            profile_context = (
                f"\n\n### USER CONTEXT:\n"
                f"- User Role: {user_profile.role or 'Finance Professional'}\n"
                f"- Sectors Followed: {', '.join(user_profile.industries_followed) if user_profile.industries_followed else 'General Markets'}\n"
                f"- Active Watchlist: {', '.join(user_profile.watchlists) if user_profile.watchlists else 'None yet'}\n"
                f"- Timezone: {user_profile.notification_schedule.timezone}\n"
            )
            instruction += profile_context
        return instruction

    @classmethod
    async def generate_response(
        cls,
        user_message: str,
        history: List[ConversationMessage],
        user_profile: Optional[UserProfile] = None,
        telegram_id: int = 0
    ) -> str:
        """Processes conversational input with history, tool execution, and Groq synthesis."""
        client = cls.get_client()
        if not client:
            return (
                "⚠️ *Atlas AI Configuration Notice*: Groq API Key is not configured. "
                "Please configure `GROQ_API_KEY` in `.env` to enable ultra-fast autonomous financial reasoning."
            )

        system_instruction = cls._build_system_instruction(user_profile)

        # Build sanitized chat messages (limit history to last 6 messages to prevent context drift)
        messages = [{"role": "system", "content": system_instruction}]
        recent_history = (history or [])[-6:]
        for msg in recent_history:
            role = "user" if msg.role.value == "user" else "assistant"
            # Strip any internal function/think tags from history
            clean_content = cls._clean_think_tags(msg.content)
            clean_content = re.sub(r"<function=.*?</function>", "", clean_content, flags=re.DOTALL).strip()
            if clean_content:
                messages.append({"role": role, "content": clean_content})

        # Append current user prompt
        messages.append({"role": "user", "content": user_message})

        try:
            # Initial call to Groq with tool bindings & retry
            response = await cls._create_chat_completion_with_retry(
                client=client,
                model=settings.GROQ_MODEL,
                messages=messages,
                tools=GROQ_FINANCIAL_TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1024
            )

            response_message = response.choices[0].message

            # Tool Execution Loop (up to 3 turns)
            max_turns = 3
            current_turn = 0

            while response_message.tool_calls and current_turn < max_turns:
                current_turn += 1
                messages.append(response_message)

                for tool_call in response_message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except Exception:
                        fn_args = {}

                    logger.info(f"Groq requested tool execution: {fn_name}({fn_args})")
                    tool_result = await FinancialToolExecutor.execute(fn_name, fn_args, telegram_id)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": json.dumps(tool_result)
                    })

                # Follow-up completion after tool execution
                follow_up_resp = await cls._create_chat_completion_with_retry(
                    client=client,
                    model=settings.GROQ_MODEL,
                    messages=messages,
                    tools=GROQ_FINANCIAL_TOOLS,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=1024
                )
                response_message = follow_up_resp.choices[0].message

            raw_text = response_message.content or "I have processed your request."
            return cls._clean_think_tags(raw_text)

        except Exception as e:
            logger.warning(f"Groq tool-calling exception: {e}, attempting automatic recovery...")
            err_str = str(e)
            
            # Check for failed_generation pseudo-function (e.g., <function=get_financial_summary{"ticker": "TSLA"}</function>)
            match = re.search(r"<function=([a-zA-Z0-9_]+)\s*(\{.*?\})", err_str, re.DOTALL)
            if match:
                fn_name = match.group(1)
                try:
                    fn_args = json.loads(match.group(2))
                except Exception:
                    fn_args = {}
                logger.info(f"Recovering pseudo-function call: {fn_name}({fn_args})")
                tool_result = await FinancialToolExecutor.execute(fn_name, fn_args, telegram_id)
                
                # Direct synthesis without tools
                synth_messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_message},
                    {"role": "system", "content": f"Tool '{fn_name}' data: {json.dumps(tool_result)}\nSynthesize a complete, polished response for the user."}
                ]
                try:
                    synth_resp = await cls._create_chat_completion_with_retry(
                        client=client,
                        model=settings.GROQ_MODEL,
                        messages=synth_messages,
                        temperature=0.3,
                        max_tokens=1024
                    )
                    return cls._clean_think_tags(synth_resp.choices[0].message.content or "")
                except Exception as synth_err:
                    logger.error(f"Synthesis fallback failed: {synth_err}")

            # General direct fallback without tools
            try:
                direct_messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_message}
                ]
                fallback_resp = await cls._create_chat_completion_with_retry(
                    client=client,
                    model=settings.GROQ_MODEL,
                    messages=direct_messages,
                    temperature=0.3,
                    max_tokens=1024
                )
                return cls._clean_think_tags(fallback_resp.choices[0].message.content or "")
            except Exception as direct_err:
                logger.error(f"Direct fallback also failed: {direct_err}", exc_info=True)
                return f"I encountered an analytical error processing your request: {str(e)}"

    @classmethod
    async def transcribe_and_reason_audio(
        cls,
        audio_bytes: bytes,
        filename: str = "voice.ogg",
        history: Optional[List[ConversationMessage]] = None,
        user_profile: Optional[UserProfile] = None,
        telegram_id: int = 0
    ) -> str:
        """Transcribes incoming voice note with Groq Whisper and returns financial analysis."""
        client = cls.get_client()
        if not client:
            return "Groq API key is not configured for voice transcription."

        try:
            # 1. Transcribe with Groq Whisper
            audio_file = (filename, audio_bytes)
            transcription = await client.audio.transcriptions.create(
                model=settings.GROQ_AUDIO_MODEL,
                file=audio_file,
                response_format="text"
            )

            transcribed_text = str(transcription).strip()
            logger.info(f"Groq Whisper transcription: '{transcribed_text}'")

            if not transcribed_text:
                return "I couldn't hear any clear audio in that voice note."

            # 2. Process transcribed text with full Groq financial intelligence & tools
            analysis = await cls.generate_response(
                user_message=transcribed_text,
                history=history or [],
                user_profile=user_profile,
                telegram_id=telegram_id
            )

            return f"🗣️ *Transcript*: _\"{transcribed_text}\"_\n\n{analysis}"

        except Exception as e:
            logger.error(f"Error processing audio with Groq Whisper: {e}")
            return f"Could not process voice note: {str(e)}"

    @classmethod
    async def analyze_image(
        cls,
        image_bytes: bytes,
        caption: Optional[str] = None,
        mime_type: str = "image/jpeg",
        history: Optional[List[ConversationMessage]] = None,
        user_profile: Optional[UserProfile] = None,
        telegram_id: int = 0
    ) -> str:
        """
        Multimodal Image & Chart Analyzer.
        Processes stock/crypto charts, financial statements, tables, and screenshots
        using Groq Vision (llama-3.2-11b-vision-preview / llama-3.2-90b-vision-preview).
        """
        client = cls.get_client()
        if not client:
            # Fallback to Gemini if available
            try:
                from app.services.ai.gemini_service import GeminiService
                return await GeminiService.analyze_image(
                    image_bytes=image_bytes,
                    caption=caption,
                    mime_type=mime_type,
                    history=history,
                    user_profile=user_profile,
                    telegram_id=telegram_id
                )
            except Exception:
                return (
                    "⚠️ *Atlas AI Configuration Notice*: Groq API Key is not configured for image analysis. "
                    "Please configure `GROQ_API_KEY` in `.env` to enable financial vision reasoning."
                )

        user_query = caption.strip() if (caption and caption.strip()) else DEFAULT_IMAGE_ANALYSIS_QUERY
        base_instruction = FINANCIAL_IMAGE_SYSTEM_INSTRUCTION
        if user_profile:
            base_instruction += (
                f"\n\n### USER CONTEXT:\n"
                f"- User Role: {user_profile.role or 'Finance Professional'}\n"
                f"- Sectors: {', '.join(user_profile.industries_followed) if user_profile.industries_followed else 'General'}\n"
                f"- Watchlist: {', '.join(user_profile.watchlists) if user_profile.watchlists else 'None'}\n"
            )

        # Normalize and optimize image with PIL
        processed_bytes = image_bytes
        processed_mime = mime_type
        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                # Convert RGBA/P/LA to RGB
                if pil_img.mode in ("RGBA", "P", "LA", "CMYK"):
                    pil_img = pil_img.convert("RGB")
                elif pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                # Upscale tiny images to minimum 100x100 for vision model processing
                w, h = pil_img.size
                if w < 64 or h < 64:
                    scale = max(100 / max(w, 1), 100 / max(h, 1))
                    pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

                # Downscale excessively large images to max 2048px on longest edge for speed
                if max(w, h) > 2048:
                    pil_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

                out_buf = io.BytesIO()
                pil_img.save(out_buf, format="JPEG", quality=92)
                processed_bytes = out_buf.getvalue()
                processed_mime = "image/jpeg"
        except Exception as img_err:
            logger.warning(f"Image preprocessing warning: {img_err}, using original bytes.")

        try:
            # Base64 encode image
            base64_image = base64.b64encode(processed_bytes).decode("utf-8")
            image_data_url = f"data:{processed_mime};base64,{base64_image}"

            # Build messages
            messages = [{"role": "system", "content": base_instruction}]
            for msg in (history or []):
                role = "user" if msg.role.value == "user" else "assistant"
                messages.append({"role": role, "content": msg.content})

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            })

            # Call Groq Vision Model
            response = await cls._create_chat_completion_with_retry(
                client=client,
                model=settings.GROQ_VISION_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1500
            )

            raw_text = response.choices[0].message.content or "Image analysis complete."
            return cls._clean_think_tags(raw_text)

        except Exception as e:
            logger.error(f"Error in Groq analyze_image: {e}", exc_info=True)
            # Try Gemini fallback if configured
            try:
                from app.services.ai.gemini_service import GeminiService
                gemini_resp = await GeminiService.analyze_image(
                    image_bytes=image_bytes,
                    caption=caption,
                    mime_type=mime_type,
                    history=history,
                    user_profile=user_profile,
                    telegram_id=telegram_id
                )
                if not gemini_resp.startswith("Gemini API is not configured"):
                    return gemini_resp
            except Exception:
                pass

            return f"I encountered an analytical error while processing the image: {str(e)}"

    @staticmethod
    def _clean_think_tags(text: str) -> str:
        """Removes internal thinking blocks (e.g. from reasoning/vision models) for clean user presentation."""
        if not text:
            return ""
        if "</think>" in text:
            cleaned = text.split("</think>")[-1].strip()
            if cleaned:
                return cleaned
        if "<think>" in text:
            cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if cleaned:
                return cleaned
            return re.sub(r"^<think>\s*", "", text).strip()
        return text.strip()

    @classmethod
    async def reason_document(
        cls,
        query: str,
        document_context: str,
        file_name: str,
        user_profile: Optional[UserProfile] = None
    ) -> str:
        """Answers contextual financial questions based on uploaded documents / filings."""
        client = cls.get_client()
        if not client:
            return "Groq API is not configured for document reasoning."

        system_instruction = cls._build_system_instruction(user_profile)
        doc_prompt = f"""DOCUMENT CONTEXT FOR '{file_name}':
==================================================
{document_context[:12000]}
==================================================

USER QUESTION:
{query}

TASK:
Answer the user's question accurately using the document context above. Cite specific pages or sections if available. If information is not in the document, state so clearly."""

        try:
            response = await cls._create_chat_completion_with_retry(
                client=client,
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": doc_prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            return response.choices[0].message.content or "Document analysis complete."
        except Exception as e:
            logger.error(f"Error reasoning over document with Groq: {e}")
            return f"Error analyzing document: {str(e)}"

    @classmethod
    async def generate_morning_brief(
        cls,
        user_profile: UserProfile,
        market_data: str,
        news_data: str
    ) -> str:
        """Synthesizes a personalized morning intelligence brief."""
        client = cls.get_client()
        if not client:
            return f"🌅 *Morning Intelligence Brief*\n\nWatchlist: {', '.join(user_profile.watchlists)}\n\n{market_data}"

        prompt = MORNING_BRIEF_PROMPT_TEMPLATE.format(
            role=user_profile.role or "Financial Professional",
            industries=", ".join(user_profile.industries_followed) if user_profile.industries_followed else "Equities",
            watchlists=", ".join(user_profile.watchlists) if user_profile.watchlists else "Top Tech Equities",
            market_data=market_data,
            news_data=news_data
        )

        try:
            response = await cls._create_chat_completion_with_retry(
                client=client,
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are Atlas, delivering the premier Morning Intelligence Brief to an investment professional. Be succinct and analytical."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content or "Morning brief generated."
        except Exception as e:
            logger.error(f"Error generating morning brief with Groq: {e}")
            return f"🌅 *Atlas Morning Brief*\n\n{market_data}"

    @classmethod
    async def generate_market_alert(
        cls,
        user_profile: UserProfile,
        ticker: str,
        current_price: float,
        currency: str,
        change_pct: float,
        headline: str
    ) -> str:
        """Synthesizes an urgent market volatility alert."""
        client = cls.get_client()
        if not client:
            direction = "🔺 UP" if change_pct > 0 else "🔻 DOWN"
            return f"⚡ *Market Volatility Alert*: {ticker} is {direction} {abs(change_pct)}% at {current_price} {currency}.\nCatalyst: {headline}"

        prompt = MARKET_ALERT_PROMPT_TEMPLATE.format(
            ticker=ticker,
            current_price=current_price,
            currency=currency,
            change_pct=change_pct,
            headline=headline
        )

        try:
            response = await cls._create_chat_completion_with_retry(
                client=client,
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are Atlas. Deliver a concise 2-sentence alert for high intraday volatility."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=250
            )
            return response.choices[0].message.content or f"⚡ Alert: {ticker} moved {change_pct}%."
        except Exception as e:
            logger.error(f"Error generating alert synthesis with Groq: {e}")
            return f"⚡ *Atlas Market Alert*: {ticker} moved {change_pct}% today to {current_price} {currency}."
