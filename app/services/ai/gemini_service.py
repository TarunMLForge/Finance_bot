"""
Atlas AI Financial Assistant - Gemini AI Service
Integrates Google Gemini for conversational intelligence, tool execution, audio transcription, and document analysis.
"""

from typing import List, Dict, Any, Optional
import json
import base64
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
from app.services.financial.tools import FINANCIAL_TOOL_DECLARATIONS, FinancialToolExecutor

# Import Google GenAI SDK (supports both google.genai v2 and google.generativeai)
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_NEW_GENAI = True
except ImportError:
    HAS_NEW_GENAI = False
    try:
        import google.generativeai as genai_v1
        HAS_OLD_GENAI = True
    except ImportError:
        HAS_OLD_GENAI = False


class GeminiService:
    """Core AI engine managing Gemini reasoning, tool dispatch, audio notes, and documents."""

    _client = None

    @classmethod
    def get_client(cls):
        """Initializes and returns the Gemini client."""
        if cls._client is None and settings.GEMINI_API_KEY:
            if HAS_NEW_GENAI:
                try:
                    cls._client = genai.Client(api_key=settings.GEMINI_API_KEY)
                    logger.info("Google GenAI v2 Client initialized successfully.")
                except Exception as e:
                    logger.error(f"Error initializing GenAI v2 Client: {e}")
            elif HAS_OLD_GENAI:
                try:
                    genai_v1.configure(api_key=settings.GEMINI_API_KEY)
                    cls._client = genai_v1
                    logger.info("Google GenerativeAI v1 configured successfully.")
                except Exception as e:
                    logger.error(f"Error configuring GenerativeAI v1: {e}")
        return cls._client

    @classmethod
    def _build_system_instruction(cls, user_profile: Optional[UserProfile] = None) -> str:
        """Constructs system instruction with user profile context."""
        instruction = ATLAS_SYSTEM_PROMPT
        if user_profile:
            profile_context = f"\n\n### USER CONTEXT:\n- User Role: {user_profile.role or 'Finance Professional'}\n- Sectors Followed: {', '.join(user_profile.industries_followed) if user_profile.industries_followed else 'General Markets'}\n- Active Watchlist: {', '.join(user_profile.watchlists) if user_profile.watchlists else 'None yet'}\n- Timezone: {user_profile.notification_schedule.timezone}\n"
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
        """Processes conversational input with history, tool execution, and synthesis."""
        client = cls.get_client()
        if not client:
            return (
                "⚠️ *Atlas AI Configuration Notice*: Gemini API Key is not configured. "
                "Please configure `GEMINI_API_KEY` in `.env` to enable full autonomous financial reasoning."
            )

        system_instruction = cls._build_system_instruction(user_profile)

        if HAS_NEW_GENAI:
            return await cls._generate_with_new_genai(
                client=client,
                user_message=user_message,
                history=history,
                system_instruction=system_instruction,
                telegram_id=telegram_id
            )
        else:
            return await cls._generate_with_v1_genai(
                user_message=user_message,
                history=history,
                system_instruction=system_instruction,
                telegram_id=telegram_id
            )

    @classmethod
    async def _generate_with_new_genai(
        cls,
        client: Any,
        user_message: str,
        history: List[ConversationMessage],
        system_instruction: str,
        telegram_id: int
    ) -> str:
        """Handles multi-turn conversational reasoning and tool calling using google.genai v2."""
        try:
            # Build conversation history contents
            contents = []
            for msg in history:
                role = "user" if msg.role.value == "user" else "model"
                contents.append(
                    genai_types.Content(
                        role=role,
                        parts=[genai_types.Part.from_text(text=msg.content)]
                    )
                )

            # Add current user prompt
            contents.append(
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text=user_message)]
                )
            )

            # Convert tools to genai types
            function_declarations = []
            for t in FINANCIAL_TOOL_DECLARATIONS:
                function_declarations.append(
                    genai_types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=t.get("parameters")
                    )
                )
            tools = [genai_types.Tool(function_declarations=function_declarations)]

            # Initial generation call
            config = genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools,
                temperature=0.3,
            )

            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=contents,
                config=config
            )

            # Handle Function Calling Loop (up to 3 turns)
            max_turns = 3
            current_turn = 0
            while response.function_calls and current_turn < max_turns:
                current_turn += 1
                function_call = response.function_calls[0]
                fn_name = function_call.name
                fn_args = dict(function_call.args) if function_call.args else {}

                logger.info(f"Gemini requested tool execution: {fn_name}({fn_args})")
                tool_result = await FinancialToolExecutor.execute(fn_name, fn_args, telegram_id)

                # Append model response containing the tool call
                contents.append(response.candidates[0].content)

                # Append function response
                tool_response_part = genai_types.Part.from_function_response(
                    name=fn_name,
                    response={"result": tool_result}
                )
                contents.append(
                    genai_types.Content(
                        role="user",
                        parts=[tool_response_part]
                    )
                )

                # Re-generate with tool output
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=contents,
                    config=config
                )

            return response.text or "I have processed your request."

        except Exception as e:
            logger.error(f"Error in Gemini generate_response: {e}", exc_info=True)
            return f"I encountered an analytical error processing your request: {str(e)}"

    @classmethod
    async def _generate_with_v1_genai(
        cls,
        user_message: str,
        history: List[ConversationMessage],
        system_instruction: str,
        telegram_id: int
    ) -> str:
        """Fallback generator using google.generativeai v1 SDK."""
        try:
            model = genai_v1.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                system_instruction=system_instruction
            )
            chat_history = []
            for msg in history:
                role = "user" if msg.role.value == "user" else "model"
                chat_history.append({"role": role, "parts": [msg.content]})

            chat = model.start_chat(history=chat_history)
            response = chat.send_message(user_message)
            return response.text
        except Exception as e:
            logger.error(f"Error in Gemini v1 fallback: {e}")
            return f"Unable to process query: {str(e)}"

    @classmethod
    async def transcribe_and_reason_audio(
        cls,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        history: Optional[List[ConversationMessage]] = None,
        user_profile: Optional[UserProfile] = None,
        telegram_id: int = 0
    ) -> str:
        """Transcribes incoming voice note and produces financial intelligence."""
        client = cls.get_client()
        if not client:
            return "Gemini API is not configured for voice processing."

        system_instruction = cls._build_system_instruction(user_profile)
        system_instruction += "\n\nNote: The user provided their query via a voice recording. Analyze their spoken words and respond with precise financial intelligence."

        try:
            if HAS_NEW_GENAI:
                part = genai_types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
                )
                prompt_part = genai_types.Part.from_text(
                    text="Listen to this voice message from a financial professional. Transcribe it mentally and deliver the requested market analysis or answer."
                )

                config = genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3
                )
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=[genai_types.Content(role="user", parts=[part, prompt_part])],
                    config=config
                )
                return response.text or "Voice note processed."
        except Exception as e:
            logger.error(f"Error processing voice note with Gemini: {e}")
            return f"Could not process voice note: {str(e)}"

        return "Voice processing requires Google GenAI client."

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
        """Analyzes financial charts, statements, and screenshots using Gemini Vision."""
        client = cls.get_client()
        if not client:
            return "Gemini API is not configured for image analysis."

        user_query = caption.strip() if (caption and caption.strip()) else DEFAULT_IMAGE_ANALYSIS_QUERY
        system_instruction = FINANCIAL_IMAGE_SYSTEM_INSTRUCTION
        if user_profile:
            system_instruction += (
                f"\n\n### USER CONTEXT:\n"
                f"- User Role: {user_profile.role or 'Finance Professional'}\n"
                f"- Sectors: {', '.join(user_profile.industries_followed) if user_profile.industries_followed else 'General'}\n"
                f"- Watchlist: {', '.join(user_profile.watchlists) if user_profile.watchlists else 'None'}\n"
            )

        try:
            if HAS_NEW_GENAI:
                part = genai_types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )
                prompt_part = genai_types.Part.from_text(text=user_query)

                config = genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2
                )
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=[genai_types.Content(role="user", parts=[part, prompt_part])],
                    config=config
                )
                return response.text or "Image analysis complete."
            elif HAS_OLD_GENAI:
                model = genai_v1.GenerativeModel(model_name=settings.GEMINI_MODEL)
                image_blob = {"mime_type": mime_type, "data": image_bytes}
                response = model.generate_content([image_blob, f"{system_instruction}\n\nTask: {user_query}"])
                return response.text or "Image analysis complete."
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {e}")
            return f"Could not analyze image: {str(e)}"

        return "Image analysis requires Google GenAI client."

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
            return "Gemini API is not configured for document reasoning."

        system_instruction = cls._build_system_instruction(user_profile)
        doc_prompt = f"""DOCUMENT CONTEXT FOR '{file_name}':
==================================================
{document_context[:15000]}
==================================================

USER QUESTION:
{query}

TASK:
Answer the user's question accurately using the document context above. Cite specific pages or sections if available. If information is not in the document, state so clearly."""

        try:
            if HAS_NEW_GENAI:
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=doc_prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2
                    )
                )
                return response.text or "Document analysis complete."
        except Exception as e:
            logger.error(f"Error reasoning over document with Gemini: {e}")
            return f"Error analyzing document: {str(e)}"

        return "Document reasoning completed."

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
            if HAS_NEW_GENAI:
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction="You are Atlas, delivering the premier Morning Intelligence Brief to an investment professional. Be succinct and analytical.",
                        temperature=0.3
                    )
                )
                return response.text or "Morning brief generated."
        except Exception as e:
            logger.error(f"Error generating morning brief: {e}")

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
            if HAS_NEW_GENAI:
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction="You are Atlas. Deliver a concise 2-sentence alert for high intraday volatility.",
                        temperature=0.2
                    )
                )
                return response.text or f"⚡ Alert: {ticker} moved {change_pct}%."
        except Exception as e:
            logger.error(f"Error generating alert synthesis: {e}")

        return f"⚡ *Atlas Market Alert*: {ticker} moved {change_pct}% today to {current_price} {currency}."
