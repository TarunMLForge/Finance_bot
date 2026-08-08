"""
Atlas AI Financial Assistant - Telegram Message & Multimodal Handlers
Implements Zero-Command conversational interactions, voice note comprehension, and document intelligence.
"""

from datetime import datetime, timezone
import io
from typing import List, Optional
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from app.core.config import settings
from app.core.logger import logger
from app.core.database import (
    get_users_collection,
    get_conversations_collection,
    get_documents_collection
)
from app.models.user import UserProfile, NotificationSchedule
from app.models.conversation import ConversationMessage, MessageRole
from app.models.document import DocumentRecord
from app.services.ai.groq_service import GroqService
from app.services.ai.prompts import ONBOARDING_SYSTEM_INSTRUCTION
from app.services.document.pdf_parser import PDFParserService


# Resilient In-Memory State Fallbacks (used when MongoDB is offline or unconfigured)
IN_MEMORY_USERS: dict = {}
IN_MEMORY_CONVERSATIONS: dict = {}


async def get_or_create_user(update: Update) -> UserProfile:
    """Fetches user profile from MongoDB or in-memory fallback."""
    user = update.effective_user
    telegram_id = user.id
    users_col = get_users_collection()

    if users_col is not None:
        try:
            doc = await users_col.find_one({"telegram_id": telegram_id})
            if doc:
                profile = UserProfile.from_mongo(doc)
                IN_MEMORY_USERS[telegram_id] = profile
                return profile
            
            # Create new user record
            new_profile = UserProfile(
                telegram_id=telegram_id,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
                notification_schedule=NotificationSchedule(
                    brief_time=settings.DEFAULT_BRIEF_TIME,
                    timezone=settings.DEFAULT_TIMEZONE
                )
            )
            await users_col.insert_one(new_profile.to_mongo())
            logger.info(f"Created new user profile for telegram_id={telegram_id} ({user.first_name})")
            IN_MEMORY_USERS[telegram_id] = new_profile
            return new_profile
        except Exception as e:
            logger.warning(f"Database access failed ({e}), falling back to in-memory profile.")

    # Fallback in-memory profile
    if telegram_id in IN_MEMORY_USERS:
        return IN_MEMORY_USERS[telegram_id]

    fallback_profile = UserProfile(
        telegram_id=telegram_id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        notification_schedule=NotificationSchedule(
            brief_time=settings.DEFAULT_BRIEF_TIME,
            timezone=settings.DEFAULT_TIMEZONE
        )
    )
    IN_MEMORY_USERS[telegram_id] = fallback_profile
    return fallback_profile


async def get_recent_history(telegram_id: int, limit: int = 10) -> List[ConversationMessage]:
    """Fetches the last N conversation messages for context window management."""
    conv_col = get_conversations_collection()
    if conv_col is not None:
        try:
            cursor = conv_col.find({"telegram_id": telegram_id}).sort("timestamp", -1).limit(limit)
            messages = []
            async for doc in cursor:
                messages.append(ConversationMessage.from_mongo(doc))
            messages.reverse()
            return messages
        except Exception as e:
            logger.warning(f"Database conversation retrieval failed ({e}), using in-memory history.")

    # Fallback in-memory messages
    return IN_MEMORY_CONVERSATIONS.get(telegram_id, [])[-limit:]


async def save_interaction(telegram_id: int, user_text: str, assistant_text: str, metadata: dict = None) -> None:
    """Persists user query and assistant response to MongoDB and in-memory cache."""
    user_msg = ConversationMessage(
        telegram_id=telegram_id,
        role=MessageRole.USER,
        content=user_text,
        metadata=metadata or {}
    )
    assistant_msg = ConversationMessage(
        telegram_id=telegram_id,
        role=MessageRole.ASSISTANT,
        content=assistant_text,
        metadata=metadata or {}
    )

    # Always keep in in-memory list
    if telegram_id not in IN_MEMORY_CONVERSATIONS:
        IN_MEMORY_CONVERSATIONS[telegram_id] = []
    IN_MEMORY_CONVERSATIONS[telegram_id].extend([user_msg, assistant_msg])
    if len(IN_MEMORY_CONVERSATIONS[telegram_id]) > settings.MAX_DB_CONVERSATION_LIMIT:
        IN_MEMORY_CONVERSATIONS[telegram_id] = IN_MEMORY_CONVERSATIONS[telegram_id][-settings.MAX_DB_CONVERSATION_LIMIT:]

    conv_col = get_conversations_collection()
    if conv_col is not None:
        try:
            await conv_col.insert_many([user_msg.to_mongo(), assistant_msg.to_mongo()])
            total_count = await conv_col.count_documents({"telegram_id": telegram_id})
            if total_count > settings.MAX_DB_CONVERSATION_LIMIT:
                excess = total_count - settings.MAX_DB_CONVERSATION_LIMIT
                oldest_docs = await conv_col.find(
                    {"telegram_id": telegram_id}
                ).sort("timestamp", 1).limit(excess).to_list(length=excess)
                oldest_ids = [d["_id"] for d in oldest_docs]
                await conv_col.delete_many({"_id": {"$in": oldest_ids}})
        except Exception as e:
            logger.warning(f"Could not persist conversation to MongoDB: {e}")


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Mandatory /start boot handler.
    Welcomes user naturally and initiates conversational onboarding.
    """
    user_profile = await get_or_create_user(update)
    telegram_id = update.effective_user.id

    await update.message.chat.send_action(ChatAction.TYPING)

    # Initial conversational welcome
    welcome_text = (
        f"👋 Welcome to *Atlas*, {user_profile.first_name or 'there'}.\n\n"
        "I am your autonomous AI Financial Assistant. I provide real-time market data, "
        "fundamental equity valuation, SEC filing breakdowns, and personalized daily morning briefs.\n\n"
        "To tailor your intelligence feed, what is your primary focus in the markets "
        "(e.g., equity research, portfolio management, day trading), and what tickers or sectors are on your radar?"
    )

    await update.message.reply_text(welcome_text, parse_mode="Markdown")
    await save_interaction(telegram_id, "/start", welcome_text)


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Pure Conversational Text Handler (Zero Command UI).
    Processes all natural financial queries, watchlist modifications, and preference updates.
    """
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    telegram_id = update.effective_user.id
    user_profile = await get_or_create_user(update)

    # Show typing indicator
    await update.message.chat.send_action(ChatAction.TYPING)

    # Check if this is a follow-up about an uploaded document
    doc_col = get_documents_collection()
    latest_doc = None
    if doc_col is not None:
        try:
            latest_doc = await doc_col.find_one(
                {"telegram_id": telegram_id},
                sort=[("upload_date", -1)]
            )
        except Exception:
            pass

    # If user mentions document/filing/page or has recent doc, include in reasoning
    if latest_doc and any(k in user_text.lower() for k in ["page", "document", "filing", "pdf", "report", "risk", "10-k", "10-q"]):
        doc_record = DocumentRecord.from_mongo(latest_doc)
        doc_context = f"Document: {doc_record.file_name}\nSummary: {doc_record.extracted_text_summary}\n\nContent:\n" + "\n---\n".join(doc_record.full_text_chunks[:8])
        response_text = await GroqService.reason_document(
            query=user_text,
            document_context=doc_context,
            file_name=doc_record.file_name,
            user_profile=user_profile
        )
    else:
        # Standard conversation with history and tools
        history = await get_recent_history(telegram_id, limit=settings.MAX_CONVERSATION_HISTORY)
        response_text = await GroqService.generate_response(
            user_message=user_text,
            history=history,
            user_profile=user_profile,
            telegram_id=telegram_id
        )

    # Send response to user (with fallback if markdown formatting has special chars)
    try:
        await update.message.reply_text(response_text, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(response_text)

    # Save to MongoDB
    await save_interaction(telegram_id, user_text, response_text)


async def voice_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Multimodal Voice Handler.
    Transcribes audio notes directly using Gemini and returns financial analysis.
    """
    voice = update.message.voice or update.message.audio
    if not voice:
        return

    telegram_id = update.effective_user.id
    user_profile = await get_or_create_user(update)

    await update.message.chat.send_action(ChatAction.RECORD_VOICE)

    try:
        # Download audio file into memory buffer
        file_obj = await context.bot.get_file(voice.file_id)
        audio_buffer = io.BytesIO()
        await file_obj.download_to_memory(audio_buffer)
        audio_bytes = audio_buffer.getvalue()

        # Send to Groq Whisper and LLM
        history = await get_recent_history(telegram_id, limit=5)
        
        response_text = await GroqService.transcribe_and_reason_audio(
            audio_bytes=audio_bytes,
            filename="voice_note.ogg",
            history=history,
            user_profile=user_profile,
            telegram_id=telegram_id
        )

        try:
            await update.message.reply_text(f"🎙️ *Voice Note Analysis*:\n\n{response_text}", parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(f"🎙️ Voice Note Analysis:\n\n{response_text}")

        await save_interaction(telegram_id, "[Voice Note]", response_text, metadata={"is_voice": True})

    except Exception as e:
        logger.error(f"Error handling voice note: {e}")
        await update.message.reply_text(f"Sorry, I had trouble processing that voice note: {str(e)}")


async def photo_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Multimodal Photo & Image Handler.
    Processes stock/crypto technical charts, financial statements, balance sheets, tables, and screenshots.
    """
    if not update.message or not update.message.photo:
        return

    telegram_id = update.effective_user.id
    user_profile = await get_or_create_user(update)
    caption = (update.message.caption or "").strip()

    # Show typing / photo indicator
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        # Get highest resolution photo (the last element in the photo array)
        photo_size = update.message.photo[-1]
        file_obj = await context.bot.get_file(photo_size.file_id)

        image_buffer = io.BytesIO()
        await file_obj.download_to_memory(image_buffer)
        image_bytes = image_buffer.getvalue()

        # Fetch recent conversation context for continuity
        history = await get_recent_history(telegram_id, limit=settings.MAX_CONVERSATION_HISTORY)

        # Process with Groq Vision Engine
        response_text = await GroqService.analyze_image(
            image_bytes=image_bytes,
            caption=caption,
            mime_type="image/jpeg",
            history=history,
            user_profile=user_profile,
            telegram_id=telegram_id
        )

        try:
            await update.message.reply_text(response_text, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(response_text)

        # Save interaction to MongoDB
        prompt_label = f"[Photo Analysis] {caption}".strip() if caption else "[Photo / Chart Analysis]"
        await save_interaction(
            telegram_id,
            prompt_label,
            response_text,
            metadata={"is_photo": True, "file_id": photo_size.file_id}
        )

    except Exception as e:
        logger.error(f"Error handling photo in Telegram: {e}", exc_info=True)
        await update.message.reply_text(f"⚠️ Unable to analyze photo: {str(e)}")


async def document_upload_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Multimodal Document Handler (PDF Filings & Image Attachments).
    Handles both PDF financial filings and uncompressed image documents (PNG, JPG, WEBP).
    """
    doc = update.message.document
    if not doc:
        return

    file_name = (doc.file_name or "document.pdf").lower()
    telegram_id = update.effective_user.id
    user_profile = await get_or_create_user(update)
    caption = (update.message.caption or "").strip()

    # Supported image extensions
    image_extensions = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff"
    }

    # Check if document is an image file
    matched_ext = next((ext for ext in image_extensions if file_name.endswith(ext)), None)

    if matched_ext:
        await update.message.chat.send_action(ChatAction.TYPING)
        try:
            file_obj = await context.bot.get_file(doc.file_id)
            image_buffer = io.BytesIO()
            await file_obj.download_to_memory(image_buffer)
            image_bytes = image_buffer.getvalue()

            history = await get_recent_history(telegram_id, limit=settings.MAX_CONVERSATION_HISTORY)
            response_text = await GroqService.analyze_image(
                image_bytes=image_bytes,
                caption=caption,
                mime_type=image_extensions[matched_ext],
                history=history,
                user_profile=user_profile,
                telegram_id=telegram_id
            )

            try:
                await update.message.reply_text(response_text, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(response_text)

            prompt_label = f"[Image Doc: {doc.file_name}] {caption}".strip()
            await save_interaction(
                telegram_id,
                prompt_label,
                response_text,
                metadata={"is_image_doc": True, "file_name": doc.file_name}
            )
            return
        except Exception as e:
            logger.error(f"Error handling image document: {e}", exc_info=True)
            await update.message.reply_text(f"⚠️ Failed to process image document: {str(e)}")
            return

    if not file_name.endswith(".pdf"):
        await update.message.reply_text(
            "Please upload a financial document in PDF format (e.g. 10-K, earnings deck) "
            "or an image (PNG, JPG, WEBP)."
        )
        return

    await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)
    await update.message.reply_text(f"📄 *Processing {doc.file_name or 'document.pdf'}*... Extracting financial statements and key sections.", parse_mode="Markdown")

    try:
        file_obj = await context.bot.get_file(doc.file_id)
        pdf_buffer = io.BytesIO()
        await file_obj.download_to_memory(pdf_buffer)
        pdf_bytes = pdf_buffer.getvalue()

        # Parse PDF
        parsed = PDFParserService.parse_pdf_bytes(pdf_bytes, file_name=doc.file_name or "document.pdf")
        if not parsed.get("success"):
            await update.message.reply_text(f"Error extracting PDF: {parsed.get('error')}")
            return

        # Store in MongoDB
        doc_record = DocumentRecord(
            telegram_id=telegram_id,
            document_id=doc.file_id,
            file_name=doc.file_name or "document.pdf",
            file_size_bytes=doc.file_size or len(pdf_bytes),
            extracted_text_summary=parsed.get("preview_summary", ""),
            full_text_chunks=parsed.get("chunks", []),
            upload_date=datetime.now(timezone.utc)
        )

        doc_col = get_documents_collection()
        if doc_col is not None:
            await doc_col.update_one(
                {"telegram_id": telegram_id, "document_id": doc.file_id},
                {"$set": doc_record.to_mongo()},
                upsert=True
            )

        # Generate quick executive summary of document
        summary_query = "Provide a high-level 3-bullet executive summary of this filing, highlighting major financial performance, risks, or key announcements."
        doc_context = "\n---\n".join(parsed.get("chunks", [])[:5])
        summary_response = await GroqService.reason_document(
            query=summary_query,
            document_context=doc_context,
            file_name=doc.file_name or "document.pdf",
            user_profile=user_profile
        )

        reply_msg = (
            f"✅ *Document Analyzed*: `{doc.file_name or 'document.pdf'}` ({parsed.get('num_pages')} pages)\n\n"
            f"{summary_response}\n\n"
            "_You can now ask me any specific questions about this document (e.g., 'What are the top 3 risk factors on page 4?' or 'What was the Q4 gross margin?')_"
        )

        try:
            await update.message.reply_text(reply_msg, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(reply_msg)

        await save_interaction(
            telegram_id,
            f"[Uploaded PDF: {doc.file_name or 'document.pdf'}]",
            reply_msg,
            metadata={"file_name": doc.file_name or "document.pdf"}
        )

    except Exception as e:
        logger.error(f"Error processing uploaded PDF: {e}")
        await update.message.reply_text(f"Failed to process document: {str(e)}")
