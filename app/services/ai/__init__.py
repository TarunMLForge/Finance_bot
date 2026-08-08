"""
Atlas AI Engine Services Package
"""

from app.services.ai.groq_service import GroqService
from app.services.ai.prompts import ATLAS_SYSTEM_PROMPT, ONBOARDING_SYSTEM_INSTRUCTION

# Provide alias AIService for modularity
AIService = GroqService

__all__ = ["GroqService", "AIService", "ATLAS_SYSTEM_PROMPT", "ONBOARDING_SYSTEM_INSTRUCTION"]
