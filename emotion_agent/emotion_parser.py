"""
MANORA Emotion Agent - Output Parser and Sanitizer.
Validates LLM output against the EmotionAnalysis schema and handles JSON edge cases.
"""

import json
import logging
import re
from typing import Any, Dict
from pydantic import ValidationError

from emotion_agent.emotion_schema import EmotionAnalysis, EmotionItem, GoalRelevance

logger = logging.getLogger("manora.emotion.parser")


class EmotionParseError(Exception):
    """Raised when emotion response parsing or validation fails."""
    pass


class EmotionParser:
    """Parses, sanitizes, and validates raw LLM output into EmotionAnalysis."""

    @staticmethod
    def _clean_json_string(raw_text: str) -> str:
        """Strips markdown code blocks, backticks, and extraneous whitespace."""
        text = raw_text.strip()
        # Extract content between ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        return text

    @classmethod
    def parse(cls, raw_text: str, interaction_id: str) -> EmotionAnalysis:
        """
        Parses raw LLM string into EmotionAnalysis model.
        Falls back to a safe structured representation if JSON is partially malformed.
        """
        cleaned = cls._clean_json_string(raw_text)

        try:
            data = json.loads(cleaned)
            # Ensure interaction_id is properly populated
            if not data.get("interaction_id"):
                data["interaction_id"] = interaction_id
            return EmotionAnalysis.model_validate(data)

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Emotion JSON validation issue ({e}). Constructing sanitized fallback.")
            return cls._build_safe_fallback(raw_text, interaction_id)

    @classmethod
    def _build_safe_fallback(cls, raw_text: str, interaction_id: str) -> EmotionAnalysis:
        """Constructs a valid fallback EmotionAnalysis if LLM produces free-form text."""
        return EmotionAnalysis(
            interaction_id=interaction_id,
            primary_emotion="neutral",
            emotions=[
                EmotionItem(
                    emotion="neutral",
                    intensity=0.5,
                    confidence=0.6,
                    source="parser_fallback",
                )
            ],
            emotional_summary=raw_text[:200] if raw_text else "Student expressed mixed thoughts.",
            behavioral_signals=["shared thoughts with support system"],
            decision_signals=[],
            goal_relevance=GoalRelevance(related=False, goal=None),
        )
