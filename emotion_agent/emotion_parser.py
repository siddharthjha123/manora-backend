"""
MANORA Emotion Agent - Output Parser and Sanitizer.
Validates LLM output against the EmotionAnalysis schema and handles JSON edge cases.
"""

import json
import logging
import re
from typing import Any, Dict

from pydantic import ValidationError

from emotion_agent.emotion_schema import (
    EmotionAnalysis,
    EmotionItem,
    GoalRelevance,
)

logger = logging.getLogger("manora.emotion.parser")


class EmotionParseError(Exception):
    """Raised when emotion response parsing or validation fails."""
    pass


class EmotionParser:
    """Parses, sanitizes, and validates raw LLM output into EmotionAnalysis."""

    @staticmethod
    def _clean_json_string(raw_text: str) -> str:
        """Strips markdown code blocks and surrounding whitespace."""
        if not raw_text:
            raise EmotionParseError("Emotion LLM returned empty response.")

        text = raw_text.strip()

        match = re.search(
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            text,
            re.IGNORECASE,
        )

        if match:
            text = match.group(1).strip()

        return text

    @classmethod
    def parse(
        cls,
        raw_text: str,
        interaction_id: str,
    ) -> EmotionAnalysis:

        try:
            cleaned = cls._clean_json_string(raw_text)
        except EmotionParseError as e:
            logger.warning(str(e))
            return cls._build_safe_fallback("", interaction_id)

        # ---------------------------------------------------------
        # 1. Normal JSON parsing
        # ---------------------------------------------------------
        try:
            data = json.loads(cleaned)

            if not data.get("interaction_id"):
                data["interaction_id"] = interaction_id

            return EmotionAnalysis.model_validate(data)

        except json.JSONDecodeError as e:
            logger.warning(
                "Emotion JSON malformed: %s",
                e,
            )

            logger.debug(
                "Raw emotion response: %r",
                cleaned[:2000],
            )

        except ValidationError as e:
            logger.warning(
                "Emotion schema validation failed: %s",
                e,
            )

        # ---------------------------------------------------------
        # 2. Attempt partial JSON recovery
        # ---------------------------------------------------------
        recovered = cls._recover_partial_json(cleaned)

        if recovered:
            try:
                if not recovered.get("interaction_id"):
                    recovered["interaction_id"] = interaction_id

                return EmotionAnalysis.model_validate(recovered)

            except ValidationError as e:
                logger.warning(
                    "Recovered emotion JSON failed schema validation: %s",
                    e,
                )

        # ---------------------------------------------------------
        # 3. Safe fallback
        # ---------------------------------------------------------
        return cls._build_safe_fallback(
            cleaned,
            interaction_id,
        )

    @classmethod
    def _recover_partial_json(
        cls,
        raw_text: str,
    ) -> Dict[str, Any] | None:
        """
        Attempts conservative recovery from partially malformed JSON.

        The method extracts fields independently instead of attempting
        to reconstruct arbitrary JSON.
        """

        data: Dict[str, Any] = {}

        # ---------------------------------------------------------
        # primary_emotion
        # ---------------------------------------------------------

        match = re.search(
            r'"primary_emotion"\s*:\s*"([^"]+)"',
            raw_text,
            re.IGNORECASE,
        )

        if match:
            data["primary_emotion"] = match.group(1).strip()

        # ---------------------------------------------------------
        # emotional_summary
        # ---------------------------------------------------------

        match = re.search(
            r'"emotional_summary"\s*:\s*"([^"]*)"',
            raw_text,
            re.IGNORECASE,
        )

        if match:
            data["emotional_summary"] = match.group(1).strip()

        # ---------------------------------------------------------
        # emotions
        # ---------------------------------------------------------

        emotions_match = re.search(
            r'"emotions"\s*:\s*\[(.*?)(?:\]|$)',
            raw_text,
            re.DOTALL | re.IGNORECASE,
        )

        if emotions_match:
            emotions_raw = emotions_match.group(1)

            # Recover individual emotion objects.
            emotion_objects = re.findall(
                r'\{(.*?)\}',
                emotions_raw,
                re.DOTALL,
            )

            recovered_emotions = []

            for obj in emotion_objects:
                emotion_match = re.search(
                    r'"emotion"\s*:\s*"([^"]+)"',
                    obj,
                    re.IGNORECASE,
                )

                intensity_match = re.search(
                    r'"intensity"\s*:\s*([0-9.]+)',
                    obj,
                    re.IGNORECASE,
                )

                confidence_match = re.search(
                    r'"confidence"\s*:\s*([0-9.]+)',
                    obj,
                    re.IGNORECASE,
                )

                source_match = re.search(
                    r'"source"\s*:\s*"([^"]+)"',
                    obj,
                    re.IGNORECASE,
                )

                if not emotion_match:
                    continue

                recovered_emotions.append(
                    {
                        "emotion": emotion_match.group(1).strip(),
                        "intensity": float(
                            intensity_match.group(1)
                        )
                        if intensity_match
                        else 0.5,
                        "confidence": float(
                            confidence_match.group(1)
                        )
                        if confidence_match
                        else 0.5,
                        "source": (
                            source_match.group(1).strip()
                            if source_match
                            else "partial_json_recovery"
                        ),
                    }
                )

            if recovered_emotions:
                data["emotions"] = recovered_emotions

        # ---------------------------------------------------------
        # behavioral_signals
        # ---------------------------------------------------------

        match = re.search(
            r'"behavioral_signals"\s*:\s*\[(.*?)\]',
            raw_text,
            re.DOTALL | re.IGNORECASE,
        )

        if match:
            try:
                data["behavioral_signals"] = json.loads(
                    "[" + match.group(1) + "]"
                )
            except json.JSONDecodeError:
                pass

        # ---------------------------------------------------------
        # decision_signals
        # ---------------------------------------------------------

        match = re.search(
            r'"decision_signals"\s*:\s*\[(.*?)\]',
            raw_text,
            re.DOTALL | re.IGNORECASE,
        )

        if match:
            try:
                data["decision_signals"] = json.loads(
                    "[" + match.group(1) + "]"
                )
            except json.JSONDecodeError:
                pass

        # ---------------------------------------------------------
        # goal_relevance
        # ---------------------------------------------------------

        related_match = re.search(
            r'"related"\s*:\s*(true|false)',
            raw_text,
            re.IGNORECASE,
        )

        goal_match = re.search(
            r'"goal"\s*:\s*"([^"]*)"',
            raw_text,
            re.IGNORECASE,
        )

        if related_match:
            data["goal_relevance"] = {
                "related": related_match.group(1).lower() == "true",
                "goal": (
                    goal_match.group(1).strip()
                    if goal_match
                    else None
                ),
            }

        # ---------------------------------------------------------
        # Cannot recover anything useful
        # ---------------------------------------------------------

        if "primary_emotion" not in data:
            return None

        # ---------------------------------------------------------
        # Fill only missing required fields
        # ---------------------------------------------------------

        data.setdefault(
            "emotions",
            [
                {
                    "emotion": data["primary_emotion"],
                    "intensity": 0.5,
                    "confidence": 0.5,
                    "source": "partial_json_recovery",
                }
            ],
        )

        data.setdefault(
            "emotional_summary",
            "Emotion inferred from partially recovered model output.",
        )

        data.setdefault(
            "behavioral_signals",
            [],
        )

        data.setdefault(
            "decision_signals",
            [],
        )

        data.setdefault(
            "goal_relevance",
            {
                "related": False,
                "goal": None,
            },
        )

        return data

    @classmethod
    def _build_safe_fallback(
        cls,
        raw_text: str,
        interaction_id: str,
    ) -> EmotionAnalysis:
        """Constructs a safe fallback EmotionAnalysis."""

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
            emotional_summary=(
                raw_text[:200]
                if raw_text
                else "Student expressed mixed thoughts."
            ),
            behavioral_signals=[
                "shared thoughts with support system"
            ],
            decision_signals=[],
            goal_relevance=GoalRelevance(
                related=False,
                goal=None,
            ),
        )