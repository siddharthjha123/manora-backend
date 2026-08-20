"""
MANORA Emotion Agent - Engine.
Coordinates ML emotion probabilities, contextual prompt building, LLM execution,
and structured parsing to produce EmotionAnalysis.
"""

import logging
from typing import Any, Dict, List, Optional

from emotion_agent.emotion_parser import EmotionParser
from emotion_agent.emotion_prompt import build_emotion_analysis_prompt
from emotion_agent.emotion_schema import EmotionAnalysis
from llm.base import LLMClient, llm_client
from ml.emotion_classifier import EmotionClassifier, emotion_classifier
from observability.metrics import EMOTION_PREDICTIONS_TOTAL

logger = logging.getLogger("manora.emotion.engine")


class EmotionAgent:
    """
    Emotion Agent responsible for analyzing what the student is feeling
    and identifying emotional/behavioral signals.
    """

    def __init__(
        self,
        classifier: Optional[EmotionClassifier] = None,
        llm: Optional[LLMClient] = None,
    ):
        self.classifier = classifier or emotion_classifier
        self.llm = llm or llm_client

    async def analyze(
        self,
        interaction_id: str,
        user_id: str,
        session_id: str,
        text: str,
        ml_probabilities: Optional[Dict[str, float]] = None,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        memories: Optional[List[Dict[str, Any]]] = None,
        goals: Optional[List[Dict[str, Any]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> EmotionAnalysis:
        """
        Executes complete emotion analysis pipeline:
        1. Obtains ML emotion probabilities (if not already passed).
        2. Constructs contextualized system and user prompt.
        3. Executes LLM reasoning.
        4. Parses and validates structured EmotionAnalysis output.
        """
        # Step 1: Run ML classification if not supplied
        if ml_probabilities is None:
            ml_probabilities = self.classifier.predict(text)

        logger.debug(f"Emotion ML probabilities for interaction {interaction_id}: {ml_probabilities}")

        # Step 2: Build prompt
        messages = build_emotion_analysis_prompt(
            interaction_id=interaction_id,
            text=text,
            ml_probabilities=ml_probabilities,
            recent_context=recent_context,
            memories=memories,
            goals=goals,
            user_context=user_context,
        )

        # Step 3: LLM inference
        raw_response = await self.llm.generate(
            messages=messages,
            temperature=0.3,
            json_mode=True,
        )

        # Step 4: Validate and parse output
        analysis = EmotionParser.parse(raw_response, interaction_id=interaction_id)

        # Record emotion prediction metric (using low-cardinality primary_emotion label)
        primary = (analysis.primary_emotion or "unknown").lower().strip()
        EMOTION_PREDICTIONS_TOTAL.labels(primary_emotion=primary).inc()

        logger.info(
            f"Emotion analysis completed for interaction {interaction_id}: "
            f"primary={analysis.primary_emotion}, emotions={len(analysis.emotions)}"
        )

        return analysis


# Global singleton instance
emotion_agent = EmotionAgent()
