"""
MANORA Buddy Agent - Engine.
Coordinates Buddy response generation by integrating student emotion analysis,
Buddy's internal emotional state, and retrieved memory context into natural dialogue.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from buddy.buddy_prompt import build_buddy_prompt
from buddy.buddy_schema import BuddyExpression, BuddyResponse, BuddyResponseType
from emotion_agent.emotion_schema import EmotionAnalysis
from llm.base import LLMClient, llm_client
from state.state_engine import BuddyState

logger = logging.getLogger("manora.buddy.engine")


class BuddyAgent:
    """
    Buddy Agent responsible for deciding what Buddy should say,
    the facial expression, expression intensity, and conversational stance.
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or llm_client

    def _clean_json_string(self, raw_text: str) -> str:
        text = raw_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        return text

    async def generate(
        self,
        user_text: str,
        emotion_analysis: EmotionAnalysis,
        buddy_state: BuddyState,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        memories: Optional[List[Dict[str, Any]]] = None,
        goals: Optional[List[Dict[str, Any]]] = None,
    ) -> BuddyResponse:
        """
        Generates structured BuddyResponse without modifying Buddy State directly.
        """
        messages = build_buddy_prompt(
            user_text=user_text,
            emotion_analysis=emotion_analysis,
            buddy_state=buddy_state,
            recent_context=recent_context,
            memories=memories,
            goals=goals,
        )

        try:
            raw_response = await self.llm.generate(
                messages=messages,
                temperature=0.7,
                json_mode=True,
            )
            cleaned = self._clean_json_string(raw_response)
            data = json.loads(cleaned)
            response = BuddyResponse.model_validate(data)

            logger.info(
                f"Buddy response generated: expression={response.expression}, "
                f"type={response.response_type}, intensity={response.intensity}"
            )
            return response

        except Exception as e:
            logger.warning(f"Error parsing Buddy response JSON ({e}). Using robust fallback.")
            return self._build_fallback_response(user_text, emotion_analysis, buddy_state)

    def _build_fallback_response(
        self,
        user_text: str,
        emotion_analysis: EmotionAnalysis,
        buddy_state: BuddyState,
    ) -> BuddyResponse:
        """Fallback response builder if LLM fails or is unavailable."""
        if buddy_state.concern > 0.6 and buddy_state.frustration > 0.4:
            return BuddyResponse(
                text="You're repeating the same pattern again. Do you actually want to achieve this goal?",
                expression=BuddyExpression.CONCERNED.value,
                intensity=round(buddy_state.concern, 2),
                response_type=BuddyResponseType.CHALLENGE.value,
            )
        elif buddy_state.sadness > 0.4 or "sad" in emotion_analysis.primary_emotion:
            return BuddyResponse(
                text="I hear how heavy this feels right now. Take your time, I'm here with you.",
                expression=BuddyExpression.THOUGHTFUL.value,
                intensity=round(buddy_state.warmth, 2),
                response_type=BuddyResponseType.SUPPORT.value,
            )
        else:
            return BuddyResponse(
                text="I hear you. Let's look at what's going on and take it one step at a time.",
                expression=BuddyExpression.ENCOURAGING.value,
                intensity=round(buddy_state.warmth, 2),
                response_type=BuddyResponseType.REFLECTION.value,
            )


# Global singleton instance
buddy_agent = BuddyAgent()
