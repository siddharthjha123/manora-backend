"""
MANORA Mock Data Agent.
Simulates the candidate memory extraction interface.
Enables full end-to-end pipeline execution and allows the real Data Agent
to be swapped in without modifying other components.
"""

import abc
import logging
from typing import Any, Dict, List

from data_agent.data_schema import (
    CandidateMemory,
    CandidateMemoryBehavior,
    CandidateMemoryDecision,
    CandidateMemoryEmotion,
    CandidateMemoryEvent,
    CandidateMemoryGoalRelevance,
)
from emotion_agent.emotion_schema import EmotionAnalysis

logger = logging.getLogger("manora.data_agent.mock")


class BaseDataAgent(abc.ABC):
    """Abstract interface for Data Agent implementations."""

    @abc.abstractmethod
    def process(
        self,
        interaction: Dict[str, Any],
        emotion: EmotionAnalysis,
    ) -> List[CandidateMemory]:
        """
        Extracts candidate memories from the interaction and emotion analysis.
        """
        pass


class MockDataAgent(BaseDataAgent):
    """
    Mock Data Agent simulating memory identification from student interaction signals.
    """

    def process(
        self,
        interaction: Dict[str, Any],
        emotion: EmotionAnalysis,
    ) -> List[CandidateMemory]:
        """
        Extracts candidate memories based on detected behavioral and decision signals.
        """
        raw_text = interaction.get("raw_text", "")
        candidates: List[CandidateMemory] = []

        # If there are behavioral or decision signals, construct a candidate memory
        has_behaviors = bool(emotion.behavioral_signals)
        has_decisions = bool(emotion.decision_signals)
        is_high_emotion = any(e.intensity >= 0.6 for e in emotion.emotions)

        if has_behaviors or has_decisions or is_high_emotion:
            # Build memory content narrative
            primary_behavior = (
                emotion.behavioral_signals[0]
                if emotion.behavioral_signals
                else "Student experienced emotional conflict during daily activity"
            )
            primary_decision = (
                emotion.decision_signals[0]
                if emotion.decision_signals
                else "Student prioritized immediate activity over planned schedule"
            )

            content = (
                f"Student engaged in: {primary_behavior}. "
                f"Decision context: {primary_decision}."
            )

            # Build emotional state items
            mem_emotions = [
                CandidateMemoryEmotion(
                    emotion=e.emotion,
                    confidence=round(e.confidence, 2),
                )
                for e in emotion.emotions[:2]
            ]

            candidate = CandidateMemory(
                content=content,
                context={
                    "topic": "academic_routine" if emotion.goal_relevance.related else "general_behavior",
                    "subtopic": "avoidance_pattern" if "avoid" in raw_text.lower() or "netflix" in raw_text.lower() else "emotional_reflection",
                },
                emotional_state=mem_emotions,
                events=[
                    CandidateMemoryEvent(
                        type="behavioral_pattern",
                        description=primary_behavior,
                    )
                ],
                behavior=CandidateMemoryBehavior(
                    type="avoidance" if "avoid" in primary_behavior.lower() else "action",
                    description=primary_behavior,
                ),
                decision=CandidateMemoryDecision(
                    description=primary_decision,
                ),
                goal_relevance=CandidateMemoryGoalRelevance(
                    related=emotion.goal_relevance.related,
                    goal=emotion.goal_relevance.goal,
                ),
                importance=0.84 if emotion.goal_relevance.related else 0.65,
                confidence=0.88,
            )

            candidates.append(candidate)

        logger.debug(f"Mock Data Agent produced {len(candidates)} candidate memories.")
        return candidates


# Global singleton instance
mock_data_agent = MockDataAgent()
