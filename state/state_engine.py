"""
MANORA Buddy State Engine.
Maintains Buddy's internal emotional state.
All state calculations are purely deterministic and mathematically bounded.
"""

import logging
import math
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, field_validator

from config.settings import get_settings
from emotion_agent.emotion_schema import EmotionAnalysis
from state.state_rules import BASELINE_STATE, STATE_MAX, STATE_MIN, calculate_state_deltas

logger = logging.getLogger("manora.state.engine")


class BuddyState(BaseModel):
    """
    Internal emotional state of the Buddy Agent.
    All 7 dimensions are strictly numeric values in [0.0, 1.0].
    """
    happiness: float = Field(default=0.60, ge=0.0, le=1.0, description="Buddy's level of joy and satisfaction")
    sadness: float = Field(default=0.10, ge=0.0, le=1.0, description="Buddy's level of sorrow/empathic grief")
    frustration: float = Field(default=0.10, ge=0.0, le=1.0, description="Buddy's level of constructive irritation")
    concern: float = Field(default=0.20, ge=0.0, le=1.0, description="Buddy's level of protective worry/alertness")
    warmth: float = Field(default=0.80, ge=0.0, le=1.0, description="Buddy's unconditional positive regard & affection")
    patience: float = Field(default=0.80, ge=0.0, le=1.0, description="Buddy's tolerance and calm presence")
    energy: float = Field(default=0.70, ge=0.0, le=1.0, description="Buddy's cognitive/conversational vitality")

    @field_validator("happiness", "sadness", "frustration", "concern", "warmth", "patience", "energy", mode="before")
    @classmethod
    def clamp_dimension(cls, v: Any) -> float:
        try:
            val = float(v)
            return round(max(STATE_MIN, min(STATE_MAX, val)), 3)
        except (ValueError, TypeError):
            return 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "happiness": self.happiness,
            "sadness": self.sadness,
            "frustration": self.frustration,
            "concern": self.concern,
            "warmth": self.warmth,
            "patience": self.patience,
            "energy": self.energy,
        }


class StateEngine:
    """Deterministic state engine for managing Buddy's emotional transitions and decay."""

    @staticmethod
    def initialize_state() -> BuddyState:
        """Initializes Buddy state to baseline equilibrium."""
        return BuddyState(**BASELINE_STATE)

    @staticmethod
    def validate_state(state: Union[BuddyState, Dict[str, Any]]) -> BuddyState:
        """Ensures all state values are strictly bounded within [0.0, 1.0]."""
        if isinstance(state, BuddyState):
            return state
        return BuddyState(**state)

    @classmethod
    def update_state(
        cls,
        current_state: Union[BuddyState, Dict[str, Any]],
        emotion_analysis: EmotionAnalysis,
    ) -> BuddyState:
        """
        Deterministically updates Buddy's internal state based on structured EmotionAnalysis.
        """
        current = cls.validate_state(current_state)

        # Build emotion intensity lookup map
        intensities: Dict[str, float] = {}
        for item in emotion_analysis.emotions:
            intensities[item.emotion.lower()] = item.intensity
        if emotion_analysis.primary_emotion:
            intensities[emotion_analysis.primary_emotion.lower()] = max(
                intensities.get(emotion_analysis.primary_emotion.lower(), 0.0),
                0.75
            )

        # Calculate deltas
        deltas = calculate_state_deltas(
            primary_emotion=emotion_analysis.primary_emotion,
            emotion_intensities=intensities,
            behavioral_signals=emotion_analysis.behavioral_signals,
            decision_signals=emotion_analysis.decision_signals,
            goal_related=emotion_analysis.goal_relevance.related,
        )

        new_values = {}
        for dim, current_val in current.to_dict().items():
            delta = deltas.get(dim, 0.0)
            updated = current_val + delta
            # Clamp strictly to [0.0, 1.0]
            new_values[dim] = round(max(STATE_MIN, min(STATE_MAX, updated)), 3)

        new_state = BuddyState(**new_values)
        logger.debug(f"State updated: from={current.to_dict()} to={new_state.to_dict()}")
        return new_state

    @classmethod
    def apply_decay(
        cls,
        current_state: Union[BuddyState, Dict[str, Any]],
        time_delta_hours: float = 1.0,
        decay_rate: Optional[float] = None,
    ) -> BuddyState:
        """
        Gradually decays Buddy state towards baseline over elapsed time.
        Uses exponential decay formula: V(t) = Baseline + (V0 - Baseline) * e^(-lambda * t)
        """
        current = cls.validate_state(current_state)
        settings = get_settings()
        rate = decay_rate if decay_rate is not None else settings.BUDDY_DECAY_RATE

        decay_factor = math.exp(-rate * max(0.0, time_delta_hours))

        decayed_values = {}
        for dim, current_val in current.to_dict().items():
            baseline_val = BASELINE_STATE[dim]
            # Drift towards baseline
            decayed = baseline_val + (current_val - baseline_val) * decay_factor
            decayed_values[dim] = round(max(STATE_MIN, min(STATE_MAX, decayed)), 3)

        return BuddyState(**decayed_values)


# Global singleton instance
state_engine = StateEngine()
