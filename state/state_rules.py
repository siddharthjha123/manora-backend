"""
MANORA Buddy State Engine - Transition Rules and Constants.
Provides deterministic mathematical transition rules mapping student EmotionAnalysis
to Buddy internal emotional state adjustments.
"""

from typing import Any, Dict

# Baseline emotional equilibrium of Buddy
BASELINE_STATE: Dict[str, float] = {
    "happiness": 0.60,
    "sadness": 0.10,
    "frustration": 0.10,
    "concern": 0.20,
    "warmth": 0.80,
    "patience": 0.80,
    "energy": 0.70,
}

# Clamping bounds
STATE_MIN: float = 0.0
STATE_MAX: float = 1.0


def calculate_state_deltas(
    primary_emotion: str,
    emotion_intensities: Dict[str, float],
    behavioral_signals: list,
    decision_signals: list,
    goal_related: bool,
) -> Dict[str, float]:
    """
    Deterministically computes deltas for each of Buddy's 7 emotional dimensions
    based on the structured EmotionAnalysis components.
    """
    deltas = {
        "happiness": 0.0,
        "sadness": 0.0,
        "frustration": 0.0,
        "concern": 0.0,
        "warmth": 0.0,
        "patience": 0.0,
        "energy": 0.0,
    }

    # 1. Evaluate emotion intensities
    frustration_val = emotion_intensities.get("frustration", 0.0)
    guilt_val = emotion_intensities.get("guilt", 0.0)
    stress_val = emotion_intensities.get("stress", 0.0)
    anxiety_val = emotion_intensities.get("anxiety", 0.0)
    sadness_val = emotion_intensities.get("sadness", 0.0)
    fatigue_val = emotion_intensities.get("fatigue", 0.0)
    happiness_val = emotion_intensities.get("happiness", 0.0)
    motivation_val = emotion_intensities.get("motivation", 0.0)

    # Detect repeated pattern / avoidance signals
    all_signals_text = " ".join([str(s) for s in behavioral_signals + decision_signals]).lower()
    has_avoidance = any(w in all_signals_text for w in ["avoid", "delayed", "entertainment", "netflix", "procrastinat", "escap"])
    has_conflict = any(w in all_signals_text for w in ["conflict", "despite", "instead", "chose"])

    # High student conflict / study avoidance
    if frustration_val > 0.4 or guilt_val > 0.4 or has_avoidance:
        # Concern goes up significantly
        deltas["concern"] += 0.15 * max(frustration_val, guilt_val, 0.5)
        # Buddy feels a bit of constructive frustration/urgency if student repeats avoidance
        if has_avoidance or has_conflict:
            deltas["frustration"] += 0.14 * max(frustration_val, guilt_val, 0.5)
            # Patience reduces slightly to provoke constructive challenge
            deltas["patience"] -= 0.10 * max(frustration_val, 0.5)
        deltas["happiness"] -= 0.10 * max(frustration_val, guilt_val, 0.4)

    # High student stress / anxiety
    if stress_val > 0.4 or anxiety_val > 0.4:
        deltas["concern"] += 0.20 * max(stress_val, anxiety_val)
        deltas["warmth"] += 0.10 * max(stress_val, anxiety_val)  # Increase supportive warmth
        deltas["happiness"] -= 0.10 * max(stress_val, anxiety_val)
        deltas["energy"] += 0.05  # Lean in to assist

    # Student sadness / burnout / fatigue
    if sadness_val > 0.4 or fatigue_val > 0.4:
        deltas["sadness"] += 0.15 * sadness_val  # Empathic resonance
        deltas["concern"] += 0.18 * max(sadness_val, fatigue_val)
        deltas["warmth"] += 0.12 * max(sadness_val, fatigue_val)
        deltas["happiness"] -= 0.12 * max(sadness_val, fatigue_val)
        deltas["energy"] -= 0.08 * fatigue_val

    # Positive student feelings / motivation / accomplishment
    if happiness_val > 0.4 or motivation_val > 0.4:
        deltas["happiness"] += 0.20 * max(happiness_val, motivation_val)
        deltas["energy"] += 0.15 * max(happiness_val, motivation_val)
        deltas["warmth"] += 0.08
        deltas["patience"] += 0.12
        deltas["concern"] -= 0.15
        deltas["frustration"] -= 0.12

    # Goal relevance influence
    if goal_related and (frustration_val > 0.5 or has_avoidance):
        # Goal at stake amplifies concern
        deltas["concern"] += 0.08

    return deltas
