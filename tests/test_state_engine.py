"""
MANORA Tests - Buddy State Engine.
Validates deterministic state updates, dimension bounds, clamp limits, repeated emotional triggers, and temporal decay.
"""

from emotion_agent.emotion_schema import EmotionAnalysis, EmotionItem, GoalRelevance
from state.state_engine import BuddyState, StateEngine


class TestBuddyStateEngine:
    """Tests for deterministic Buddy State Engine transitions."""

    def test_initialize_state(self):
        engine = StateEngine()
        state = engine.initialize_state()
        assert isinstance(state, BuddyState)
        assert state.happiness == 0.60
        assert state.sadness == 0.10
        assert state.frustration == 0.10
        assert state.concern == 0.20
        assert state.warmth == 0.80
        assert state.patience == 0.80
        assert state.energy == 0.70

    def test_state_bounds_clamping(self):
        # Setting values out of range should be clamped
        state = BuddyState(
            happiness=1.5,
            sadness=-0.4,
            frustration=2.0,
            concern=0.5,
            warmth=1.0,
            patience=0.0,
            energy=0.7,
        )
        assert state.happiness == 1.0
        assert state.sadness == 0.0
        assert state.frustration == 1.0

    def test_update_state_on_study_avoidance(self):
        engine = StateEngine()
        initial = engine.initialize_state()

        # Simulate user avoiding study with high frustration and guilt
        analysis = EmotionAnalysis(
            interaction_id="test-id",
            primary_emotion="frustration",
            emotions=[
                EmotionItem(emotion="frustration", intensity=0.84, confidence=0.9, source="test"),
                EmotionItem(emotion="guilt", intensity=0.72, confidence=0.8, source="test"),
            ],
            emotional_summary="Student avoided planned study session and watched series.",
            behavioral_signals=["avoided planned study activity"],
            decision_signals=["chose entertainment instead of study"],
            goal_relevance=GoalRelevance(related=True, goal="academics"),
        )

        updated = engine.update_state(current_state=initial, emotion_analysis=analysis)

        # Concern and frustration should increase
        assert updated.concern > initial.concern
        assert updated.frustration > initial.frustration
        # Patience should slightly decrease
        assert updated.patience < initial.patience
        # All values must remain in [0.0, 1.0]
        for val in updated.to_dict().values():
            assert 0.0 <= val <= 1.0

    def test_repeated_emotional_triggers_do_not_overflow(self):
        engine = StateEngine()
        current = engine.initialize_state()

        analysis = EmotionAnalysis(
            interaction_id="test-id",
            primary_emotion="frustration",
            emotions=[EmotionItem(emotion="frustration", intensity=1.0, confidence=1.0, source="test")],
            emotional_summary="Repeated severe frustration",
            behavioral_signals=["avoided everything"],
            decision_signals=["gave up"],
            goal_relevance=GoalRelevance(related=True, goal="degree"),
        )

        # Apply 20 times in a row
        for _ in range(20):
            current = engine.update_state(current_state=current, emotion_analysis=analysis)

        for val in current.to_dict().values():
            assert 0.0 <= val <= 1.0
        assert current.concern == 1.0
        assert current.frustration == 1.0

    def test_apply_decay_drifts_to_baseline(self):
        engine = StateEngine()
        high_concern_state = BuddyState(
            happiness=0.20,
            sadness=0.50,
            frustration=0.80,
            concern=0.90,
            warmth=0.80,
            patience=0.40,
            energy=0.30,
        )

        # Apply 24 hours of decay
        decayed = engine.apply_decay(
            current_state=high_concern_state,
            time_delta_hours=24.0,
            decay_rate=0.10,
        )

        # Concern should have drifted downwards toward 0.20
        assert decayed.concern < high_concern_state.concern
        # Happiness should have drifted upwards toward 0.60
        assert decayed.happiness > high_concern_state.happiness
        # Patience should have drifted upwards toward 0.80
        assert decayed.patience > high_concern_state.patience
