"""
MANORA Tests - Buddy Agent.
Validates response schemas, expression parsing, tone normalization, and response generation.
"""

import pytest
from buddy.buddy_engine import BuddyAgent
from buddy.buddy_schema import BuddyExpression, BuddyResponse, BuddyResponseType
from emotion_agent.emotion_schema import EmotionAnalysis, EmotionItem, GoalRelevance
from state.state_engine import BuddyState


class TestBuddyAgent:
    """Tests for Buddy Agent schema validation and generation."""

    def test_buddy_response_schema_normalization(self):
        # Test expression and response_type synonyms and bounds
        resp = BuddyResponse(
            text="Let's take a look at your study goals together.",
            expression="worried",  # should normalize to 'concerned'
            intensity=1.8,  # should clamp to 1.0
            response_type="support",
        )
        assert resp.expression == BuddyExpression.CONCERNED.value
        assert resp.intensity == 1.0
        assert resp.response_type == BuddyResponseType.SUPPORT.value

    @pytest.mark.asyncio
    async def test_buddy_agent_generate(self):
        agent = BuddyAgent()
        analysis = EmotionAnalysis(
            interaction_id="test-buddy-interaction",
            primary_emotion="frustration",
            emotions=[EmotionItem(emotion="frustration", intensity=0.8, confidence=0.9, source="test")],
            emotional_summary="Student is struggling with recurring procrastination.",
            behavioral_signals=["avoided study session"],
            decision_signals=["watched series instead"],
            goal_relevance=GoalRelevance(related=True, goal="placement_prep"),
        )
        state = BuddyState(
            happiness=0.4,
            sadness=0.1,
            frustration=0.5,
            concern=0.7,
            warmth=0.8,
            patience=0.6,
            energy=0.7,
        )

        response = await agent.generate(
            user_text="I planned to study at 10 but watched Netflix for 2 hours again.",
            emotion_analysis=analysis,
            buddy_state=state,
        )

        assert isinstance(response, BuddyResponse)
        assert len(response.text) > 0
        assert response.expression in [e.value for e in BuddyExpression]
        assert 0.0 <= response.intensity <= 1.0
