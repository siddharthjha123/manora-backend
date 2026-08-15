"""
MANORA Tests - Emotion Agent and Classifier.
Validates emotion classifier probabilities, schema bounds, parser robustness, and emotion agent flow.
"""

import pytest
from emotion_agent.emotion_engine import EmotionAgent
from emotion_agent.emotion_parser import EmotionParser
from emotion_agent.emotion_schema import EmotionAnalysis, EmotionItem, GoalRelevance
from ml.emotion_classifier import RuleAndLexiconEmotionClassifier


class TestEmotionClassifier:
    """Tests for ML emotion classification logic."""

    def test_classifier_keyword_detection(self):
        classifier = RuleAndLexiconEmotionClassifier()
        result = classifier.predict("I planned to study at 10 but I ended up watching a series for two hours. So frustrated.")
        assert "frustration" in result
        assert "guilt" in result
        assert result["frustration"] >= 0.3
        assert result["guilt"] >= 0.3
        assert all(0.0 <= v <= 1.0 for v in result.values())

    def test_classifier_empty_input(self):
        classifier = RuleAndLexiconEmotionClassifier()
        result = classifier.predict("")
        assert isinstance(result, dict)
        assert all(0.0 <= v <= 1.0 for v in result.values())


class TestEmotionSchemaAndParser:
    """Tests for schema validation and LLM parsing."""

    def test_emotion_item_clamping(self):
        item = EmotionItem(
            emotion="stress",
            intensity=1.8,
            confidence=-0.5,
            source="test",
        )
        assert item.intensity == 1.0
        assert item.confidence == 0.0

    def test_emotion_parser_valid_json(self):
        json_text = """
        ```json
        {
            "interaction_id": "test-interaction-123",
            "primary_emotion": "frustration",
            "emotions": [
                {"emotion": "frustration", "intensity": 0.85, "confidence": 0.90, "source": "model_inferred"},
                {"emotion": "guilt", "intensity": 0.70, "confidence": 0.80, "source": "model_inferred"}
            ],
            "emotional_summary": "Student is feeling guilty and frustrated due to study avoidance.",
            "behavioral_signals": ["avoided planned study session", "watched entertainment"],
            "decision_signals": ["chose watching over studying"],
            "goal_relevance": {"related": true, "goal": "academic_progress"}
        }
        ```
        """
        parsed = EmotionParser.parse(json_text, interaction_id="test-interaction-123")
        assert isinstance(parsed, EmotionAnalysis)
        assert parsed.interaction_id == "test-interaction-123"
        assert parsed.primary_emotion == "frustration"
        assert len(parsed.emotions) == 2
        assert parsed.goal_relevance.related is True
        assert len(parsed.behavioral_signals) == 2

    def test_emotion_parser_malformed_json_fallback(self):
        malformed = "This is not valid json at all!"
        parsed = EmotionParser.parse(malformed, interaction_id="test-fallback-id")
        assert isinstance(parsed, EmotionAnalysis)
        assert parsed.interaction_id == "test-fallback-id"
        assert parsed.primary_emotion == "neutral"


@pytest.mark.asyncio
async def test_emotion_agent_execution():
    """Tests EmotionAgent end-to-end reasoning and analysis."""
    agent = EmotionAgent()
    analysis = await agent.analyze(
        interaction_id="test-interaction-uuid",
        user_id="test-user-uuid",
        session_id="test-session-uuid",
        text="I am really stressed about my upcoming placement interviews and cannot sleep.",
    )
    assert isinstance(analysis, EmotionAnalysis)
    assert analysis.interaction_id == "test-interaction-uuid"
    assert analysis.primary_emotion is not None
    assert len(analysis.emotional_summary) > 0
