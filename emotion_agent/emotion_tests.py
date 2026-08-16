"""
MANORA Emotion Agent Tests.
Unit and validation tests for emotion schemas, parser, classifier, and agent.
"""

import unittest
from emotion_agent.emotion_parser import EmotionParser
from emotion_agent.emotion_schema import EmotionAnalysis, EmotionItem, GoalRelevance
from ml.emotion_classifier import RuleAndLexiconEmotionClassifier


class TestEmotionAgentComponents(unittest.TestCase):
    """Tests schema validation and parsing for Emotion Agent."""

    def test_emotion_schema_bounds(self):
        item = EmotionItem(
            emotion="frustration",
            intensity=1.5,  # should be clamped to 1.0
            confidence=-0.2,  # should be clamped to 0.0
            source="test",
        )
        self.assertEqual(item.intensity, 1.0)
        self.assertEqual(item.confidence, 0.0)

    def test_classifier_detection(self):
        classifier = RuleAndLexiconEmotionClassifier()
        res = classifier.predict("I planned to study but watched Netflix for two hours again. So frustrated.")
        self.assertIn("frustration", res)
        self.assertIn("guilt", res)
        self.assertGreater(res["frustration"], 0.2)
        self.assertGreater(res["guilt"], 0.2)

    def test_emotion_parser_valid_json(self):
        raw = """
        ```json
        {
            "interaction_id": "11111111-1111-1111-1111-111111111111",
            "primary_emotion": "frustration",
            "emotions": [
                {"emotion": "frustration", "intensity": 0.85, "confidence": 0.9, "source": "model_inferred"}
            ],
            "emotional_summary": "Student is feeling frustrated with study avoidance.",
            "behavioral_signals": ["watched series"],
            "decision_signals": ["delayed study"],
            "goal_relevance": {"related": true, "goal": "academics"}
        }
        ```
        """
        parsed = EmotionParser.parse(raw, "11111111-1111-1111-1111-111111111111")
        self.assertIsInstance(parsed, EmotionAnalysis)
        self.assertEqual(parsed.primary_emotion, "frustration")
        self.assertEqual(len(parsed.emotions), 1)
        self.assertTrue(parsed.goal_relevance.related)


if __name__ == "__main__":
    unittest.main()
