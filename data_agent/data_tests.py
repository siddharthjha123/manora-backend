"""
MANORA Data Agent Tests.
Unit and validation tests for candidate memory extraction.
"""

import unittest
from data_agent.mock_data_agent import MockDataAgent
from emotion_agent.emotion_schema import EmotionAnalysis, EmotionItem, GoalRelevance


class TestDataAgentComponents(unittest.TestCase):
    def test_mock_data_agent_extraction(self):
        agent = MockDataAgent()
        interaction = {"raw_text": "I watched Netflix instead of studying."}
        analysis = EmotionAnalysis(
            interaction_id="test-id",
            primary_emotion="frustration",
            emotions=[EmotionItem(emotion="frustration", intensity=0.8, confidence=0.9, source="test")],
            emotional_summary="Frustrated with study avoidance",
            behavioral_signals=["avoided study", "watched Netflix"],
            decision_signals=["chose entertainment over study"],
            goal_relevance=GoalRelevance(related=True, goal="academics"),
        )
        candidates = agent.process(interaction, analysis)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].behavior.type, "avoidance")
        self.assertTrue(candidates[0].goal_relevance.related)


if __name__ == "__main__":
    unittest.main()
