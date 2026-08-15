"""
MANORA Data Agent - Engine Scaffolding.
Exposes the DataAgent entry point, currently delegating to MockDataAgent.
"""

from typing import Any, Dict, List
from data_agent.data_schema import CandidateMemory
from data_agent.mock_data_agent import MockDataAgent, mock_data_agent
from emotion_agent.emotion_schema import EmotionAnalysis


class DataAgent:
    """
    Data Agent entry point.
    In V1, delegates to MockDataAgent.
    In future versions, can be switched to LLMDataAgent seamlessly.
    """

    def __init__(self, agent: MockDataAgent = mock_data_agent):
        self.agent = agent

    def process(
        self,
        interaction: Dict[str, Any],
        emotion: EmotionAnalysis,
    ) -> List[CandidateMemory]:
        return self.agent.process(interaction, emotion)


# Global singleton instance
data_agent = DataAgent()
