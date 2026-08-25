"""
MANORA Data Agent - Parser Scaffolding.
Output parser for candidate memory extraction.
"""

import json
from typing import Any, Dict, List
from data_agent.data_schema import CandidateMemory, DataAgentOutput


class DataParser:
    """Parses LLM output into validated CandidateMemory instances."""

    @staticmethod
    def parse(raw_text: str) -> List[CandidateMemory]:
        try:
            data = json.loads(raw_text)
            output = DataAgentOutput.model_validate(data)
            return output.candidate_memories
        except Exception:
            return []
