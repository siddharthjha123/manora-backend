"""
Data Agent LLM reasoning layer.

This module prepares the Data Agent request and asks the
dedicated Data Agent LLM client to reason about memory.
"""

from typing import Any, Dict, List

from data_agent.data_llm_client import (
    DataAgentLLMClient,
    data_agent_llm_client,
)
from data_agent.data_prompt import build_data_agent_messages
from data_agent.data_schema import DataAgentResult


class DataAgentLLM:
    """
    Converts Data Agent memory context into an LLM request.

    This class does not know how HTTP requests work.
    That responsibility belongs to DataAgentLLMClient.
    """

    def __init__(
        self,
        client: DataAgentLLMClient | None = None,
    ):
        self.client = client or data_agent_llm_client

    async def reason(
        self,
        *,
        user_id: str,
        candidate_memories: List[Dict[str, Any]],
        existing_long_term_memories: List[Dict[str, Any]],
        graph_context: List[Dict[str, Any]],
        semantic_context: List[Dict[str, Any]],
    ) -> DataAgentResult:
        """
        Ask the Data Agent LLM to consolidate candidate memories.
        """

        messages = build_data_agent_messages(
            user_id=user_id,
            candidate_memories=candidate_memories,
            existing_long_term_memories=existing_long_term_memories,
            graph_context=graph_context,
            semantic_context=semantic_context,
        )

        result = await self.client.generate_json(
            messages=messages,
            schema=DataAgentResult,
            temperature=0.2,
            max_tokens=3000,
        )

        return result


data_agent_llm = DataAgentLLM()