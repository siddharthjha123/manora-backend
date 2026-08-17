"""Data Agent-specific use of the shared project LLM client."""

from typing import Any, Dict, List, Optional

from data_agent.data_prompt import build_data_agent_messages
from data_agent.data_schema import DataAgentResult
from llm.base import LLMClient, llm_client


class DataAgentLLM:
    """Call the shared configured model for memory reasoning only."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or llm_client

    async def reason(
        self,
        *,
        user_id: str,
        candidate_memories: List[Dict[str, Any]],
        existing_long_term_memories: List[Dict[str, Any]],
        graph_context: List[Dict[str, Any]],
        semantic_context: List[Dict[str, Any]],
    ) -> DataAgentResult:
        messages = build_data_agent_messages(
            user_id=user_id,
            candidate_memories=candidate_memories,
            existing_long_term_memories=existing_long_term_memories,
            graph_context=graph_context,
            semantic_context=semantic_context,
        )
        return await self.client.generate_json(
            messages=messages,
            schema=DataAgentResult,
            temperature=0.2,
        )


data_agent_llm = DataAgentLLM()
