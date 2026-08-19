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
from data_agent.data_prompt import (
    build_data_agent_messages,
    build_stage1_messages,
    build_stage2_messages,
)
from data_agent.data_schema import (
    DataAgentResult,
    Stage1ConsolidationResult,
    Stage2MemoryDecisionResult,
)


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

    async def consolidate_candidates(
        self,
        *,
        user_id: str,
        candidate_memories: List[Dict[str, Any]],
    ) -> Stage1ConsolidationResult:
        """Ask Qwen to group candidate evidence without persistence reasoning."""

        return await self.client.generate_json(
            messages=build_stage1_messages(
                user_id=user_id,
                candidate_memories=candidate_memories,
            ),
            schema=Stage1ConsolidationResult,
            temperature=0.2,
            max_tokens=3000,
        )

    async def decide_memory_action(
        self,
        *,
        user_id: str,
        consolidated_memory: Dict[str, Any],
        existing_long_term_memories: List[Dict[str, Any]],
    ) -> Stage2MemoryDecisionResult:
        """Ask Qwen for CREATE, UPDATE, or REJECT using retrieved memories."""

        return await self.client.generate_json(
            messages=build_stage2_messages(
                user_id=user_id,
                consolidated_memory=consolidated_memory,
                existing_long_term_memories=existing_long_term_memories,
            ),
            schema=Stage2MemoryDecisionResult,
            temperature=0.2,
            max_tokens=2000,
        )


data_agent_llm = DataAgentLLM()
