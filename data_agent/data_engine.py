"""Readable orchestration and validation for the isolated Data Agent V1."""

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set

from data_agent.data_llm import DataAgentLLM, data_agent_llm
from data_agent.data_schema import (
    CandidateMemory,
    DataAgentResult,
    ExistingLongTermMemory,
    MemoryActionType,
    Relationship,
    RelationshipEntityType,
)
from data_agent.mock_data_agent import MockDataAgent, mock_data_agent
from emotion_agent.emotion_schema import EmotionAnalysis


class DataAgentValidationError(ValueError):
    """Raised when input scope or LLM output violates memory safety rules."""


class DataAgent:
    """Expose legacy extraction plus isolated LLM memory consolidation."""

    def __init__(
        self,
        agent: MockDataAgent = mock_data_agent,
        reasoning_agent: Optional[DataAgentLLM] = None,
    ):
        self.agent = agent
        self.reasoning_agent = reasoning_agent or data_agent_llm

    def process(
        self,
        interaction: Dict[str, Any],
        emotion: EmotionAnalysis,
    ) -> List[CandidateMemory]:
        """Preserve the existing extraction interface used by older callers."""
        return self.agent.process(interaction, emotion)

    async def consolidate(
        self,
        candidate_memories: List[CandidateMemory],
        existing_long_term_memories: Optional[List[ExistingLongTermMemory]] = None,
        graph_context: Optional[List[Dict[str, Any]]] = None,
        semantic_context: Optional[List[Dict[str, Any]]] = None,
    ) -> DataAgentResult:
        """Ask the LLM to reason, then validate its proposed memory actions."""
        if not candidate_memories:
            return DataAgentResult()

        existing_memories = existing_long_term_memories or []
        user_id = self._validate_user_scope(candidate_memories, existing_memories)
        candidate_payload = self._prepare_candidates(candidate_memories)
        existing_payload = self._prepare_existing_memories(existing_memories)

        result = await self.reasoning_agent.reason(
            user_id=user_id,
            candidate_memories=candidate_payload,
            existing_long_term_memories=existing_payload,
            graph_context=list(graph_context or []),
            semantic_context=list(semantic_context or []),
        )

        self._validate_result_user(result, user_id)
        self._validate_memory_actions(result, candidate_memories, existing_memories)
        self._validate_relationships(result, candidate_memories, existing_memories)
        return result

    def _validate_user_scope(
        self,
        candidates: Sequence[CandidateMemory],
        existing_memories: Sequence[ExistingLongTermMemory],
    ) -> str:
        """Require one explicit user across every supplied memory object."""
        candidate_users = {
            candidate.user_id.strip()
            for candidate in candidates
            if candidate.user_id and candidate.user_id.strip()
        }
        if len(candidate_users) != 1 or any(
            not candidate.user_id or not candidate.user_id.strip()
            for candidate in candidates
        ):
            raise DataAgentValidationError(
                "All candidate memories must have the same non-empty user_id"
            )

        user_id = next(iter(candidate_users))
        if any(memory.user_id.strip() != user_id for memory in existing_memories):
            raise DataAgentValidationError(
                "Existing long-term memories must belong to the candidate user"
            )

        self._validate_unique_ids(candidates, existing_memories)
        return user_id

    @staticmethod
    def _validate_unique_ids(
        candidates: Sequence[CandidateMemory],
        existing_memories: Sequence[ExistingLongTermMemory],
    ) -> None:
        candidate_ids = [candidate.id for candidate in candidates]
        if any(not memory_id or not memory_id.strip() for memory_id in candidate_ids):
            raise DataAgentValidationError("Every candidate memory must have an ID")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise DataAgentValidationError("Candidate memory IDs must be unique")

        existing_ids = [memory.id for memory in existing_memories]
        if len(existing_ids) != len(set(existing_ids)):
            raise DataAgentValidationError("Existing long-term memory IDs must be unique")

    @staticmethod
    def _prepare_candidates(
        candidates: Sequence[CandidateMemory],
    ) -> List[Dict[str, Any]]:
        """Serialize immutable candidate evidence for the reasoning prompt."""
        return [candidate.model_dump(mode="json") for candidate in candidates]

    @staticmethod
    def _prepare_existing_memories(
        existing_memories: Sequence[ExistingLongTermMemory],
    ) -> List[Dict[str, Any]]:
        """Serialize existing memory context for UPDATE reasoning."""
        return [memory.model_dump(mode="json") for memory in existing_memories]

    @staticmethod
    def _validate_result_user(result: DataAgentResult, expected_user_id: str) -> None:
        if result.user_id != expected_user_id:
            raise DataAgentValidationError(
                "LLM result user_id does not match the validated input user"
            )

    def _validate_memory_actions(
        self,
        result: DataAgentResult,
        candidates: Sequence[CandidateMemory],
        existing_memories: Sequence[ExistingLongTermMemory],
    ) -> None:
        """Validate action IDs, candidate coverage, evidence, and UPDATE targets."""
        candidate_ids = {candidate.id for candidate in candidates if candidate.id}
        existing_ids = {memory.id for memory in existing_memories}
        allowed_evidence_ids = self._allowed_evidence_ids(candidates, existing_memories)

        action_ids = [action.action_id for action in result.memory_actions]
        if len(action_ids) != len(set(action_ids)):
            raise DataAgentValidationError("LLM action_id values must be unique")

        assigned_ids = [
            candidate_id
            for action in result.memory_actions
            for candidate_id in action.candidate_ids
        ]
        unknown_candidates = set(assigned_ids) - candidate_ids
        if unknown_candidates:
            raise DataAgentValidationError(
                f"LLM invented candidate IDs: {sorted(unknown_candidates)}"
            )
        if set(assigned_ids) != candidate_ids:
            missing_ids = candidate_ids - set(assigned_ids)
            raise DataAgentValidationError(
                f"Every candidate must receive one action; missing: {sorted(missing_ids)}"
            )
        duplicates = sorted(
            memory_id for memory_id, count in Counter(assigned_ids).items() if count > 1
        )
        if duplicates:
            raise DataAgentValidationError(
                f"Candidates may not appear in multiple actions: {duplicates}"
            )

        for action in result.memory_actions:
            if not action.candidate_ids:
                raise DataAgentValidationError("Every memory action needs candidate_ids")
            self._validate_evidence_ids(
                action.evidence_ids,
                allowed_evidence_ids,
                owner=f"action {action.action_id}",
            )
            if not set(action.candidate_ids).issubset(set(action.evidence_ids)):
                raise DataAgentValidationError(
                    f"Action {action.action_id} must retain its candidate IDs as evidence"
                )
            if action.action == MemoryActionType.UPDATE:
                if action.memory_id not in existing_ids:
                    raise DataAgentValidationError(
                        f"UPDATE references unknown memory_id: {action.memory_id}"
                    )

    def _validate_relationships(
        self,
        result: DataAgentResult,
        candidates: Sequence[CandidateMemory],
        existing_memories: Sequence[ExistingLongTermMemory],
    ) -> None:
        """Validate relationship evidence and any references with known identities."""
        allowed_evidence_ids = self._allowed_evidence_ids(candidates, existing_memories)
        candidate_ids = {candidate.id for candidate in candidates if candidate.id}
        existing_ids = {memory.id for memory in existing_memories}
        action_ids = {action.action_id for action in result.memory_actions}

        for relationship in result.relationships:
            if relationship.source_id == relationship.target_id:
                raise DataAgentValidationError("A relationship cannot point to itself")
            if not relationship.evidence_ids:
                raise DataAgentValidationError("Every relationship must contain evidence")
            self._validate_evidence_ids(
                relationship.evidence_ids,
                allowed_evidence_ids,
                owner="relationship",
            )
            self._validate_relationship_endpoint(
                relationship,
                endpoint="source",
                candidate_ids=candidate_ids,
                existing_ids=existing_ids,
                action_ids=action_ids,
            )
            self._validate_relationship_endpoint(
                relationship,
                endpoint="target",
                candidate_ids=candidate_ids,
                existing_ids=existing_ids,
                action_ids=action_ids,
            )

    @staticmethod
    def _allowed_evidence_ids(
        candidates: Sequence[CandidateMemory],
        existing_memories: Sequence[ExistingLongTermMemory],
    ) -> Set[str]:
        evidence = {candidate.id for candidate in candidates if candidate.id}
        for memory in existing_memories:
            evidence.update(memory.evidence_ids)
        return evidence

    @staticmethod
    def _validate_evidence_ids(
        evidence_ids: Sequence[str],
        allowed_ids: Set[str],
        *,
        owner: str,
    ) -> None:
        unknown_ids = set(evidence_ids) - allowed_ids
        if unknown_ids:
            raise DataAgentValidationError(
                f"{owner} contains invented evidence IDs: {sorted(unknown_ids)}"
            )

    @staticmethod
    def _validate_relationship_endpoint(
        relationship: Relationship,
        *,
        endpoint: str,
        candidate_ids: Set[str],
        existing_ids: Set[str],
        action_ids: Set[str],
    ) -> None:
        entity_id = getattr(relationship, f"{endpoint}_id")
        entity_type = getattr(relationship, f"{endpoint}_type")
        known_ids = {
            RelationshipEntityType.CANDIDATE_MEMORY: candidate_ids,
            RelationshipEntityType.LONG_TERM_MEMORY: existing_ids,
            RelationshipEntityType.MEMORY_ACTION: action_ids,
        }
        if entity_type in known_ids and entity_id not in known_ids[entity_type]:
            raise DataAgentValidationError(
                f"Relationship {endpoint} references unknown {entity_type.value}: {entity_id}"
            )


data_agent = DataAgent()
