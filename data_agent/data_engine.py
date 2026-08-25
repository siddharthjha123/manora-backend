"""Readable orchestration and validation for the isolated Data Agent V1."""

from collections import Counter
import logging
from typing import Any, Dict, List, Optional, Sequence, Set

from data_agent.data_llm import DataAgentLLM, data_agent_llm
from data_agent.data_schema import (
    CandidateMemory,
    CandidateMemoryBehavior,
    CandidateMemoryDecision,
    CandidateMemoryEmotion,
    CandidateMemoryEvent,
    CandidateMemoryGoalRelevance,
    ConsolidatedMemory,
    ExistingLongTermMemory,
    MemoryDecision,
    MemoryDecisionType,
    Stage1ConsolidationResult,
)
from emotion_agent.emotion_schema import EmotionAnalysis


logger = logging.getLogger("manora.data_agent")


class DataAgentValidationError(ValueError):
    """Raised when input scope or LLM output violates memory safety rules."""


class DataAgent:
    """Extract candidate evidence and perform the two LLM memory stages."""

    MEMORY_WORTHY_EMOTIONS = {
        "anger",
        "anxiety",
        "burnout",
        "fear",
        "frustration",
        "guilt",
        "hopelessness",
        "insecurity",
        "isolation",
        "loneliness",
        "overwhelm",
        "sadness",
        "stress",
        "worry",
    }
    MEMORY_WORTHY_SIGNAL_MARKERS = {
        "alone",
        "avoid",
        "cannot",
        "commit",
        "conflict",
        "delay",
        "distract",
        "fail",
        "goal",
        "intend",
        "isolate",
        "keep",
        "plan",
        "procrast",
        "repeat",
        "routine",
        "skip",
        "struggle",
        "unable",
        "withdraw",
    }

    def __init__(
        self,
        reasoning_agent: Optional[DataAgentLLM] = None,
    ):
        self.reasoning_agent = reasoning_agent or data_agent_llm

    def process(
        self,
        interaction: Dict[str, Any],
        emotion: EmotionAnalysis,
    ) -> List[CandidateMemory]:
        """Create candidate evidence directly from structured Emotion Agent output.

        Positive greetings and other ordinary conversation are intentionally not
        promoted merely because an emotion model assigns high intensity. Durable
        negative emotion, behavioral signals, or decision signals can produce a
        candidate for Stage 1; Stage 2 remains responsible for final persistence.
        """

        raw_text = str(interaction.get("raw_text") or "").strip()
        if not raw_text:
            return []

        meaningful_emotions = [
            item
            for item in emotion.emotions
            if item.emotion.lower() in self.MEMORY_WORTHY_EMOTIONS
            and item.intensity >= 0.6
        ]
        signal_text = " ".join(
            emotion.behavioral_signals + emotion.decision_signals
        ).lower()
        has_durable_signal = any(
            marker in signal_text
            for marker in self.MEMORY_WORTHY_SIGNAL_MARKERS
        )
        if not meaningful_emotions and not has_durable_signal:
            return []

        candidate_emotions = [
            CandidateMemoryEmotion(
                emotion=item.emotion,
                confidence=round(item.confidence, 3),
            )
            for item in (meaningful_emotions or emotion.emotions[:2])
        ]
        events = [
            CandidateMemoryEvent(type="behavioral_signal", description=signal)
            for signal in emotion.behavioral_signals
        ]
        behavior = (
            CandidateMemoryBehavior(
                type="observed_behavior",
                description=emotion.behavioral_signals[0],
            )
            if emotion.behavioral_signals
            else None
        )
        decision = (
            CandidateMemoryDecision(description=emotion.decision_signals[0])
            if emotion.decision_signals
            else None
        )

        maximum_intensity = max(
            (item.intensity for item in meaningful_emotions),
            default=0.6,
        )
        importance = 0.62 + (0.18 * maximum_intensity)
        if has_durable_signal:
            importance += 0.06
        if emotion.goal_relevance.related:
            importance += 0.05
        confidence = (
            sum(item.confidence for item in candidate_emotions)
            / len(candidate_emotions)
            if candidate_emotions
            else 0.75
        )

        return [
            CandidateMemory(
                content=raw_text,
                context={
                    "topic": (
                        "goal_related"
                        if emotion.goal_relevance.related
                        else "emotional_wellbeing"
                    ),
                    "emotional_summary": emotion.emotional_summary,
                },
                emotional_state=candidate_emotions,
                events=events,
                behavior=behavior,
                decision=decision,
                goal_relevance=CandidateMemoryGoalRelevance(
                    related=emotion.goal_relevance.related,
                    goal=emotion.goal_relevance.goal,
                ),
                importance=round(min(0.95, importance), 3),
                confidence=round(confidence, 3),
            )
        ]

    async def consolidate_candidates(
        self,
        candidate_memories: List[CandidateMemory],
    ) -> Stage1ConsolidationResult:
        """Group candidate evidence into consolidated memories using Stage 1 Qwen."""

        if not candidate_memories:
            return Stage1ConsolidationResult()

        user_id = self._validate_user_scope(candidate_memories, [])
        logger.info("Stage 1: consolidating candidate memories")
        result = await self.reasoning_agent.consolidate_candidates(
            user_id=user_id,
            candidate_memories=self._prepare_candidates(candidate_memories),
        )
        self._validate_stage1_result(result, candidate_memories, user_id)
        logger.info(
            "Stage 1 complete: %d consolidated memories",
            len(result.consolidated_memories),
        )
        return result

    async def decide_memory_actions(
        self,
        *,
        user_id: str,
        consolidated_memory: ConsolidatedMemory,
        existing_long_term_memories: List[ExistingLongTermMemory],
    ) -> MemoryDecision:
        """Choose CREATE, UPDATE, or REJECT for one consolidated memory."""

        if any(memory.user_id != user_id for memory in existing_long_term_memories):
            raise DataAgentValidationError(
                "Stage 2 existing memories must belong to the consolidated-memory user"
            )
        existing_ids = [memory.id for memory in existing_long_term_memories]
        if len(existing_ids) != len(set(existing_ids)):
            raise DataAgentValidationError("Stage 2 existing memory IDs must be unique")

        logger.info("Stage 2: deciding memory action")
        result = await self.reasoning_agent.decide_memory_action(
            user_id=user_id,
            consolidated_memory=consolidated_memory.model_dump(mode="json"),
            existing_long_term_memories=self._prepare_existing_memories(
                existing_long_term_memories
            ),
        )
        if result.user_id != user_id:
            raise DataAgentValidationError("Stage 2 result changed the user_id")

        decision = result.decision
        if (
            len(decision.candidate_ids) != len(consolidated_memory.candidate_ids)
            or set(decision.candidate_ids) != set(consolidated_memory.candidate_ids)
        ):
            raise DataAgentValidationError(
                "Stage 2 must preserve the consolidated candidate_ids"
            )

        allowed_evidence = set(consolidated_memory.evidence_ids)
        for memory in existing_long_term_memories:
            allowed_evidence.update(memory.evidence_ids)
        if not set(consolidated_memory.evidence_ids).issubset(decision.evidence_ids):
            raise DataAgentValidationError(
                "Stage 2 must preserve consolidated evidence_ids"
            )
        self._validate_evidence_ids(
            decision.evidence_ids,
            allowed_evidence,
            owner="Stage 2 decision",
        )
        if decision.action == MemoryDecisionType.UPDATE:
            if decision.memory_id not in set(existing_ids):
                raise DataAgentValidationError(
                    f"UPDATE references unknown memory_id: {decision.memory_id}"
                )

        logger.info(
            "Stage 2 decision: %s %s",
            decision.action.value,
            decision.memory_id or "new memory",
        )
        return decision

    def _validate_stage1_result(
        self,
        result: Stage1ConsolidationResult,
        candidates: Sequence[CandidateMemory],
        user_id: str,
    ) -> None:
        """Ensure Stage 1 covers every candidate once and preserves evidence."""

        if result.user_id != user_id:
            raise DataAgentValidationError("Stage 1 result changed the user_id")

        candidate_ids = {candidate.id for candidate in candidates if candidate.id}
        consolidation_ids = [
            memory.consolidation_id for memory in result.consolidated_memories
        ]
        if len(consolidation_ids) != len(set(consolidation_ids)):
            raise DataAgentValidationError("Stage 1 consolidation_id values must be unique")

        assigned_ids = [
            candidate_id
            for memory in result.consolidated_memories
            for candidate_id in memory.candidate_ids
        ] + list(result.rejected_candidate_ids)
        if set(assigned_ids) != candidate_ids:
            missing = candidate_ids - set(assigned_ids)
            invented = set(assigned_ids) - candidate_ids
            raise DataAgentValidationError(
                f"Stage 1 candidate coverage is invalid; missing={sorted(missing)}, "
                f"invented={sorted(invented)}"
            )
        duplicates = [item for item, count in Counter(assigned_ids).items() if count > 1]
        if duplicates:
            raise DataAgentValidationError(
                f"Stage 1 assigned candidates more than once: {sorted(duplicates)}"
            )

        for memory in result.consolidated_memories:
            if not set(memory.candidate_ids).issubset(memory.evidence_ids):
                raise DataAgentValidationError(
                    f"{memory.consolidation_id} did not preserve candidate evidence"
                )
            self._validate_evidence_ids(
                memory.evidence_ids,
                candidate_ids,
                owner=memory.consolidation_id,
            )

        known_groups = set(consolidation_ids)
        for relationship in result.relationships:
            if relationship.source_id not in known_groups:
                raise DataAgentValidationError("Stage 1 relationship has unknown source")
            if relationship.target_id not in known_groups:
                raise DataAgentValidationError("Stage 1 relationship has unknown target")
            if relationship.source_id == relationship.target_id:
                raise DataAgentValidationError("Stage 1 relationship cannot self-reference")
            if not relationship.evidence_ids:
                raise DataAgentValidationError(
                    "Stage 1 relationships must preserve evidence_ids"
                )
            self._validate_evidence_ids(
                relationship.evidence_ids,
                candidate_ids,
                owner="Stage 1 relationship",
            )


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

data_agent = DataAgent()