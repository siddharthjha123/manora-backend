"""Deterministic, side-effect-free MANORA Data Agent V1.

The existing ``process`` method is intentionally preserved for the production
interaction pipeline.  ``consolidate`` is a separate batch boundary that turns
candidate evidence into higher-level knowledge without using an LLM or any
database/vector/graph service.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from statistics import fmean
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from data_agent.data_schema import (
    CandidateMemory,
    CandidateMemoryEmotion,
    DataAgentResult,
    Pattern,
    PromotedMemory,
    Relationship,
)
from data_agent.mock_data_agent import MockDataAgent, mock_data_agent
from emotion_agent.emotion_schema import EmotionAnalysis


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "in",
    "into", "is", "it", "its", "of", "on", "or", "other", "she", "student",
    "that", "the", "their", "them", "they", "this", "to", "was", "when",
    "while", "with", "would",
}

_SUBTOPIC_ALIASES = {
    "loneliness": "social_connection",
    "social_support": "social_connection",
    "social_connection": "social_connection",
    "loneliness_and_support": "social_connection",
    "social_comparison_fomo": "social_comparison",
    "social_comparison": "social_comparison",
    "study_concentration": "study_focus",
    "study_environment": "study_focus",
    "placement": "career_progress",
    "career_comparison": "career_progress",
    "friendship_loss": "friendship_trust",
    "trust_and_relationships": "friendship_trust",
    "sleep_and_overthinking": "sleep_overthinking",
    "digital_coping": "avoidance_coping",
    "emotional_support": "family_support",
    "future_aspirations": "future_goals",
}

_TOPIC_SUMMARIES = {
    "social_connection": (
        "Student experiences persistent loneliness and wants a close, genuine "
        "social connection they can rely on."
    ),
    "social_comparison": (
        "Student repeatedly compares their social or personal progress with others, "
        "which reinforces fear of missing out and feeling behind."
    ),
    "study_focus": (
        "Student has difficulty focusing while studying alone and concentrates better "
        "in a shared study environment such as a library."
    ),
    "career_progress": (
        "Student strongly values career and placement progress, while comparison with "
        "peers increases concern about falling behind."
    ),
    "friendship_trust": (
        "A past close friendship becoming distant contributes to hesitation about "
        "forming new close relationships."
    ),
}

_TOPIC_RELATIONSHIPS: Tuple[Tuple[str, str, str], ...] = (
    ("friendship_trust", "INFLUENCES", "social_connection"),
    ("social_comparison", "AMPLIFIES", "career_progress"),
    ("study_focus", "SUPPORTS", "future_goals"),
    ("career_progress", "ALIGNS_WITH", "future_goals"),
    ("social_connection", "ALIGNS_WITH", "future_goals"),
    ("avoidance_coping", "MAY_REINFORCE", "sleep_overthinking"),
)


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

    def consolidate(self, candidates: List[CandidateMemory]) -> DataAgentResult:
        """Consolidate candidate evidence into deterministic long-term knowledge.

        Candidates are partitioned by ``user_id`` before normalization, scoring,
        or similarity checks.  Missing user IDs are rejected as an invalid batch
        because treating them as one anonymous user would violate isolation.
        """
        if not candidates:
            return DataAgentResult()

        partitions: Dict[str, List[CandidateMemory]] = defaultdict(list)
        for candidate in candidates:
            if not candidate.user_id or not candidate.user_id.strip():
                raise ValueError(
                    "Every CandidateMemory passed to consolidate() must have user_id"
                )
            partitions[candidate.user_id.strip()].append(candidate)

        promoted: List[PromotedMemory] = []
        patterns: List[Pattern] = []
        relationships: List[Relationship] = []
        rejected_ids: List[str] = []

        for user_id in sorted(partitions):
            user_result = self._consolidate_user(user_id, partitions[user_id])
            promoted.extend(user_result.promoted_memories)
            patterns.extend(user_result.patterns)
            relationships.extend(user_result.relationships)
            rejected_ids.extend(user_result.rejected_memory_ids)

        return DataAgentResult(
            promoted_memories=promoted,
            patterns=patterns,
            relationships=relationships,
            rejected_memory_ids=sorted(set(rejected_ids)),
        )

    def _consolidate_user(
        self,
        user_id: str,
        candidates: Sequence[CandidateMemory],
    ) -> DataAgentResult:
        prepared: List[CandidateMemory] = []
        rejected: List[str] = []

        for candidate in sorted(candidates, key=self._candidate_sort_key):
            candidate_id = self._candidate_id(candidate, user_id)
            # CandidateMemory is evidence: never mutate the caller's object.
            prepared_candidate = candidate.model_copy(update={"id": candidate_id})
            if not self._is_promotable_evidence(prepared_candidate):
                rejected.append(candidate_id)
            else:
                prepared.append(prepared_candidate)

        groups = self._group_candidates(prepared)
        promoted = [self._promote_group(user_id, group) for group in groups]
        patterns = [
            self._build_pattern(memory, group)
            for memory, group in zip(promoted, groups)
            if len(group) >= 2
        ]
        relationships = self._build_relationships(user_id, promoted, patterns)

        return DataAgentResult(
            promoted_memories=promoted,
            patterns=patterns,
            relationships=relationships,
            rejected_memory_ids=sorted(rejected),
        )

    @staticmethod
    def _candidate_sort_key(candidate: CandidateMemory) -> Tuple[str, str]:
        return (candidate.id or "", candidate.content.casefold())

    @staticmethod
    def _stable_id(prefix: str, *parts: str) -> str:
        payload = "|".join(parts).encode("utf-8")
        return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:12]}"

    def _candidate_id(self, candidate: CandidateMemory, user_id: str) -> str:
        if candidate.id and candidate.id.strip():
            return candidate.id.strip()
        return self._stable_id("cm", user_id, self._normalize_text(candidate.content))

    @staticmethod
    def _is_promotable_evidence(candidate: CandidateMemory) -> bool:
        return (
            bool(candidate.content.strip())
            and candidate.importance >= 0.60
            and candidate.confidence >= 0.60
        )

    @staticmethod
    def _normalize_text(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9\s]", " ", value.casefold())
        return re.sub(r"\s+", " ", normalized).strip()

    def _tokens(self, value: str) -> Set[str]:
        return {
            token
            for token in self._normalize_text(value).split()
            if len(token) > 2 and token not in _STOP_WORDS
        }

    def _topic(self, candidate: CandidateMemory) -> str:
        subtopic = str(candidate.context.get("subtopic", "")).strip().casefold()
        if subtopic in _SUBTOPIC_ALIASES:
            return _SUBTOPIC_ALIASES[subtopic]

        topic = str(
            candidate.context.get("topic") or candidate.context.get("domain") or ""
        ).strip().casefold()
        topic = re.sub(r"[^a-z0-9]+", "_", topic).strip("_")
        if topic:
            return topic

        tokens = sorted(self._tokens(candidate.content))
        return "_".join(tokens[:3]) if tokens else "general"

    @staticmethod
    def _jaccard(left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _similarity(self, left: CandidateMemory, right: CandidateMemory) -> float:
        if self._normalize_text(left.content) == self._normalize_text(right.content):
            return 1.0

        token_score = self._jaccard(self._tokens(left.content), self._tokens(right.content))
        topic_match = self._topic(left) == self._topic(right)
        left_context = self._tokens(" ".join(map(str, left.context.values())))
        right_context = self._tokens(" ".join(map(str, right.context.values())))
        context_score = self._jaccard(left_context, right_context)
        return min(1.0, token_score * 0.55 + context_score * 0.15 + (0.55 if topic_match else 0.0))

    def _group_candidates(
        self,
        candidates: Sequence[CandidateMemory],
    ) -> List[List[CandidateMemory]]:
        """Build connected components from deterministic pairwise similarity."""
        if not candidates:
            return []

        parents = list(range(len(candidates)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                if self._similarity(candidates[left], candidates[right]) >= 0.55:
                    union(left, right)

        components: Dict[int, List[CandidateMemory]] = defaultdict(list)
        for index, candidate in enumerate(candidates):
            components[find(index)].append(candidate)

        groups = [
            sorted(group, key=self._candidate_sort_key)
            for group in components.values()
        ]
        return sorted(groups, key=lambda group: self._candidate_sort_key(group[0]))

    def _promote_group(
        self,
        user_id: str,
        group: Sequence[CandidateMemory],
    ) -> PromotedMemory:
        evidence_ids = sorted(candidate.id for candidate in group if candidate.id)
        topic_counts: Dict[str, int] = defaultdict(int)
        for candidate in group:
            topic_counts[self._topic(candidate)] += 1
        topic = sorted(topic_counts, key=lambda item: (-topic_counts[item], item))[0]

        importance = min(1.0, fmean(c.importance for c in group) + 0.04 * (len(group) - 1))
        confidence = min(1.0, fmean(c.confidence for c in group) + 0.03 * (len(group) - 1))
        memory_id = self._stable_id("pm", user_id, *evidence_ids)

        return PromotedMemory(
            id=memory_id,
            user_id=user_id,
            content=self._consolidated_content(topic, group),
            topic=topic,
            evidence_ids=evidence_ids,
            emotional_state=self._merge_emotions(group),
            contexts=self._unique_contexts(group),
            goals=self._collect_goals(group),
            support_count=len(group),
            importance=round(importance, 3),
            confidence=round(confidence, 3),
        )

    def _consolidated_content(
        self,
        topic: str,
        group: Sequence[CandidateMemory],
    ) -> str:
        if len(group) >= 2 and topic in _TOPIC_SUMMARIES:
            return _TOPIC_SUMMARIES[topic]

        ranked = sorted(
            group,
            key=lambda candidate: (
                -(candidate.importance + candidate.confidence),
                -len(candidate.content),
                candidate.id or "",
            ),
        )
        return ranked[0].content.strip()

    @staticmethod
    def _merge_emotions(
        group: Sequence[CandidateMemory],
    ) -> List[CandidateMemoryEmotion]:
        confidences: Dict[str, List[float]] = defaultdict(list)
        for candidate in group:
            for emotion in candidate.emotional_state:
                label = emotion.emotion.strip().casefold()
                if label:
                    confidences[label].append(emotion.confidence)

        merged = [
            CandidateMemoryEmotion(
                emotion=emotion,
                confidence=round(min(1.0, fmean(values) + 0.02 * (len(values) - 1)), 3),
            )
            for emotion, values in confidences.items()
        ]
        return sorted(merged, key=lambda item: (-item.confidence, item.emotion))

    @staticmethod
    def _unique_contexts(group: Sequence[CandidateMemory]) -> List[Dict[str, Any]]:
        seen: Set[str] = set()
        contexts: List[Dict[str, Any]] = []
        for candidate in group:
            marker = repr(sorted(candidate.context.items()))
            if marker not in seen:
                seen.add(marker)
                contexts.append(dict(candidate.context))
        return contexts

    @staticmethod
    def _collect_goals(group: Sequence[CandidateMemory]) -> List[str]:
        goals = {
            candidate.goal_relevance.goal.strip()
            for candidate in group
            if candidate.goal_relevance
            and candidate.goal_relevance.related
            and candidate.goal_relevance.goal
            and candidate.goal_relevance.goal.strip()
        }
        return sorted(goals)

    def _build_pattern(
        self,
        memory: PromotedMemory,
        group: Sequence[CandidateMemory],
    ) -> Pattern:
        normalized_contents = {self._normalize_text(candidate.content) for candidate in group}
        pattern_type = "duplicate_observation" if len(normalized_contents) == 1 else "recurring_theme"
        pattern_id = self._stable_id("pat", memory.user_id, memory.topic, *memory.evidence_ids)
        topic_label = memory.topic.replace("_", " ")
        return Pattern(
            id=pattern_id,
            user_id=memory.user_id,
            name=f"Repeated {topic_label}",
            description=(
                f"{len(group)} observations consistently support the {topic_label} theme."
            ),
            pattern_type=pattern_type,
            topic=memory.topic,
            evidence_ids=list(memory.evidence_ids),
            promoted_memory_ids=[memory.id],
            occurrence_count=len(group),
            emotional_state=list(memory.emotional_state),
            importance=memory.importance,
            confidence=memory.confidence,
        )

    def _build_relationships(
        self,
        user_id: str,
        memories: Sequence[PromotedMemory],
        patterns: Sequence[Pattern],
    ) -> List[Relationship]:
        relationships: List[Relationship] = []

        for pattern in patterns:
            memory_id = pattern.promoted_memory_ids[0]
            relationships.append(
                self._relationship(
                    user_id=user_id,
                    source_id=pattern.id,
                    source_type="pattern",
                    relation="SUMMARIZES",
                    target_id=memory_id,
                    target_type="promoted_memory",
                    evidence_ids=pattern.evidence_ids,
                    confidence=pattern.confidence,
                )
            )

        by_topic = {memory.topic: memory for memory in memories}
        for source_topic, relation, target_topic in _TOPIC_RELATIONSHIPS:
            source = by_topic.get(source_topic)
            target = by_topic.get(target_topic)
            if not source or not target:
                continue
            relationships.append(
                self._relationship(
                    user_id=user_id,
                    source_id=source.id,
                    source_type="promoted_memory",
                    relation=relation,
                    target_id=target.id,
                    target_type="promoted_memory",
                    evidence_ids=sorted(set(source.evidence_ids + target.evidence_ids)),
                    confidence=round(min(source.confidence, target.confidence) * 0.85, 3),
                )
            )

        return sorted(relationships, key=lambda item: item.id)

    def _relationship(
        self,
        *,
        user_id: str,
        source_id: str,
        source_type: str,
        relation: str,
        target_id: str,
        target_type: str,
        evidence_ids: Iterable[str],
        confidence: float,
    ) -> Relationship:
        rel_id = self._stable_id("rel", user_id, source_id, relation, target_id)
        return Relationship(
            id=rel_id,
            user_id=user_id,
            source_id=source_id,
            source_type=source_type,
            relation=relation,
            target_id=target_id,
            target_type=target_type,
            evidence_ids=sorted(set(evidence_ids)),
            confidence=confidence,
        )


# Global singleton instance
data_agent = DataAgent()
