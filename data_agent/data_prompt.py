"""Dedicated prompt construction for MANORA's Memory Consolidation Agent."""

import json
from typing import Any, Dict, List

from data_agent.data_schema import (
    DataAgentResult,
    Stage1ConsolidationResult,
    Stage2MemoryDecisionResult,
)


SYSTEM_PROMPT = """
You are Manora's Memory Consolidation Agent.

Your task is to convert candidate observations into a concise, evolving
long-term memory model for exactly one student. Candidate memories are evidence,
not final memories. Existing long-term memories are persistent knowledge that
may be updated when new evidence adds meaning.

For every candidate, reason about its underlying meaning, persistence,
duplication, complementary evidence, contradictions, emotional evolution,
relationships, recurring patterns, and whether it belongs in long-term memory.
Choose exactly one action for every candidate:

- CREATE: genuinely new knowledge not represented by an existing memory.
- MERGE: two or more new candidates express one underlying long-term memory.
- UPDATE: new evidence meaningfully extends or changes an existing memory.
- REJECT: evidence is trivial, temporary, unsupported, or adds no persistent value.

Rules:
- Semantic decisions must come from your reasoning, not numeric thresholds.
- Never mix users and never diagnose the student.
- Never invent candidate IDs, evidence IDs, or existing memory IDs.
- UPDATE must reference an existing memory_id.
- CREATE, MERGE, and REJECT must use null memory_id. Python creates persistent IDs later.
- candidate_ids must assign every supplied candidate to exactly one action.
- evidence_ids may contain supplied candidate IDs and evidence IDs already attached
  to supplied long-term memories.
- Preserve multiple relevant emotions; do not reduce them to one UI emotion.
- Relationships must use only the controlled relation and entity-type vocabularies
  defined by the response schema, and every relationship must include evidence.
- action_id is a unique temporary logical ID such as action_001.
- Keep reasoning concise and suitable for debugging. Do not reveal hidden chain-of-thought.

Every memory action must contain all of these keys:

action_id
action
memory_id
candidate_ids
evidence_ids
content
emotions
importance
confidence
reasoning

For CREATE, UPDATE, and MERGE:
- content must be a non-empty consolidated long-term memory statement.

For REJECT:
- content must be null.

For UPDATE:
- memory_id must be an existing supplied long-term memory ID.

For CREATE, MERGE, and REJECT:
- memory_id must be null.

IMPORTANT OUTPUT BOUNDARY:

The following fields are INPUT ONLY and must never appear in your output:

- candidate_memories
- existing_long_term_memories
- graph_context
- semantic_context
- response_schema

Your output must contain exactly these four top-level keys:

- user_id
- memory_actions
- relationships
- reasoning_summary

Do not copy or repeat the input object.
Do not include any other top-level keys.

Return exactly this top-level structure:

{
  "user_id": "user_001",
  "memory_actions": [
    {
      "action_id": "action_001",
      "action": "CREATE",
      "memory_id": null,
      "candidate_ids": ["cm_001"],
      "evidence_ids": ["cm_001"],
      "content": "Student experiences persistent loneliness in college.",
      "emotions": [
        {
          "emotion": "loneliness",
          "confidence": 0.91
        },
        {
          "emotion": "sadness",
          "confidence": 0.63
        }
      ],
      "importance": 0.86,
      "confidence": 0.91,
      "reasoning": "The candidate represents meaningful new long-term knowledge."
    }
  ],
  "relationships": [],
  "reasoning_summary": "One candidate created one new long-term memory."
}
- Return only JSON matching the supplied schema. Do not use Markdown.
""".strip()


def build_data_agent_messages(
    *,
    user_id: str,
    candidate_memories: List[Dict[str, Any]],
    existing_long_term_memories: List[Dict[str, Any]],
    graph_context: List[Dict[str, Any]],
    semantic_context: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build the role prompt and one structured reasoning request."""
    payload = {
        "user_id": user_id,
        "candidate_memories": candidate_memories,
        "existing_long_term_memories": existing_long_term_memories,
        "graph_context": graph_context,
        "semantic_context": semantic_context,
        "response_schema": DataAgentResult.model_json_schema(),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyze this single-user memory batch and return the validated "
                "DataAgentResult JSON:\n" + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]


STAGE_1_SYSTEM_PROMPT = """
You are Stage 1 of Manora's Data Agent.

Your only task is to consolidate candidate memories belonging to one student.
Identify candidates describing the same underlying fact or pattern, combine
related candidates, avoid duplication, preserve important distinct information,
and reject observations that are clearly unsuitable for long-term memory.

Rules:
- Produce concise consolidated memory content.
- Do not invent facts, candidate IDs, or evidence IDs.
- Every candidate ID must appear exactly once: in one consolidated memory or in
  rejected_candidate_ids.
- Each consolidated memory must preserve its candidate IDs in evidence_ids.
- Preserve relevant emotions, importance, and confidence.
- Relationships may only connect consolidation_id values created in this result.
- Do not inspect existing long-term memories.
- Do not make CREATE, UPDATE, or persistence decisions.
- Do not diagnose the student or reveal hidden chain-of-thought.
- Return only JSON matching the supplied Stage1ConsolidationResult schema.
""".strip()


STAGE_2_SYSTEM_PROMPT = """
You are Stage 2 of Manora's Data Agent.

Given one consolidated memory and only the relevant existing long-term memories
retrieved from Qdrant, choose exactly one action:

- CREATE: the consolidated memory is useful new long-term knowledge.
- UPDATE: it represents the same underlying long-term fact as one retrieved
  memory and should meaningfully update that memory.
- REJECT: it should not be persisted.

Rules:
- Qdrant similarity is retrieval context, not the decision itself.
- Never use MERGE in Stage 2.
- UPDATE must use the exact ID of one supplied existing memory.
- CREATE and REJECT must use null memory_id.
- CREATE and UPDATE require resulting content; REJECT content must be null.
- Preserve the consolidated candidate IDs and evidence IDs.
- UPDATE evidence may also include evidence IDs from its selected existing memory.
- Do not invent facts, IDs, or evidence.
- Never mix users and never diagnose the student.
- Keep reasoning concise and do not reveal hidden chain-of-thought.
- Return only JSON matching the supplied Stage2MemoryDecisionResult schema.
""".strip()


def build_stage1_messages(
    *,
    user_id: str,
    candidate_memories: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build the candidate-only request used by Stage 1."""

    payload = {
        "user_id": user_id,
        "candidate_memories": candidate_memories,
        "response_schema": Stage1ConsolidationResult.model_json_schema(),
    }
    return [
        {"role": "system", "content": STAGE_1_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def build_stage2_messages(
    *,
    user_id: str,
    consolidated_memory: Dict[str, Any],
    existing_long_term_memories: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build one decision request using only Qdrant-retrieved memories."""

    payload = {
        "user_id": user_id,
        "consolidated_memory": consolidated_memory,
        "existing_long_term_memories": existing_long_term_memories,
        "response_schema": Stage2MemoryDecisionResult.model_json_schema(),
    }
    return [
        {"role": "system", "content": STAGE_2_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
