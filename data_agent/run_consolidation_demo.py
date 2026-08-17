"""Manual real-model demo for Data Agent V1; performs no persistence."""

import asyncio
import json

from data_agent.data_engine import DataAgent
from data_agent.data_mockup import get_two_candidate_memory_models
from data_agent.data_schema import ExistingLongTermMemory, MemoryActionType


def print_actions(result, action_type: MemoryActionType) -> None:
    actions = [action for action in result.memory_actions if action.action == action_type]
    print(f"\n=== {action_type.value} ({len(actions)}) ===")
    for action in actions:
        print(json.dumps(action.model_dump(mode="json"), indent=2, ensure_ascii=False))


async def main() -> None:
    candidates = get_two_candidate_memory_models()

    existing_memory_single = ExistingLongTermMemory(
        id="mem_001",
        user_id="user_001",
        content=(
            "Student experiences loneliness and difficulty forming close "
            "relationships at college."
        ),
        emotions=[
            {
                "emotion": "loneliness",
                "confidence": 0.90
            }
        ],
        importance=0.88,
        confidence=0.90,
        evidence_ids=[
            "cm_old_001",
            "cm_old_002"
        ],
    )

    result = await DataAgent().consolidate(
        candidate_memories=candidates,
        existing_long_term_memories=[existing_memory_single],
        graph_context=[],
        semantic_context=[],
    )

    print(f"Validated user: {result.user_id}")
    print(f"Loaded candidate memories: {len(candidates)}")
    for action_type in MemoryActionType:
        print_actions(result, action_type)

    print(f"\n=== RELATIONSHIPS ({len(result.relationships)}) ===")
    for relationship in result.relationships:
        print(json.dumps(relationship.model_dump(mode="json"), indent=2))

    print("\n=== REASONING SUMMARY ===")
    print(result.reasoning_summary)


if __name__ == "__main__":
    asyncio.run(main())
