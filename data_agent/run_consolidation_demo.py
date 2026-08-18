"""Manual real-model demo for Data Agent V1; performs no persistence."""

import asyncio
import json

from data_agent.data_engine import DataAgent
from data_agent.data_mockup import get_two_candidate_memory_models
from data_agent.data_schema import ExistingLongTermMemory, MemoryActionType
from memory.memory_engine import MemoryEngine 
from database.connection import db

def print_actions(result, action_type: MemoryActionType) -> None:
    actions = [action for action in result.memory_actions if action.action == action_type]
    print(f"\n=== {action_type.value} ({len(actions)}) ===")
    for action in actions:
        print(json.dumps(action.model_dump(mode="json"), indent=2, ensure_ascii=False))


async def main() -> None:
    await db.initialize()

    print("PostgreSQL connected:", db.is_postgres_connected)

    try:

        candidates = get_two_candidate_memory_models()

        result = await DataAgent().consolidate(
            candidate_memories=candidates,
            existing_long_term_memories=[],
            graph_context=[],
            semantic_context=[],
        )
        
        persisted_memories = await MemoryEngine().persist_memory_candidates(result)

        print("---------Persisted Memories---------")
        print(persisted_memories)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
