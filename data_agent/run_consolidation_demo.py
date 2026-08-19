"""Live demonstration of the two-stage long-term-memory pipeline."""

import asyncio
import json
import logging

from data_agent.data_mockup import get_two_candidate_memory_models
from database.connection import db
from memory.memory_engine import MemoryEngine


def print_json(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    await db.initialize()

    try:
        candidates = get_two_candidate_memory_models()
        result = await MemoryEngine().process_long_term_memories(
            candidates,
            score_threshold=0.0,
        )

        print("\n========== STAGE 1: CONSOLIDATION ==========")
        print(f"\nCandidate memories: {len(candidates)}")
        print(f"Consolidated memories: {len(result.stage1.consolidated_memories)}")
        for memory in result.stage1.consolidated_memories:
            print_json(memory.model_dump(mode="json"))
        if result.stage1.rejected_candidate_ids:
            print("Rejected candidate IDs:", result.stage1.rejected_candidate_ids)

        print("\n========== QDRANT RETRIEVAL ==========")
        for retrieval in result.retrievals:
            print("\nQuery:")
            print_json(retrieval.query)
            print("Existing memories:")
            if not retrieval.existing_memories:
                print("  none")
            for memory in retrieval.existing_memories:
                score = memory.get("metadata", {}).get("score")
                print(f"  {memory['id']}   score={score}")

        print("\n========== STAGE 2: MEMORY DECISION ==========")
        for decision in result.decisions:
            print_json(decision.model_dump(mode="json"))

        print("\n========== PERSISTENCE ==========")
        for persisted in result.persistence:
            memory_id = persisted.memory_id or "none"
            print(f"PostgreSQL: {persisted.postgres_operation} {memory_id}")
            print(f"Qdrant: {persisted.qdrant_operation} {memory_id}")

        print("\n========== FINAL RESULT ==========")
        print_json(result.model_dump(mode="json"))
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
