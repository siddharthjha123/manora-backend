"""Run Data Agent V1 against the existing 16-candidate mock dataset.

Usage from the repository root:
    python -m data_agent.run_consolidation_demo
"""

import json

from data_agent.data_engine import DataAgent
from data_agent.data_mockup import get_candidate_memory_models


def _print_section(title, items):
    print(f"\n=== {title} ({len(items)}) ===")
    if not items:
        print("[]")
        return
    for item in items:
        print(json.dumps(item.model_dump(), indent=2, ensure_ascii=False))


def main():
    candidates = get_candidate_memory_models()
    result = DataAgent().consolidate(candidates)

    print(f"Loaded candidate memories: {len(candidates)}")
    _print_section("PROMOTED MEMORIES", result.promoted_memories)
    _print_section("PATTERNS", result.patterns)
    _print_section("RELATIONSHIPS", result.relationships)

    print(f"\n=== REJECTED CANDIDATES ({len(result.rejected_memory_ids)}) ===")
    print(json.dumps(result.rejected_memory_ids, indent=2))


if __name__ == "__main__":
    main()
