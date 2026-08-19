# MANORA Backend Developer Documentation

This document is a practical guide for a developer who is seeing the MANORA
backend for the first time. It explains what the important files do, why they
exist, how data moves through the application, which layer owns each decision,
and where a change should be made without breaking unrelated features.

## 1. What MANORA is

MANORA is a student-support backend built around three different responsibilities:

1. Understand the student's current message and emotional state.
2. Maintain evidence-backed long-term knowledge without storing every message.
3. Generate a contextual Buddy response and support read-only exploration tools.

The backend uses FastAPI. PostgreSQL is the source of truth for structured
records, Qdrant is the semantic index, and Neo4j stores memory relationships.
LLM calls are separated by purpose: the Emotion Agent interprets the current
message, the Data Agent reasons about memory, and the Buddy Agent writes the
conversation response.

## 2. Recommended reading order

Read files in this order when reverse-engineering the project:

1. `main.py`
2. `api/interactions.py`
3. `services/interaction_service.py`
4. `config/settings.py`
5. `database/connection.py`
6. `ml/emotion_classifier.py`
7. `emotion_agent/emotion_schema.py`
8. `emotion_agent/emotion_engine.py`
9. `data_agent/data_schema.py`
10. `data_agent/data_engine.py`
11. `data_agent/data_prompt.py`
12. `data_agent/data_llm.py`
13. `data_agent/data_llm_client.py`
14. `memory/memory_engine.py`
15. `vector_db/qdrant_client.py`
16. `graph_db/neo4j_client.py`
17. `state/state_engine.py`
18. `buddy/buddy_schema.py`
19. `buddy/buddy_engine.py`
20. `memory/memory_tree_schema.py`
21. `memory/memory_tree_service.py`
22. `api/memory_tree.py`
23. `alternate_timeline/timeline_schema.py`
24. `alternate_timeline/timeline_service.py`
25. `api/alternate_timeline.py`
26. `tests/test_interaction_memory_pipeline.py`
27. `data_agent/test_data_consolidation.py`
28. `tests/test_memory_pipeline.py`
29. `tests/test_neo4j_memory_graph.py`
30. `tests/test_real_memory_integration.py`
31. `tests/test_feature_endpoints.py`

The first three files show the complete request path. The middle section explains
the agents and storage layers. The final test files provide executable examples
of the intended architecture.

## 3. High-level architecture

```text
                        FastAPI routes
                              |
             +----------------+----------------+
             |                |                |
       Interaction       Memory Tree      Alternate Timeline
         Service            Service             Service
             |                |                |
             +----------------+----------------+
                              |
                         MemoryEngine
                              |
              +---------------+---------------+
              |               |               |
          PostgreSQL        Qdrant           Neo4j
```

Routes translate HTTP input into validated Python models. Services orchestrate a
feature. Engines own domain behavior. Adapters own communication with external
databases. This separation prevents API files from becoming large collections of
database and LLM logic.

## 4. The main `/interactions` pipeline

`POST /interactions` is the normal chat endpoint. Its public response contract is
kept small even though substantial work happens internally.

```text
User message
    |
    v
Save user interaction
    |
    v
Load recent conversation and active goals
    |
    v
Optional conversational memory retrieval
    |
    v
ML classifier + Emotion Agent
    |
    v
Temporary candidate memories
    |
    v
Data Agent Stage 1 consolidation
    |
    v
Qdrant search for similar existing long-term memories
    |
    v
Data Agent Stage 2 CREATE / UPDATE / REJECT
    |
    v
PostgreSQL -> Qdrant -> Neo4j persistence
    |
    v
Buddy State update
    |
    v
Buddy Agent response
    |
    v
Save Buddy response
```

### Two retrieval operations that must remain separate

Conversational retrieval asks: "Which existing memories help understand and
answer this message?" Its results are supplied to the Emotion Agent and Buddy.

Consolidation retrieval asks: "Does this new consolidated observation represent
an existing long-term memory?" It occurs independently inside MemoryEngine after
Stage 1, and its results are supplied only to Data Agent Stage 2.

Never replace Buddy's conversational context with Stage 1 output. Never reuse
conversational search results as Stage 2 decision context.

## 5. Memory lifecycle

There are three identities in the memory pipeline:

```text
candidate_id        Temporary evidence from one interaction
consolidation_id    Temporary Stage 1 grouping identity
memory_id           Permanent PostgreSQL UUID
```

Only `memory_id` survives into PostgreSQL, Qdrant, and Neo4j. Candidate IDs are
retained as `evidence_ids` so every persistent fact can be traced to its source.

### CREATE

PostgreSQL creates a UUID first. Qdrant and Neo4j receive the same UUID.

### UPDATE

Stage 2 may update only a memory returned by the same user's Qdrant retrieval.
Old evidence IDs and new evidence IDs are combined before PostgreSQL and Qdrant
are updated.

### REJECT

Nothing is written. Rejected candidates do not receive permanent memory IDs.

### Relationships

Stage 1 relationships use consolidation IDs. MemoryEngine waits until Stage 2
and persistence complete, builds `consolidation_id -> memory_id`, and only then
creates final Memory-to-Memory relationships in Neo4j. A relationship is skipped
if either endpoint was rejected.

## 6. Endpoint reference

No endpoint uses an `/api` prefix. The root paths are mounted directly.

### `POST /interactions`

Normal Buddy conversation. This is the only endpoint that runs the full Emotion
Agent -> candidate -> Data Agent -> Buddy pipeline.

Request:

```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "650e8400-e29b-41d4-a716-446655440000",
  "text": "I am stressed about placements."
}
```

### `GET /memory-tree/{user_id}`

Returns the five stable UI branches: `happy`, `sad`, `angry`, `anxious`, and
`calm`. Internal emotions are mapped deterministically; Qwen is not called just
to count branches.

### `GET /memory-tree/{user_id}/emotions/{emotion}`

Returns ranked memories for one branch. Valid branch names are the five listed
above. This endpoint is read-only.

### `POST /memory-tree/{user_id}/reflect`

Reads memories for a branch and asks the configured Qwen client for a concise,
evidence-bound reflection. It does not call candidate extraction, Stage 1,
Stage 2, or persistence.

### `POST /alternate-timeline/tasks`

Creates a V1 planned task. Task management is not stored as conversation and
does not create a memory.

### `GET /alternate-timeline/tasks/{user_id}?date=YYYY-MM-DD`

Returns the user's tasks for the selected date, sorted by start time.

### `POST /alternate-timeline/tasks/{task_id}/decision`

Records `complete`, `skip`, or `cancel` plus an optional reason. A reason such as
"too tired" remains task-local and is not automatically promoted to memory.

### `POST /alternate-timeline/predict`

Uses MemoryEngine to retrieve same-user semantic and graph context, then asks
Qwen for cautious possible outcomes. The result is returned but not persisted
in V1. The task owner is checked before retrieval or reasoning.

## 7. File-by-file guide

### `main.py`

The application entry point. It creates the FastAPI application, configures CORS,
mounts routers, and opens/closes the global PostgreSQL connection during app
lifespan. Add a new router here only after its service and tests exist. Business
logic does not belong here.

### `api/interactions.py`

The thin HTTP adapter for `/interactions`. It validates request fields and calls
`InteractionService.process_interaction()`. It converts service exceptions into
HTTP errors. Changes to memory reasoning do not belong in this file.

### `services/interaction_service.py`

The top-level chat orchestrator. This is the best file for understanding the
runtime order. It coordinates persistence, context retrieval, Emotion Agent,
MemoryEngine, Buddy State, Buddy Agent, and response persistence. It does not
manually run Stage 1/2 or write long-term memories.

### `config/settings.py`

Central configuration model. It reads `.env` values for PostgreSQL, Qdrant,
Neo4j, OpenRouter, the dedicated Data Agent Qwen endpoint, model names, and app
metadata. Secrets should be configured here through environment variables and
must never be logged or hard-coded.

### `database/connection.py`

Repository layer for PostgreSQL with in-memory fallbacks for offline development.
It owns SQL for interactions, analyses, candidate legacy storage, Buddy state,
goals, scheduled tasks, and long-term-memory CRUD. The additive
`scheduled_tasks` table is created idempotently during database initialization.
PostgreSQL `long_term_memories.id` is the
canonical permanent memory ID.

### `ml/emotion_classifier.py`

Produces fast emotion probabilities from raw text. It can use a transformer or
the rule/lexicon fallback. Its result helps the Emotion Agent but is not itself a
long-term-memory decision.

### `emotion_agent/emotion_schema.py`

Defines validated `EmotionAnalysis`, individual emotion items, behavioral and
decision signals, and goal relevance. These models form the contract consumed by
Data Agent candidate extraction and Buddy State.

### `emotion_agent/emotion_engine.py`

Builds the contextual Emotion Agent prompt, calls the shared LLM client, parses
the result, and returns `EmotionAnalysis`. It receives conversational memories,
not Stage 1 consolidation results.

### `data_agent/data_schema.py`

Contains the memory pipeline contracts: `CandidateMemory`, `ConsolidatedMemory`,
Stage 1 relationships, Stage 2 decisions, persistence results, graph results,
and `MemoryPipelineResult`. Read this before changing DataAgent or MemoryEngine.

### `data_agent/data_engine.py`

Owns candidate extraction from structured Emotion Agent output and validates the
two Data Agent reasoning stages. Positive greetings are excluded; durable
negative emotions and meaningful behavioral/decision markers can become
candidates. Stage 1 and Stage 2 enforce user, evidence, and ID boundaries.

### `data_agent/data_prompt.py`

Builds explicit messages for Data Agent Stage 1 and Stage 2. Stage 1 sees only
candidates. Stage 2 sees one consolidated memory plus Qdrant-retrieved existing
memories. Prompt changes can alter persistence decisions and require dedicated
tests.

### `data_agent/data_llm.py`

Reasoning-layer wrapper around the dedicated Qwen client. It selects the prompt
builder and Pydantic output schema for Stage 1, Stage 2, and the retained legacy
interface. HTTP details do not belong here.

### `data_agent/data_llm_client.py`

Dedicated OpenAI-compatible client for the college Qwen model. It owns the Data
Agent API key/base URL/model, thinking controls, timeout, token usage, and response
extraction. Emotion and Buddy can continue using a different provider.

### `memory/memory_engine.py`

The long-term-memory orchestrator and shared retrieval boundary. It normalizes
candidate IDs/users, runs Stage 1, searches Qdrant independently for each group,
runs Stage 2, persists final actions, maps consolidation IDs to memory UUIDs, and
creates final Neo4j relationships. It also exposes conversational retrieval used
by InteractionService and prediction context used by Alternate Timeline.

### `vector_db/qdrant_client.py`

Qdrant adapter. It embeds text, upserts points, performs same-user search, filters
inactive memories, and supports deletion for tests. A Qdrant point ID must equal
the PostgreSQL long-term-memory UUID. Qdrant is an index, not the source of truth.

### `graph_db/neo4j_client.py`

Neo4j adapter. It creates/updates owned `Memory` nodes, links Student to Memory,
and creates allow-listed Memory-to-Memory relationships. Both endpoints must
belong to the same user. `MERGE` keeps nodes and relationships idempotent.

### `state/state_engine.py`

Purely deterministic Buddy State calculations. It updates bounded dimensions
such as concern, warmth, patience, and energy from `EmotionAnalysis`. It runs
after memory processing and does not call an LLM.

### `buddy/buddy_schema.py`

Defines the response contract returned to the frontend: text, expression,
intensity, and response type. Validators normalize unsupported model output.

### `buddy/buddy_engine.py`

Builds Buddy's conversational prompt from current text, emotion, state, recent
conversation, conversational memories, and goals. It returns a validated Buddy
response and has a robust deterministic fallback.

### `memory/memory_tree_schema.py`

Contracts for Memory Tree counts, memory summaries, reflection requests, and
reflection output. Keeping these models separate makes the route/service boundary
easy to inspect and gives FastAPI accurate OpenAPI documentation.

### `memory/memory_tree_service.py`

Read-only Memory Tree business logic. It loads active memories from PostgreSQL,
maps internal emotions into five UI categories, ranks branch memories, and calls
Qwen only for reflection. It contains no persistence calls.

### `api/memory_tree.py`

Thin `/memory-tree` router. It documents each route, delegates to
`MemoryTreeService`, and maps validation/service errors to HTTP responses.

### `alternate_timeline/timeline_schema.py`

Contracts and enums for tasks, decisions, lists, and predictions. It validates
that end time is after start time and constrains statuses/scenarios.

### `alternate_timeline/timeline_service.py`

Owns scheduled-task CRUD orchestration and read-only prediction orchestration.
Task records are stored through the database repository, while prediction
results remain transient in V1. It checks task ownership, retrieves context
through MemoryEngine, and asks Qwen for a bounded scenario. It does not call
InteractionService or the Data Agent memory pipeline.

### `api/alternate_timeline.py`

Thin `/alternate-timeline` router. It exposes task creation/listing/decision and
prediction without embedding Qdrant, Neo4j, or Qwen details in HTTP handlers.

## 8. Test files and what they prove

### `tests/test_interaction_memory_pipeline.py`

Proves `/interactions` sends Emotion-derived candidates into MemoryEngine, skips
empty candidates, preserves Buddy ordering, supports UPDATE/REJECT, and keeps
conversational retrieval separate.

### `data_agent/test_data_consolidation.py`

Proves Stage 1/2 contracts, evidence preservation, user isolation, CREATE,
UPDATE, REJECT, malformed output handling, and relationship validation.

### `tests/test_memory_pipeline.py`

Proves MemoryEngine persistence order, evidence accumulation, Qdrant identity,
consolidation-to-memory mapping, and relationship translation/skip behavior.

### `tests/test_neo4j_memory_graph.py`

Proves Neo4j idempotency, allow-listed relationship types, ownership enforcement,
cross-user rejection, and user-isolated reads.

### `tests/test_real_memory_integration.py`

Live integration against PostgreSQL, Qdrant, and Neo4j. It verifies the same UUID
in all stores and verifies a real `Memory A -[:TRIGGERS]-> Memory B` relationship.

### `tests/test_feature_endpoints.py`

Proves Memory Tree mapping/reflection, Alternate Timeline task flow/prediction,
owner isolation, and the absence of `/api`-prefixed duplicate routes.

## 9. Development rules

- Keep API routes thin.
- Put orchestration in services or MemoryEngine.
- Never write candidate memories directly as long-term memories.
- Never mix conversational and consolidation retrieval.
- Never use candidate IDs as Neo4j Memory node IDs.
- Never create cross-user graph relationships or prediction context.
- Treat PostgreSQL as source of truth and Qdrant as an index.
- Keep task decisions out of memory unless the user separately expresses a
  durable pattern through `/interactions`.
- Add Pydantic contracts and deterministic tests before changing an endpoint.
- Do not log API keys, complete prompts, or unnecessary private user content.

## 10. V1 limitations

- Scheduled tasks persist in PostgreSQL when it is connected. Offline mode uses
  the project's existing in-memory fallback and is cleared on process restart.
- Alternate Timeline predictions are returned but not saved.
- The current fallback embedding is useful for infrastructure testing but is not
  a production-grade semantic model.
- Reflection and prediction depend on the configured Data Agent Qwen endpoint.
- Authentication/authorization middleware is not yet shown here; services still
  enforce user IDs where they can, but production deployment should authenticate
  the caller rather than trusting request IDs alone.

## 11. Running the backend and tests

From the backend directory:

```powershell
python -m uvicorn main:app --reload
```

Open `http://localhost:8000/docs` to inspect every endpoint and schema.

Focused endpoint tests:

```powershell
python -m pytest tests/test_feature_endpoints.py -v
```

Core memory regressions:

```powershell
python -m pytest data_agent/test_data_consolidation.py tests/test_memory_pipeline.py tests/test_neo4j_memory_graph.py -v
```

Live databases:

```powershell
python -m pytest tests/test_real_memory_integration.py -v -s
```

The live test requires valid PostgreSQL, Qdrant, and Neo4j configuration and
cleans up its own synthetic records.
