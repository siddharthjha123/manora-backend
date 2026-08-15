# MANORA Backend - Technical Build Documentation

**MANORA** is a Digital Mental Health and Psychological Support System for Students in Higher Education. This document provides complete technical documentation of the actual backend implementation.

---

## 1. Project Architecture

The MANORA backend orchestrates a multi-stage conversational intelligence pipeline designed to separate emotional recognition, memory retrieval, internal state dynamics, memory extraction, and dialog generation.

```
                         STUDENT
                           │
                           ▼
                  POST /interactions
                           │
                           ▼
                Interaction Service
                           │
                           ▼
                 Memory Retrieval
                      Layer
                           │
                  Is context needed?
                    /             \
                  NO               YES
                  │                 │
                  │          ┌──────┴──────┐
                  │          ▼             ▼
                  │       Qdrant         Neo4j
                  │      semantic      relationships
                  │       memory          graph
                  │          │             │
                  └──────────┴──────┬──────┘
                                    ▼
                           Relevant Context
                                    │
                                    ▼
                           Emotion Agent
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                  Emotion ML              LLM Reasoning
                  Classifier                   │
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                           Emotion Analysis
                                    │
                                    ▼
                            Buddy State Engine (Deterministic)
                                    │
                                    ▼
                              Buddy State
                                    │
                                    ▼
                              Buddy Agent (LLM)
                                    │
                                    ▼
                           Buddy Response JSON
                                    │
                                    ▼
                                 STUDENT

Separately (Background Storage Flow):
Interaction + Emotion Analysis
            ↓
       Mock Data Agent
            ↓
      Candidate Memories
            ↓
       Memory Engine (Importance/Confidence Filter)
            ↓
       PostgreSQL / Supabase + Qdrant + Neo4j
```

---

## 2. Component Responsibilities

| Component | Responsibility | Core Question Answered |
|---|---|---|
| **ML Emotion Classifier** (`ml/emotion_classifier.py`) | Fast probability scoring for emotional signals. | *"What basic emotional signals are detected in this text?"* |
| **Emotion Agent** (`emotion_agent/emotion_engine.py`) | Deep contextual interpretation of student feelings, behavioral signals, decisions, and goals. | *"What is the student feeling and what behavioral patterns/conflicts are present?"* |
| **Mock Data Agent** (`data_agent/mock_data_agent.py`) | Extracts structured candidate memories from interaction and emotion signals. | *"What from this interaction is worth remembering?"* |
| **Memory Retrieval Layer** (`memory/memory_engine.py`) | Deterministic retrieval decision, semantic/graph query aggregation, and multi-DB persistence. | *"How should memories be stored, filtered, and retrieved?"* |
| **Buddy State Engine** (`state/state_engine.py`) | Deterministic mathematical calculation of Buddy's 7 internal emotional dimensions. | *"How should Buddy's internal emotional state shift?"* |
| **Buddy Agent** (`buddy/buddy_engine.py`) | Natural language generation with facial expression and stance. | *"Knowing the current context, emotions, memories, and Buddy state, what should Buddy say?"* |
| **Interaction Service** (`services/interaction_service.py`) | Main system orchestrator coordinating the entire 12-step lifecycle. | *"How do all backend components execute sequentially?"* |

---

## 3. Folder Structure

```
manora-backend/
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic configuration & environment variables
│
├── api/
│   ├── __init__.py
│   ├── interactions.py          # POST /interactions
│   ├── emotions.py              # POST /emotions/analyze
│   └── buddy.py                 # GET /buddy/state/{user_id} and /history
│
├── emotion_agent/
│   ├── __init__.py
│   ├── emotion_engine.py        # Emotion Agent execution engine
│   ├── emotion_parser.py        # Output sanitizer and Pydantic parser
│   ├── emotion_prompt.py        # Contextual prompt engineering
│   ├── emotion_schema.py        # Pydantic data contracts
│   └── emotion_tests.py         # Unit tests
│
├── buddy/
│   ├── __init__.py
│   ├── buddy_engine.py          # Buddy Agent execution engine
│   ├── buddy_prompt.py          # Buddy prompt engineering
│   └── buddy_schema.py          # BuddyResponse & Expression models
│
├── state/
│   ├── __init__.py
│   ├── state_engine.py          # Deterministic state engine & decay
│   └── state_rules.py           # Transition matrices and deltas
│
├── memory/
│   ├── __init__.py
│   └── memory_engine.py         # Retrieval heuristics & candidate persistence
│
├── ml/
│   ├── __init__.py
│   └── emotion_classifier.py    # Pluggable ML classifier (lexicon + transformers)
│
├── vector_db/
│   ├── __init__.py
│   └── qdrant_client.py         # Qdrant adapter with offline fallback
│
├── graph_db/
│   ├── __init__.py
│   └── neo4j_client.py          # Neo4j adapter with offline fallback
│
├── data_agent/
│   ├── __init__.py
│   ├── data_engine.py           # Data Agent scaffold delegating to mock
│   ├── data_parser.py           # Data parser
│   ├── data_prompt.py           # Data prompt scaffold
│   ├── data_schema.py           # Candidate memory schemas
│   ├── data_tests.py            # Data agent tests
│   └── mock_data_agent.py       # Decoupled mock Data Agent
│
├── database/
│   ├── __init__.py
│   ├── connection.py            # PostgreSQL connection pool & memory fallback
│   └── emotion_schema.sql       # PostgreSQL / Supabase DDL schema
│
├── services/
│   ├── __init__.py
│   └── interaction_service.py   # Complete conversational orchestrator
│
├── llm/
│   ├── __init__.py
│   └── base.py                  # OpenRouter/OpenAI LLM abstraction
│
├── tests/
│   ├── __init__.py
│   ├── test_emotion_agent.py    # Emotion classifier & agent tests
│   ├── test_memory_engine.py    # Retrieval decisions & vector/graph tests
│   ├── test_state_engine.py     # Deterministic state bounds & decay tests
│   ├── test_buddy_agent.py      # Buddy response & expression tests
│   └── test_interaction_flow.py # End-to-end integration tests
│
├── main.py                      # FastAPI application entrypoint
├── requirements.txt             # Project dependencies
├── .env.example                 # Environment configuration template
├── BUILD_DOCUMENTATION.md       # Comprehensive build manual
└── README.md                    # Project README
```

---

## 4. Python Version

- **Recommended**: Python `3.10`, `3.11`, or `3.12`.
- **Minimum Supported**: Python `3.9+`.

---

## 5. Environment Setup

Create your `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Key environment variables:
```ini
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/manora
OPENROUTER_API_KEY=your_openrouter_key
MODEL_NAME=openai/gpt-4o-mini
QDRANT_ENABLED=false
NEO4J_ENABLED=false
```

---

## 6. Virtual Environment Setup

Create and activate a virtual environment:

```bash
# Using standard venv
python3 -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate
```

---

## 7. Dependency Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

---

## 8. Database Setup

The backend utilizes PostgreSQL / Supabase for canonical data persistence, with an automatic in-memory fallback during local development when `DATABASE_URL` is omitted.

To apply schema to a PostgreSQL instance:
```bash
psql -U postgres -d manora -f database/emotion_schema.sql
```

---

## 9. Supabase / PostgreSQL Setup

In Supabase:
1. Navigate to the **SQL Editor**.
2. Paste the contents of `database/emotion_schema.sql` and run.
3. Obtain your Connection String from **Project Settings > Database > Connection URI**.
4. Set in `.env`:
   ```ini
   DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
   ```

---

## 10. Qdrant Setup

Qdrant stores semantic vector embeddings for meaningful candidate memories.

To run Qdrant via Docker:
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

Enable in `.env`:
```ini
QDRANT_ENABLED=true
QDRANT_URL=http://localhost:6333
```

*(If `QDRANT_ENABLED=false`, the backend uses its built-in in-memory vector store seamlessly).*

---

## 11. Neo4j Setup

Neo4j represents complex graph relationships between Students, Goals, Memories, Emotions, Behaviors, and Decisions.

To run Neo4j via Docker:
```bash
docker run -d -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

Enable in `.env`:
```ini
NEO4J_ENABLED=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

*(If `NEO4J_ENABLED=false`, the backend operates in offline graph fallback mode).*

---

## 12. ML Model Setup

The ML Emotion Classifier (`ml/emotion_classifier.py`) uses an abstract interface:
- **Default**: Fast, zero-dependency student-domain rule and lexicon classifier.
- **Transformers Mode**: If `torch` and `transformers` are installed, it automatically loads HuggingFace models such as `distilbert-base-uncased-emotion` or `j-hartmann/emotion-english-distilroberta-base`.

To install transformer support:
```bash
pip install torch transformers
```

---

## 13. OpenRouter Setup

1. Sign up at [OpenRouter](https://openrouter.ai/).
2. Create an API key.
3. Configure `.env`:
   ```ini
   OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   MODEL_NAME=openai/gpt-4o-mini
   ```
*(If no API key is provided, the client defaults to deterministic mock generation for offline development and tests).*

---

## 14. How to Start FastAPI

Start the server using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 15. Swagger URL

Interactive OpenAPI documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 16. Every Endpoint

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | System overview & status |
| `GET` | `/health` | Health check and connected service status |
| `POST` | `/interactions` | Main student interaction processing pipeline |
| `POST` | `/emotions/analyze` | Development ad-hoc emotion analysis |
| `GET` | `/buddy/state/{user_id}` | Fetch current Buddy internal emotional state |
| `GET` | `/buddy/state/{user_id}/history` | Fetch audit log of Buddy state transitions |

---

## 17. Request Examples

### `POST /interactions`
```bash
curl -X POST http://localhost:8000/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "session_id": "1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed",
    "text": "I planned to study at 10 but I ended up watching a series for two hours. I know I should study but this keeps happening."
  }'
```

### `POST /emotions/analyze`
```bash
curl -X POST http://localhost:8000/emotions/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I feel so overwhelmed by upcoming placement interviews."
  }'
```

### `GET /buddy/state/{user_id}`
```bash
curl http://localhost:8000/buddy/state/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d
```

---

## 18. Response Examples

### `POST /interactions` Response
```json
{
  "interaction_id": "4a08cfcf-346a-4933-9118-8f8bf228be3f",
  "emotion": {
    "interaction_id": "4a08cfcf-346a-4933-9118-8f8bf228be3f",
    "primary_emotion": "frustration",
    "emotions": [
      {
        "emotion": "frustration",
        "intensity": 0.84,
        "confidence": 0.91,
        "source": "model_inferred"
      },
      {
        "emotion": "guilt",
        "intensity": 0.72,
        "confidence": 0.79,
        "source": "model_inferred"
      }
    ],
    "emotional_summary": "The student appears frustrated and guilty about repeatedly avoiding planned study sessions in favor of entertainment.",
    "behavioral_signals": [
      "avoided planned study session",
      "continued watching entertainment despite recognizing the conflict"
    ],
    "decision_signals": [
      "chose entertainment instead of planned study activity"
    ],
    "goal_relevance": {
      "related": true,
      "goal": "academic progress"
    }
  },
  "buddy_state": {
    "happiness": 0.48,
    "sadness": 0.10,
    "frustration": 0.32,
    "concern": 0.65,
    "warmth": 0.80,
    "patience": 0.68,
    "energy": 0.70
  },
  "buddy": {
    "text": "You're repeating the same pattern again. Do you actually want to achieve this goal?",
    "expression": "concerned",
    "intensity": 0.72,
    "response_type": "challenge"
  }
}
```

---

## 19. How Memory Retrieval Works

The Memory Engine evaluates whether context is required prior to executing any database searches using `memory_engine.should_retrieve(text)`:
1. **Length filter**: Messages under 4 words without high-signal terms skip retrieval.
2. **Recurrence triggers**: Matches phrases such as `"again"`, `"keeps happening"`, `"every time"`, `"last month"`, `"pattern"`.
3. **Goal markers**: Matches terms such as `"placement"`, `"study"`, `"exam"`, `"grade"`, `"interview"`, `"give up"`.
4. **Behavioral markers**: Matches terms such as `"planned to"`, `"instead"`, `"netflix"`, `"procrastinating"`, `"delayed"`.

---

## 20. How Qdrant is Used

Only meaningful memories (with `importance >= 0.60` and `confidence >= 0.60`) are embedded and stored in Qdrant's `student_memories` collection. Chat history is not blindly vectorized. Similarity queries use cosine distance.

---

## 21. How Neo4j is Used

Neo4j models explicit relationships:
- `(:Student)-[:HAS_MEMORY]->(:Memory)`
- `(:Memory)-[:REFLECTS_EMOTION]->(:Emotion)`
- `(:Memory)-[:SHOWED_BEHAVIOR]->(:Behavior)`
- `(:Memory)-[:INVOLVED_DECISION]->(:Decision)`
- `(:Student)-[:HAS_GOAL]->(:Goal)`

---

## 22. How Emotion Agent Works

The Emotion Agent (`emotion_agent/emotion_engine.py`):
1. Runs the fast ML classifier to get probability priors.
2. Formats prompt with ML signals, raw message, recent dialogue, retrieved memories, and active goals.
3. Instructs LLM to identify behavioral signals and decision signals while explicitly prohibiting clinical psychiatric diagnoses.
4. Parses output into validated `EmotionAnalysis`.

---

## 23. How Buddy State Works

Buddy's state is **purely deterministic** (no LLM calculates state numbers):
- **7 Dimensions**: `happiness`, `sadness`, `frustration`, `concern`, `warmth`, `patience`, `energy` in `[0.0, 1.0]`.
- **Transitions**: Defined in `state/state_rules.py`. Study avoidance or recurring guilt increases `concern` and `frustration`, and decreases `patience`. Positive momentum increases `happiness` and `energy`.
- **Temporal Decay**: Exponential decay drifts state back to baseline over elapsed hours.

---

## 24. How Buddy Agent Works

The Buddy Agent (`buddy/buddy_engine.py`):
1. Ingests student text, Emotion Analysis, Buddy's current state, and memories.
2. Synthesizes a concise (1-3 sentence), natural response reflecting its internal state.
3. Produces a structured `BuddyResponse` including `expression` (`neutral`, `happy`, `sad`, `concerned`, `frustrated`, `angry`, `encouraging`, `thoughtful`), `intensity`, and `response_type`.
4. Never modifies Buddy State directly.

---

## 25. What is Currently Mocked

- **Data Agent**: The Data Agent extraction is currently simulated by `data_agent/mock_data_agent.py`.
- **Qdrant & Neo4j**: Default to graceful offline in-memory fallback when services are disabled.
- **PostgreSQL**: Defaults to in-memory persistence when `DATABASE_URL` is omitted.

---

## 26. How the Real Data Agent Should Replace the Mock

The future Data Agent developer should:
1. Implement the extraction logic in `data_agent/data_engine.py` using `data_agent/data_prompt.py` and `data_agent/data_parser.py`.
2. Implement the `BaseDataAgent` interface:
   ```python
   class LLMDataAgent(BaseDataAgent):
       def process(self, interaction: Dict[str, Any], emotion: EmotionAnalysis) -> List[CandidateMemory]:
           ...
   ```
3. Update `services/interaction_service.py` to inject the new instance without modifying Emotion Agent, State Engine, Memory Engine, or Buddy Agent.

---

## 27. How to Run Tests

Run the complete test suite with `pytest`:

```bash
pytest -v
```

To run individual test modules:
```bash
pytest tests/test_emotion_agent.py -v
pytest tests/test_state_engine.py -v
pytest tests/test_memory_engine.py -v
pytest tests/test_buddy_agent.py -v
pytest tests/test_interaction_flow.py -v
```

---

## 28. Troubleshooting

| Issue | Resolution |
|---|---|
| `pip: command not found` | Ensure your virtual environment is active (`source venv/bin/activate`). |
| `Qdrant connection failed` | Keep `QDRANT_ENABLED=false` for local offline mode, or verify docker container on port `6333`. |
| `Neo4j connection refused` | Keep `NEO4J_ENABLED=false` for local offline mode, or start Neo4j on port `7687`. |
| `OpenRouter 401 Unauthorized` | Verify `OPENROUTER_API_KEY` in `.env`. The system uses mock fallback if left empty. |

---

## 29. Current Limitations

1. V1 uses synchronous/deterministic embedding fallback when full HuggingFace sentence transformer models are not preloaded.
2. The real Data Agent is currently mocked.
3. Voice synthesis and audio recognition are not included in V1 scope.

---

## 30. Future STT / TTS Integration Points

1. **Speech-to-Text (STT)**: Future audio upload route `POST /interactions/audio` will transcribe student speech and pass text directly to `interaction_service.process_interaction(...)`.
2. **Text-to-Speech (TTS)**: Future frontend or audio service will consume `buddy.text` and `buddy.expression` + `buddy.intensity` to synthesize expressive prosody and drive avatar facial blendshapes.
