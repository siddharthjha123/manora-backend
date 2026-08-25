# MANORA Backend

**MANORA** is a Digital Mental Health and Psychological Support System for Students in Higher Education.

It features an AI **Buddy** with internal emotional states, multi-signal emotion detection, semantic memory retrieval via Qdrant, relationship tracking via Neo4j, and canonical storage in PostgreSQL / Supabase.

---

## 🌟 Key Features

- **Decoupled Architecture**: Clean separation between Emotion Agent, Data Agent, Memory Engine, Buddy State Engine, and Buddy Agent.
- **Deterministic State Engine**: Buddy maintains 7 internal emotional dimensions (`happiness`, `sadness`, `frustration`, `concern`, `warmth`, `patience`, `energy`) calculated mathematically without LLM hallucination.
- **Smart Memory Retrieval**: Deterministic heuristic triggers avoid querying vector/graph databases on trivial chat.
- **Pluggable ML Classification**: Fast lexicon-based and HuggingFace transformer emotion detection.
- **Mock Data Agent**: Cleanly decoupled interface allowing another developer to drop in the real Data Agent without modifying surrounding layers.
- **Zero-Friction Local Development**: Graceful in-memory fallbacks when PostgreSQL, Qdrant, Neo4j, or OpenRouter are not configured.

---

## 🚀 Quick Start

### 1. Clone & Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
```

### 3. Run FastAPI Backend
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation is available at **http://localhost:8000/docs**.

---

## 🧪 Running Tests

```bash
pytest -v
```

---

## 📖 Complete Documentation

For full architectural breakdown, database schemas, deployment steps, and endpoint references, see [BUILD_DOCUMENTATION.md](BUILD_DOCUMENTATION.md).