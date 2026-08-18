"""
MANORA Database Connection and Repository Layer.
Provides database access for PostgreSQL/Supabase with an in-memory fallback
for isolated testing and offline development.
"""

import datetime
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger("manora.database")

# In-memory persistence store for testing and when PostgreSQL is offline
_in_memory_db: Dict[str, Dict[str, Any]] = {
    "users": {},
    "sessions": {},
    "goals": {},
    "interactions": {},
    "interaction_analyses": {},
    "candidate_memories": {},
    "buddy_states": {},
    "buddy_state_history": {},
}


class DatabaseManager:
    """Manages database connections and repository operations."""

    def __init__(self):
        self.settings = get_settings()
        self.is_postgres_connected = False
        self._pool = None

    async def initialize(self):
        """Initializes PostgreSQL connection pool if configured."""
        if self.settings.DATABASE_URL:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(
                    dsn=self.settings.DATABASE_URL,
                    min_size=1,
                    max_size=10,
                )
                self.is_postgres_connected = True
                logger.info("Connected to PostgreSQL successfully.")
            except Exception as e:
                logger.warning(f"Failed to connect to PostgreSQL ({e}). Using in-memory fallback.")
                self.is_postgres_connected = False
        else:
            logger.info("DATABASE_URL not set. Running with in-memory persistence.")

    async def close(self):
        """Closes connection pool."""
        if self._pool:
            await self._pool.close()
            self.is_postgres_connected = False

    # ------------------------------------------------------------
    # Interactions Repository
    # ------------------------------------------------------------
    async def save_interaction(
        self,
        user_id: str,
        session_id: str,
        role: str,
        raw_text: str,
        interaction_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Saves a single conversation interaction."""
        interaction_id = str(interaction_id or uuid.uuid4())
        user_id = str(user_id)
        session_id = str(session_id)
        now = datetime.datetime.now(datetime.timezone.utc)

        record = {
            "id": interaction_id,
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "raw_text": raw_text,
            "created_at": now,
        }

        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO interactions (id, user_id, session_id, role, raw_text, created_at)
                        VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6::timestamptz)
                        """,
                        uuid.UUID(interaction_id),
                        uuid.UUID(user_id),
                        uuid.UUID(session_id),
                        role,
                        raw_text,
                        now,
                    )
                return record
            except Exception as e:
                logger.error(f"PostgreSQL save_interaction error: {e}. Falling back to in-memory.")

        _in_memory_db["interactions"][interaction_id] = record
        return record

    async def get_recent_interactions(
        self,
        user_id: str,
        session_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieves recent conversation messages for a session."""
        user_id = str(user_id)
        session_id = str(session_id)

        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT id, user_id, session_id, role, raw_text, created_at
                        FROM interactions
                        WHERE user_id = $1::uuid AND session_id = $2::uuid
                        ORDER BY created_at DESC
                        LIMIT $3
                        """,
                        uuid.UUID(user_id),
                        uuid.UUID(session_id),
                        limit,
                    )
                    return [dict(r) for r in reversed(rows)]
            except Exception as e:
                logger.error(f"PostgreSQL get_recent_interactions error: {e}. Falling back to in-memory.")

        matching = [
            i for i in _in_memory_db["interactions"].values()
            if i["user_id"] == user_id and i["session_id"] == session_id
        ]
        matching.sort(key=lambda x: x["created_at"])
        return matching[-limit:]

    # ------------------------------------------------------------
    # Interaction Analyses Repository
    # ------------------------------------------------------------
    async def save_interaction_analysis(
        self,
        interaction_id: str,
        analysis_dict: Dict[str, Any],
        version: int = 1,
    ) -> Dict[str, Any]:
        """Saves structured Emotion Agent analysis JSON."""
        analysis_id = str(uuid.uuid4())
        interaction_id = str(interaction_id)
        now = datetime.datetime.now(datetime.timezone.utc)

        record = {
            "id": analysis_id,
            "interaction_id": interaction_id,
            "analysis_version": version,
            "analysis_json": analysis_dict,
            "created_at": now,
        }

        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO interaction_analyses (id, interaction_id, analysis_version, analysis_json, created_at)
                        VALUES ($1::uuid, $2::uuid, $3, $4::jsonb, $5::timestamptz)
                        ON CONFLICT (interaction_id, analysis_version)
                        DO UPDATE SET analysis_json = EXCLUDED.analysis_json
                        """,
                        uuid.UUID(analysis_id),
                        uuid.UUID(interaction_id),
                        version,
                        json.dumps(analysis_dict),
                        now,
                    )
                return record
            except Exception as e:
                logger.error(f"PostgreSQL save_interaction_analysis error: {e}. Falling back to in-memory.")

        _in_memory_db["interaction_analyses"][interaction_id] = record
        return record

    async def get_interaction_analysis(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves structured analysis for an interaction."""
        interaction_id = str(interaction_id)
        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT id, interaction_id, analysis_version, analysis_json, created_at
                        FROM interaction_analyses
                        WHERE interaction_id = $1::uuid
                        ORDER BY analysis_version DESC
                        LIMIT 1
                        """,
                        uuid.UUID(interaction_id),
                    )
                    if row:
                        return dict(row)
            except Exception as e:
                logger.error(f"PostgreSQL get_interaction_analysis error: {e}.")

        return _in_memory_db["interaction_analyses"].get(interaction_id)

    # ------------------------------------------------------------
    # Candidate Memories Repository
    # ------------------------------------------------------------
    async def save_candidate_memories(
        self,
        interaction_id: str,
        user_id: str,
        candidate_memories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Saves candidate memories extracted from interaction."""
        saved = []
        now = datetime.datetime.now(datetime.timezone.utc)
        user_id = str(user_id)
        interaction_id = str(interaction_id)

        for mem in candidate_memories:
            mem_id = str(mem.get("id") or uuid.uuid4())
            record = {
                "id": mem_id,
                "interaction_id": interaction_id,
                "user_id": user_id,
                "content": mem.get("content", ""),
                "context_json": mem.get("context", {}),
                "emotional_state_json": mem.get("emotional_state", []),
                "importance": float(mem.get("importance", 0.5)),
                "confidence": float(mem.get("confidence", 0.5)),
                "status": mem.get("status", "pending"),
                "created_at": now,
            }

            if self.is_postgres_connected and self._pool:
                try:
                    async with self._pool.acquire() as conn:
                        await conn.execute(
                            """
                            INSERT INTO candidate_memories (
                                id, interaction_id, user_id, content,
                                context_json, emotional_state_json,
                                importance, confidence, status, created_at
                            )
                            VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5::jsonb, $6::jsonb, $7, $8, $9, $10::timestamptz)
                            """,
                            uuid.UUID(mem_id),
                            uuid.UUID(interaction_id),
                            uuid.UUID(user_id),
                            record["content"],
                            json.dumps(record["context_json"]),
                            json.dumps(record["emotional_state_json"]),
                            record["importance"],
                            record["confidence"],
                            record["status"],
                            now,
                        )
                except Exception as e:
                    logger.error(f"PostgreSQL save_candidate_memories error: {e}.")

            _in_memory_db["candidate_memories"][mem_id] = record
            saved.append(record)

        return saved

    # ------------------------------------------------------------
    # Buddy State Repository
    # ------------------------------------------------------------
    async def get_buddy_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves latest Buddy emotional state for a student."""
        user_id = str(user_id)
        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT id, user_id, happiness, sadness, frustration, concern, warmth, patience, energy, updated_at
                        FROM buddy_states
                        WHERE user_id = $1::uuid
                        """,
                        uuid.UUID(user_id),
                    )
                    if row:
                        return dict(row)
            except Exception as e:
                logger.error(f"PostgreSQL get_buddy_state error: {e}.")

        return _in_memory_db["buddy_states"].get(user_id)

    async def upsert_buddy_state(self, user_id: str, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Inserts or updates Buddy emotional state for a student."""
        user_id = str(user_id)
        state_id = str(state_dict.get("id") or uuid.uuid4())
        now = datetime.datetime.now(datetime.timezone.utc)

        record = {
            "id": state_id,
            "user_id": user_id,
            "happiness": float(state_dict.get("happiness", 0.6)),
            "sadness": float(state_dict.get("sadness", 0.1)),
            "frustration": float(state_dict.get("frustration", 0.1)),
            "concern": float(state_dict.get("concern", 0.2)),
            "warmth": float(state_dict.get("warmth", 0.8)),
            "patience": float(state_dict.get("patience", 0.8)),
            "energy": float(state_dict.get("energy", 0.7)),
            "updated_at": now,
        }

        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO buddy_states (
                            id, user_id, happiness, sadness, frustration, concern, warmth, patience, energy, updated_at
                        )
                        VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10::timestamptz)
                        ON CONFLICT (user_id)
                        DO UPDATE SET
                            happiness = EXCLUDED.happiness,
                            sadness = EXCLUDED.sadness,
                            frustration = EXCLUDED.frustration,
                            concern = EXCLUDED.concern,
                            warmth = EXCLUDED.warmth,
                            patience = EXCLUDED.patience,
                            energy = EXCLUDED.energy,
                            updated_at = EXCLUDED.updated_at
                        """,
                        uuid.UUID(state_id),
                        uuid.UUID(user_id),
                        record["happiness"],
                        record["sadness"],
                        record["frustration"],
                        record["concern"],
                        record["warmth"],
                        record["patience"],
                        record["energy"],
                        now,
                    )
                return record
            except Exception as e:
                logger.error(f"PostgreSQL upsert_buddy_state error: {e}.")

        _in_memory_db["buddy_states"][user_id] = record
        return record

    async def save_buddy_state_history(
        self,
        user_id: str,
        previous_state: Dict[str, Any],
        new_state: Dict[str, Any],
        trigger_interaction_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Saves audit record of Buddy state transition."""
        history_id = str(uuid.uuid4())
        user_id = str(user_id)
        now = datetime.datetime.now(datetime.timezone.utc)

        record = {
            "id": history_id,
            "user_id": user_id,
            "previous_state": previous_state,
            "new_state": new_state,
            "trigger_interaction_id": trigger_interaction_id,
            "created_at": now,
        }

        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO buddy_state_history (
                            id, user_id, previous_state, new_state, trigger_interaction_id, created_at
                        )
                        VALUES ($1::uuid, $2::uuid, $3::jsonb, $4::jsonb, $5::uuid, $6::timestamptz)
                        """,
                        uuid.UUID(history_id),
                        uuid.UUID(user_id),
                        json.dumps(previous_state),
                        json.dumps(new_state),
                        uuid.UUID(trigger_interaction_id) if trigger_interaction_id else None,
                        now,
                    )
                return record
            except Exception as e:
                logger.error(f"PostgreSQL save_buddy_state_history error: {e}.")

        if user_id not in _in_memory_db["buddy_state_history"]:
            _in_memory_db["buddy_state_history"][user_id] = []
        _in_memory_db["buddy_state_history"][user_id].append(record)
        return record

    async def get_buddy_state_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves history of Buddy state transitions."""
        user_id = str(user_id)
        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT id, user_id, previous_state, new_state, trigger_interaction_id, created_at
                        FROM buddy_state_history
                        WHERE user_id = $1::uuid
                        ORDER BY created_at DESC
                        LIMIT $2
                        """,
                        uuid.UUID(user_id),
                        limit,
                    )
                    return [dict(r) for r in rows]
            except Exception as e:
                logger.error(f"PostgreSQL get_buddy_state_history error: {e}.")

        hist = _in_memory_db["buddy_state_history"].get(user_id, [])
        return sorted(hist, key=lambda x: x["created_at"], reverse=True)[:limit]

    # ------------------------------------------------------------
    # Goals Repository
    # ------------------------------------------------------------
    async def get_user_goals(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieves active goals for a student."""
        user_id = str(user_id)
        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT id, user_id, title, description, status, created_at
                        FROM goals
                        WHERE user_id = $1::uuid AND status = 'active'
                        ORDER BY created_at DESC
                        """,
                        uuid.UUID(user_id),
                    )
                    return [dict(r) for r in rows]
            except Exception as e:
                logger.error(f"PostgreSQL get_user_goals error: {e}.")

        return [
            g for g in _in_memory_db["goals"].values()
            if g.get("user_id") == user_id and g.get("status") == "active"
        ]

    async def create_user_goal(
        self,
        user_id: str,
        title: str,
        description: Optional[str] = None,
        status: str = "active",
    ) -> Dict[str, Any]:
        """Creates a goal for a student."""
        goal_id = str(uuid.uuid4())
        user_id = str(user_id)
        now = datetime.datetime.now(datetime.timezone.utc)
        record = {
            "id": goal_id,
            "user_id": user_id,
            "title": title,
            "description": description or "",
            "status": status,
            "created_at": now,
            "updated_at": now,
        }
        _in_memory_db["goals"][goal_id] = record
        return record

    # ---------------------------------------------------------------------
    # CRUD OPERATIONS AFTER DATA AGENT RESULT
    # ---------------------------------------------------------------------

    async def create_long_term_memory(
        self,
        user_id: str,
        content: str,
        importance: float,
        confidence: float,
        emotions: List[Dict[str, Any]],
        evidence_ids: List[str],
    ) -> Dict[str, Any]:
        """Creates a new consolidated long-term memory."""

        memory_id = str(uuid.uuid4())
        user_id = str(user_id)
        now = datetime.datetime.now(datetime.timezone.utc)

        record = {
            "id": memory_id,
            "user_id": user_id,
            "content": content,
            "importance": float(importance),
            "confidence": float(confidence),
            "emotions": emotions,
            "evidence_ids": evidence_ids,
            "created_at": now,
            "updated_at": now,
            "is_active": True,
        }

        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO long_term_memories (
                            id,
                            user_id,
                            content,
                            importance,
                            confidence,
                            emotions,
                            evidence_ids,
                            created_at,
                            updated_at,
                            is_active
                        )
                        VALUES (
                            $1::uuid,
                            $2::uuid,
                            $3,
                            $4,
                            $5,
                            $6::jsonb,
                            $7::jsonb,
                            $8::timestamptz,
                            $9::timestamptz,
                            $10
                        )
                        RETURNING
                            id,
                            user_id,
                            content,
                            importance,
                            confidence,
                            emotions,
                            evidence_ids,
                            created_at,
                            updated_at,
                            is_active
                        """,
                        uuid.UUID(memory_id),
                        uuid.UUID(user_id),
                        content,
                        float(importance),
                        float(confidence),
                        json.dumps(emotions),
                        json.dumps(evidence_ids),
                        now,
                        now,
                        True,
                    )

                    if row:
                        return dict(row)

            except Exception as e:
                logger.error(
                    f"PostgreSQL create_long_term_memory error: {e}. "
                    "Falling back to in-memory."
                )

        _in_memory_db.setdefault("long_term_memories", {})
        _in_memory_db["long_term_memories"][memory_id] = record

        return record

    async def get_long_term_memory(
        self,
        memory_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves one active long-term memory for a student."""

        memory_id = str(memory_id)
        user_id = str(user_id)

        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT
                            id,
                            user_id,
                            content,
                            importance,
                            confidence,
                            emotions,
                            evidence_ids,
                            created_at,
                            updated_at,
                            is_active
                        FROM long_term_memories
                        WHERE id = $1::uuid
                          AND user_id = $2::uuid
                          AND is_active = TRUE
                        """,
                        uuid.UUID(memory_id),
                        uuid.UUID(user_id),
                    )

                    if row:
                        return dict(row)

            except Exception as e:
                logger.error(
                    f"PostgreSQL get_long_term_memory error: {e}. "
                    "Falling back to in-memory."
                )

        memory = _in_memory_db.get(
            "long_term_memories",
            {}
        ).get(memory_id)

        if memory and memory["user_id"] == user_id and memory["is_active"]:
            return memory

        return None

    async def get_long_term_memories(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieves active long-term memories for a student."""

        user_id = str(user_id)

        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT
                            id,
                            user_id,
                            content,
                            importance,
                            confidence,
                            emotions,
                            evidence_ids,
                            created_at,
                            updated_at,
                            is_active
                        FROM long_term_memories
                        WHERE user_id = $1::uuid
                          AND is_active = TRUE
                        ORDER BY updated_at DESC
                        LIMIT $2
                        """,
                        uuid.UUID(user_id),
                        limit,
                    )

                    return [dict(row) for row in rows]

            except Exception as e:
                logger.error(
                    f"PostgreSQL get_long_term_memories error: {e}. "
                    "Falling back to in-memory."
                )

        memories = [
            memory
            for memory in _in_memory_db.get(
                "long_term_memories",
                {}
            ).values()
            if memory["user_id"] == user_id
            and memory["is_active"]
        ]

        memories.sort(
            key=lambda memory: memory["updated_at"],
            reverse=True,
        )

        return memories[:limit]

    async def update_long_term_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        importance: float,
        confidence: float,
        emotions: List[Dict[str, Any]],
        evidence_ids: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Updates an existing consolidated long-term memory."""

        memory_id = str(memory_id)
        user_id = str(user_id)
        now = datetime.datetime.now(datetime.timezone.utc)

        if self.is_postgres_connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        UPDATE long_term_memories
                        SET
                            content = $1,
                            importance = $2,
                            confidence = $3,
                            emotions = $4::jsonb,
                            evidence_ids = $5::jsonb,
                            updated_at = $6::timestamptz
                        WHERE id = $7::uuid
                          AND user_id = $8::uuid
                          AND is_active = TRUE
                        RETURNING
                            id,
                            user_id,
                            content,
                            importance,
                            confidence,
                            emotions,
                            evidence_ids,
                            created_at,
                            updated_at,
                            is_active
                        """,
                        content,
                        float(importance),
                        float(confidence),
                        json.dumps(emotions),
                        json.dumps(evidence_ids),
                        now,
                        uuid.UUID(memory_id),
                        uuid.UUID(user_id),
                    )

                    if row:
                        return dict(row)

                    return None

            except Exception as e:
                logger.error(
                    f"PostgreSQL update_long_term_memory error: {e}. "
                    "Falling back to in-memory."
                )

        memories = _in_memory_db.setdefault(
            "long_term_memories",
            {},
        )

        memory = memories.get(memory_id)

        if not memory:
            return None

        if memory["user_id"] != user_id:
            return None

        if not memory["is_active"]:
            return None

        memory["content"] = content
        memory["importance"] = float(importance)
        memory["confidence"] = float(confidence)
        memory["emotions"] = emotions
        memory["evidence_ids"] = evidence_ids
        memory["updated_at"] = now

        return memory


# Global singleton instance
db = DatabaseManager()
