"""
MANORA Qdrant Vector Database Adapter.
Handles embedding generation, vector upserting, and semantic memory search.
Supports graceful offline operation when Qdrant is disabled.
"""

import hashlib
import logging
import math
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger("manora.vector_db")

from vector_db.embedding_client import embedding_client

class QdrantAdapter:
    """Adapter for Qdrant Vector Database semantic memory store."""

    COLLECTION_NAME = "long_term_memories"
    VECTOR_DIM = 1024

    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.QDRANT_ENABLED
        self.client = None
        self._in_memory_vectors: Dict[str, Dict[str, Any]] = {}

        if self.enabled:
            self._connect()

    def _connect(self):
        """Initializes connection to Qdrant server."""
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(
                url=self.settings.QDRANT_URL,
                api_key=self.settings.QDRANT_API_KEY,
            )
            self._ensure_collection()
            logger.info("Connected to Qdrant vector database successfully.")
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant ({e}). Falling back to internal memory vector store.")
            self.client = None

    def _ensure_collection(self):
        """Ensures the memory vector collection exists."""
        if not self.client:
            return
        try:
            from qdrant_client.http.models import (
                Distance,
                PayloadSchemaType,
                VectorParams,
            )
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]
            if self.COLLECTION_NAME not in names:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(size=self.VECTOR_DIM, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection '{self.COLLECTION_NAME}'.")

            collection_info = self.client.get_collection(self.COLLECTION_NAME)
            payload_schema = collection_info.payload_schema or {}
            required_indexes = {
                "user_id": PayloadSchemaType.KEYWORD,
                "is_active": PayloadSchemaType.BOOL,
            }

            for field_name, field_schema in required_indexes.items():
                if field_name not in payload_schema:
                    self.client.create_payload_index(
                        collection_name=self.COLLECTION_NAME,
                        field_name=field_name,
                        field_schema=field_schema,
                        wait=True,
                    )
                    logger.info(
                        "Created Qdrant payload index '%s' on collection '%s'.",
                        field_name,
                        self.COLLECTION_NAME,
                    )
        except Exception as e:
            logger.warning(f"Error ensuring Qdrant collection ({e}).")

    def _get_embedding(self, text: str) -> List[float]:
        """Generates embedding vector for memory text."""
        return embedding_client.embed_text(text)


    # ---------------------------------------------------------
    # Method which will store a single long term memory withe metadata 
    # ---------------------------------------------------------
    def upsert_memory(
        self,
        memory_id: str,
        text: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Embeds and stores one long-term memory in Qdrant.

        The memory_id should match the ID of the corresponding
        long-term memory stored in PostgreSQL.
        """

        memory_id = str(memory_id)
        user_id = str(user_id)

        vector = self._get_embedding(text)

        payload = {
            "memory_id": memory_id,
            "user_id": user_id,
            "text": text,
            **(metadata or {}),
        }

        # ---------------------------------------------------------
        # Live Qdrant
        # ---------------------------------------------------------
        if self.client:
            try:
                from qdrant_client.http.models import PointStruct

                self.client.upsert(
                    collection_name=self.COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id=memory_id,
                            vector=vector,
                            payload=payload,
                        )
                    ],
                )

                logger.info(
                    f"Upserted long-term memory {memory_id} to Qdrant."
                )

                return True

            except Exception as e:
                logger.error(
                    f"Qdrant long-term memory upsert error: {e}. "
                    "Falling back to in-memory vector storage."
                )

        # ---------------------------------------------------------
        # In-memory fallback
        # ---------------------------------------------------------
        self._in_memory_vectors[memory_id] = {
            "id": memory_id,
            "vector": vector,
            "payload": payload,
            "user_id": user_id,
        }

        return True

    # ------------------------------------------------------
    # Method which will search for long term memories
    # reminder: to delete the existing search_memories() method
    # ------------------------------------------------------

    def search_long_term_memories(
        self,
        user_id: str,
        query_text: str,
        limit: int = 5,
        score_threshold: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Searches for long-term memories that are semantically
        similar to the supplied query text.

        Only memories belonging to the specified user are returned.
        """

        user_id = str(user_id)
        query_vector = self._get_embedding(query_text)

        # ---------------------------------------------------------
        # Live Qdrant
        # ---------------------------------------------------------
        if self.client:
            try:
                from qdrant_client.http.models import (
                    FieldCondition,
                    Filter,
                    MatchValue,
                )

                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id),
                        ),
                        FieldCondition(
                            key="is_active",
                            match=MatchValue(value=True),
                        ),
                    ]
                )

                results = self.client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                ).points

                return [
                    {
                        "memory_id": hit.payload.get("memory_id"),
                        "text": hit.payload.get("text"),
                        "score": round(hit.score, 3),
                        "metadata": hit.payload,
                    }
                    for hit in results
                ]

            except Exception as e:
                logger.error(
                    f"Qdrant long-term memory search error: {e}. "
                    "Falling back to in-memory cosine search."
                )

        # ---------------------------------------------------------
        # In-memory fallback
        # ---------------------------------------------------------
        results = []

        for memory_id, data in self._in_memory_vectors.items():

            # Never return another student's memory.
            if data["user_id"] != user_id:
                continue

            # Ignore inactive memories.
            if not data["payload"].get("is_active", True):
                continue

            vector = data["vector"]

            # Vectors are normalized, so dot product represents
            # cosine similarity.
            similarity = sum(
                a * b
                for a, b in zip(query_vector, vector)
            )

            if similarity >= score_threshold:
                results.append(
                    {
                        "memory_id": memory_id,
                        "text": data["payload"].get("text", ""),
                        "score": round(similarity, 3),
                        "metadata": data["payload"],
                    }
                )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:limit]

    def delete_memory(self, memory_id: str) -> bool:
        """Deletes a vector point from Qdrant."""
        memory_id = str(memory_id)
        if self.client:
            try:
                from qdrant_client.http.models import PointIdsList
                self.client.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=PointIdsList(points=[memory_id]),
                )
            except Exception as e:
                logger.error(f"Qdrant delete error: {e}")

        self._in_memory_vectors.pop(memory_id, None)
        return True


# Global singleton instance
qdrant_adapter = QdrantAdapter()
