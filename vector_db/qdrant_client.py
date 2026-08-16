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


def _generate_fallback_embedding(text: str, dim: int = 128) -> List[float]:
    """Generates a deterministic pseudo-semantic dense vector for offline/testing."""
    vec = [0.0] * dim
    words = text.lower().split()
    if not words:
        return vec

    for word in words:
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        for i in range(dim):
            bit = (h >> (i % 32)) & 1
            vec[i] += 1.0 if bit else -1.0

    # Normalize vector to unit length
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [round(x / norm, 5) for x in vec]
    return vec


class QdrantAdapter:
    """Adapter for Qdrant Vector Database semantic memory store."""

    COLLECTION_NAME = "student_memories"
    VECTOR_DIM = 128

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
            from qdrant_client.http.models import Distance, VectorParams
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]
            if self.COLLECTION_NAME not in names:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(size=self.VECTOR_DIM, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection '{self.COLLECTION_NAME}'.")
        except Exception as e:
            logger.warning(f"Error ensuring Qdrant collection ({e}).")

    def _get_embedding(self, text: str) -> List[float]:
        """Generates embedding vector for memory text."""
        return _generate_fallback_embedding(text, self.VECTOR_DIM)

    def upsert_memory(
        self,
        memory_id: str,
        text: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Embeds and stores a candidate memory in Qdrant vector store.
        """
        vector = self._get_embedding(text)
        payload = {
            "memory_id": str(memory_id),
            "user_id": str(user_id),
            "text": text,
            **(metadata or {}),
        }

        # If live Qdrant is connected
        if self.client:
            try:
                from qdrant_client.http.models import PointStruct
                # Qdrant accepts integer or UUID point IDs
                self.client.upsert(
                    collection_name=self.COLLECTION_NAME,
                    points=[PointStruct(id=str(memory_id), vector=vector, payload=payload)],
                )
                logger.debug(f"Upserted memory {memory_id} to Qdrant.")
                return True
            except Exception as e:
                logger.error(f"Qdrant upsert error: {e}. Falling back to in-memory vector storage.")

        # In-memory store fallback
        self._in_memory_vectors[str(memory_id)] = {
            "id": str(memory_id),
            "vector": vector,
            "payload": payload,
            "user_id": str(user_id),
        }
        return True

    def search_memories(
        self,
        user_id: str,
        query_text: str,
        limit: int = 5,
        score_threshold: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search for student memories.
        """
        user_id = str(user_id)
        query_vector = self._get_embedding(query_text)

        # If live Qdrant is connected
        if self.client:
            try:
                from qdrant_client.http.models import FieldCondition, Filter, MatchValue
                search_filter = Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
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
                logger.error(f"Qdrant search error: {e}. Falling back to in-memory cosine search.")

        # In-memory cosine similarity search
        results = []
        for mem_id, data in self._in_memory_vectors.items():
            if data["user_id"] != user_id:
                continue
            v = data["vector"]
            # Cosine similarity for normalized vectors = dot product
            dot = sum(a * b for a, b in zip(query_vector, v))
            if dot >= score_threshold:
                results.append({
                    "memory_id": mem_id,
                    "text": data["payload"].get("text", ""),
                    "score": round(dot, 3),
                    "metadata": data["payload"],
                })

        results.sort(key=lambda x: x["score"], reverse=True)
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
