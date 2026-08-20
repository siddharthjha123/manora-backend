"""BGE-M3 embedding client used by MANORA's vector-storage layer.

This module is intentionally independent from the Qdrant adapter.  It only
owns model loading and text-to-vector conversion, which lets the embedding
implementation be verified before Qdrant starts depending on it.
"""

import logging
import threading
from typing import List, Optional, Sequence

logger = logging.getLogger("manora.embedding")


class EmbeddingClient:
    """Create normalized 1024-dimensional embeddings with BAAI/bge-m3.

    The model is loaded on the first embedding request instead of during module
    import.  This keeps commands that do not use semantic search lightweight
    and avoids downloading or allocating the model merely by importing MANORA.
    """

    MODEL_NAME = "BAAI/bge-m3"
    VECTOR_DIMENSION = 1024

    def __init__(self, device: Optional[str] = None) -> None:
        """Configure the client.

        Args:
            device: Optional SentenceTransformer device such as ``"cuda"`` or
                ``"cpu"``. When omitted, SentenceTransformer selects the best
                available device.
        """

        self.device = device
        self._model = None
        self._load_lock = threading.Lock()

    def _get_model(self):
        """Load BGE-M3 once and return the shared model instance."""

        if self._model is not None:
            return self._model

        # Two callers may request the first embedding concurrently. The lock
        # prevents duplicate downloads and duplicate model allocations.
        with self._load_lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError(
                        "sentence-transformers is required for BGE-M3 embeddings"
                    ) from exc

                logger.info("Loading embedding model %s", self.MODEL_NAME)
                self._model = SentenceTransformer(
                    self.MODEL_NAME,
                    device=self.device,
                )
                logger.info(
                    "Loaded embedding model %s on %s",
                    self.MODEL_NAME,
                    self._model.device,
                )

        return self._model

    @classmethod
    def _validate_vector(cls, vector: Sequence[float]) -> List[float]:
        """Convert one model result to plain floats and verify its dimension."""

        result = [float(value) for value in vector]
        if len(result) != cls.VECTOR_DIMENSION:
            raise ValueError(
                f"{cls.MODEL_NAME} returned {len(result)} dimensions; "
                f"expected {cls.VECTOR_DIMENSION}"
            )
        return result

    def embed_text(self, text: str) -> List[float]:
        """Embed one non-empty string as a normalized 1024-value vector."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must not be empty")

        vector = self._get_model().encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._validate_vector(vector)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed multiple strings efficiently in one model batch.

        An empty input sequence returns an empty list without loading BGE-M3.
        """

        values = list(texts)
        if not values:
            return []
        if any(not isinstance(text, str) for text in values):
            raise TypeError("every text must be a string")
        if any(not text.strip() for text in values):
            raise ValueError("texts must not contain empty strings")

        vectors = self._get_model().encode(
            values,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [self._validate_vector(vector) for vector in vectors]


# A reusable lazy singleton for the future Qdrant integration. Importing this
# object does not load the model; the first embed_text/embed_texts call does.
embedding_client = EmbeddingClient()

