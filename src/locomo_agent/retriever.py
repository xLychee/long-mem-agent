"""Simple retriever for RAG-based question answering."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SimpleRetriever:
    """A simple retriever using sentence-transformers for embedding-based retrieval.

    This is a lightweight retriever that uses FAISS for efficient similarity search.
    It's designed for the LoCoMo evaluation where we need to retrieve relevant
    dialog turns or observations to answer questions.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ) -> None:
        """Initialize the retriever.

        Args:
            model_name: Name of the sentence-transformer model to use.
            top_k: Number of top results to retrieve.
        """
        self.model_name = model_name
        self.top_k = top_k
        self._model: Any = None
        self._index: Any = None
        self._documents: list[str] = []
        self._metadata: list[dict[str, Any]] = []

    def _load_model(self) -> None:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise

    def index_documents(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Index a list of documents for retrieval.

        Args:
            documents: List of text documents to index.
            metadata: Optional metadata for each document.
        """
        import faiss

        self._load_model()

        self._documents = documents
        self._metadata = metadata or [{} for _ in documents]

        logger.info(f"Indexing {len(documents)} documents...")

        # Generate embeddings
        embeddings = self._model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self._index.add(embeddings.astype(np.float32))

        logger.info(f"Indexed {len(documents)} documents with dimension {dimension}")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query string.
            top_k: Number of results to retrieve (overrides default).

        Returns:
            List of retrieved documents with scores and metadata.
        """
        if self._index is None:
            raise ValueError("No documents indexed. Call index_documents first.")

        self._load_model()

        k = top_k or self.top_k
        k = min(k, len(self._documents))

        # Encode query
        query_embedding = self._model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Search
        scores, indices = self._index.search(query_embedding.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty results
                continue
            results.append(
                {
                    "document": self._documents[idx],
                    "score": float(score),
                    "metadata": self._metadata[idx],
                    "index": int(idx),
                }
            )

        return results


class DialogChunker:
    """Utility to chunk conversation dialogs for retrieval."""

    def __init__(
        self,
        chunk_size: int = 5,
        overlap: int = 2,
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Number of dialog turns per chunk.
            overlap: Number of overlapping turns between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_conversation(
        self,
        sessions: list[Any],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Chunk a conversation into retrievable segments.

        Args:
            sessions: List of Session objects from a Conversation.

        Returns:
            Tuple of (chunks, metadata) for indexing.
        """
        chunks: list[str] = []
        metadata: list[dict[str, Any]] = []

        for session in sessions:
            turns = session.turns
            session_id = session.session_id
            date_time = session.date_time

            # Create chunks with overlap
            for i in range(0, len(turns), self.chunk_size - self.overlap):
                chunk_turns = turns[i : i + self.chunk_size]
                if not chunk_turns:
                    continue

                # Format chunk text
                chunk_lines = [f"[{session_id} - {date_time}]"]
                dia_ids = []
                for turn in chunk_turns:
                    chunk_lines.append(f"{turn.speaker}: {turn.text}")
                    dia_ids.append(turn.dia_id)

                chunks.append("\n".join(chunk_lines))
                metadata.append(
                    {
                        "session_id": session_id,
                        "date_time": date_time,
                        "dia_ids": dia_ids,
                        "start_index": i,
                    }
                )

        return chunks, metadata

    def chunk_observations(
        self,
        observations_raw: dict[str, Any],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Chunk observations for retrieval.

        Args:
            observations_raw: Raw observations dict from LoCoMo.
                Format: {session_key: {speaker: [[text, dia_id], ...]}}

        Returns:
            Tuple of (chunks, metadata) for indexing.
        """
        chunks: list[str] = []
        metadata: list[dict[str, Any]] = []
        idx = 0

        for session_key, session_obs in observations_raw.items():
            if isinstance(session_obs, dict):
                # Format: {speaker: [[text, dia_id], ...]}
                for speaker, obs_list in session_obs.items():
                    if isinstance(obs_list, list):
                        for obs in obs_list:
                            if isinstance(obs, list) and len(obs) >= 1:
                                text = obs[0]
                                dia_id = obs[1] if len(obs) > 1 else ""
                                chunks.append(f"[{session_key}] {speaker}: {text}")
                                metadata.append(
                                    {
                                        "session_id": session_key,
                                        "speaker": speaker,
                                        "dia_id": dia_id,
                                        "observation_index": idx,
                                        "type": "observation",
                                    }
                                )
                                idx += 1

        return chunks, metadata

    def chunk_summaries(
        self,
        summaries: dict[str, str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Chunk session summaries for retrieval.

        Args:
            summaries: Dictionary of session summaries.

        Returns:
            Tuple of (chunks, metadata) for indexing.
        """
        chunks: list[str] = []
        metadata: list[dict[str, Any]] = []

        for session_key, summary in summaries.items():
            chunks.append(f"[{session_key} Summary] {summary}")
            metadata.append(
                {
                    "session_id": session_key,
                    "type": "summary",
                }
            )

        return chunks, metadata

