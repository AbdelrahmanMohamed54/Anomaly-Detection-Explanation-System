"""
LangChain tool: semantic search over the ChromaDB 'incident_reports' collection.

Exposes:
    IncidentSearchTool             — class with search() method
    incident_search_tool           — LangChain @tool decorated callable
    (module-level singleton)       — shared IncidentSearchTool instance

The collection must be ingested first:
    python scripts/ingest_incident_reports.py
"""

from __future__ import annotations

import logging

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.tools import tool

from agent.schemas import IncidentSearchResult

logger = logging.getLogger(__name__)

CHROMA_DIR: str = ".chroma"
COLLECTION_NAME: str = "incident_reports"
EMBED_MODEL: str = "all-MiniLM-L6-v2"


class IncidentSearchTool:
    """Semantic search over historical incident reports stored in ChromaDB.

    Attributes:
        collection: The ChromaDB collection for incident reports.
    """

    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        """Connect to the ChromaDB incident_reports collection.

        Args:
            chroma_dir:      Path to the ChromaDB persistence directory.
            collection_name: Name of the ChromaDB collection.
            embed_model:     Sentence-transformer model name for embeddings.
        """
        self._embed_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model)
        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "IncidentSearchTool: connected to '%s' (%d docs)",
            collection_name,
            self.collection.count(),
        )

    def search(self, query: str, top_k: int = 3) -> list[IncidentSearchResult]:
        """Embed *query* and retrieve the top_k most similar incident chunks.

        Args:
            query: Free-text query describing the anomaly or symptom.
            top_k: Number of results to return (default 3).

        Returns:
            List of IncidentSearchResult sorted by similarity (best first).
        """
        if self.collection.count() == 0:
            logger.warning("IncidentSearchTool: collection is empty.")
            return []

        response = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        results: list[IncidentSearchResult] = []
        docs = response["documents"][0]
        metas = response["metadatas"][0]
        dists = response["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to similarity score [0, 1].
            similarity = float(max(0.0, 1.0 - dist / 2.0))
            results.append(
                IncidentSearchResult(
                    text=doc,
                    source_file=meta.get("source_file", "unknown"),
                    anomaly_type=meta.get("anomaly_type", "unknown"),
                    similarity_score=round(similarity, 4),
                )
            )
        return results


# Module-level singleton (lazy-initialized inside the @tool to avoid heavy
# imports at module load time — the tool is only instantiated on first call).
_incident_search_instance: IncidentSearchTool | None = None


def _get_incident_search() -> IncidentSearchTool:
    global _incident_search_instance
    if _incident_search_instance is None:
        _incident_search_instance = IncidentSearchTool()
    return _incident_search_instance


# ── LangChain tool ─────────────────────────────────────────────────────────────


@tool
def incident_search_tool(query: str, top_k: int = 3) -> str:
    """Search historical incident reports for similar anomaly patterns and resolutions.

    Use this tool SECOND after querying the historical database. It performs
    semantic search over engineering incident reports to find the most relevant
    past cases — including symptoms, diagnostics, root causes, and resolutions.

    Args:
        query: Description of the anomaly symptoms, affected component, or anomaly type.
        top_k: Number of incident report excerpts to retrieve (default 3).

    Returns:
        Formatted string of the most relevant incident report excerpts with
        similarity scores and source references.
    """
    searcher = _get_incident_search()
    results = searcher.search(query, top_k=top_k)

    if not results:
        return "No relevant incident reports found for this query."

    lines = [f"Found {len(results)} relevant incident report(s):\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] Source: {r.source_file} | Type: {r.anomaly_type} | "
            f"Similarity: {r.similarity_score:.3f}\n{r.text}\n"
        )
    return "\n".join(lines)
