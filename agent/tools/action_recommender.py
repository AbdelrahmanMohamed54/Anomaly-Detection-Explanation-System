"""
LangChain tool: semantic search over the ChromaDB 'maintenance_logs' collection
to retrieve corrective actions relevant to an identified fault.

Exposes:
    ActionRecommenderTool          — class with search() method
    corrective_action_tool         — LangChain @tool decorated callable
    (module-level singleton)       — shared ActionRecommenderTool instance

The collection must be ingested first:
    python scripts/ingest_maintenance_logs.py
"""

from __future__ import annotations

import logging

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.tools import tool

from agent.schemas import ActionRecommendation

logger = logging.getLogger(__name__)

CHROMA_DIR: str = ".chroma"
COLLECTION_NAME: str = "maintenance_logs"
EMBED_MODEL: str = "all-MiniLM-L6-v2"


class ActionRecommenderTool:
    """Semantic search over maintenance logs stored in ChromaDB.

    Retrieves corrective and preventive maintenance procedures relevant to
    the identified root cause or fault type.

    Attributes:
        collection: The ChromaDB collection for maintenance logs.
    """

    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        """Connect to the ChromaDB maintenance_logs collection.

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
            "ActionRecommenderTool: connected to '%s' (%d docs)",
            collection_name,
            self.collection.count(),
        )

    def search(self, query: str, top_k: int = 3) -> list[ActionRecommendation]:
        """Retrieve top_k maintenance log entries most relevant to *query*.

        Args:
            query: Description of the fault, root cause, or component.
            top_k: Number of maintenance log entries to return (default 3).

        Returns:
            List of ActionRecommendation sorted by similarity (best first).
        """
        if self.collection.count() == 0:
            logger.warning("ActionRecommenderTool: collection is empty.")
            return []

        response = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        results: list[ActionRecommendation] = []
        docs = response["documents"][0]
        metas = response["metadatas"][0]
        dists = response["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            similarity = float(max(0.0, 1.0 - dist / 2.0))
            results.append(
                ActionRecommendation(
                    text=doc,
                    source_file=meta.get("source_file", "unknown"),
                    maintenance_type=meta.get("maintenance_type", "general"),
                    similarity_score=round(similarity, 4),
                )
            )
        return results


# Module-level singleton (lazy-initialized inside the @tool).
_action_recommender_instance: ActionRecommenderTool | None = None


def _get_action_recommender() -> ActionRecommenderTool:
    global _action_recommender_instance
    if _action_recommender_instance is None:
        _action_recommender_instance = ActionRecommenderTool()
    return _action_recommender_instance


# ── LangChain tool ─────────────────────────────────────────────────────────────


@tool
def corrective_action_tool(query: str, top_k: int = 3) -> str:
    """Retrieve recommended corrective actions from maintenance logs based on root cause.

    Use this tool THIRD, after querying historical anomalies and incident reports.
    It searches maintenance logs for step-by-step corrective procedures relevant
    to the identified root cause and component type.

    Args:
        query: Description of the fault type, component, or recommended repair.
        top_k: Number of maintenance log excerpts to retrieve (default 3).

    Returns:
        Formatted string of the most relevant maintenance log entries with
        similarity scores and source references.
    """
    recommender = _get_action_recommender()
    results = recommender.search(query, top_k=top_k)

    if not results:
        return "No relevant maintenance log entries found for this query."

    lines = [f"Found {len(results)} relevant maintenance log(s):\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] Source: {r.source_file} | Type: {r.maintenance_type} | "
            f"Similarity: {r.similarity_score:.3f}\n{r.text}\n"
        )
    return "\n".join(lines)
