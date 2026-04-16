"""
Ingest incident reports into ChromaDB collection 'incident_reports'.

Processing pipeline:
    1. Read all .txt files from data/incident_reports/
    2. Chunk each file on double newline (one report per chunk)
    3. Embed with sentence-transformers all-MiniLM-L6-v2
    4. Store in ChromaDB with metadata: source_file, anomaly_type, chunk_index
    5. Print ingestion summary

ChromaDB persists to: .chroma/incident_reports/

Run:
    python scripts/ingest_incident_reports.py
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── Config ────────────────────────────────────────────────────────────────────

REPORTS_DIR: Path = Path("data/incident_reports")
CHROMA_DIR: str = ".chroma"
COLLECTION_NAME: str = "incident_reports"
EMBED_MODEL: str = "all-MiniLM-L6-v2"

# Map filename stem -> anomaly_type label
_ANOMALY_TYPE_MAP: dict[str, str] = {
    "bearing_wear_incidents": "bearing_wear",
    "pressure_drop_incidents": "pressure_drop",
    "overload_incidents": "overload",
}


# ── Core functions ────────────────────────────────────────────────────────────


def load_and_chunk(filepath: Path) -> list[str]:
    """Read a report file and split it into paragraph-level chunks.

    Chunks are separated by one or more blank lines. Empty chunks and
    whitespace-only chunks are discarded.

    Args:
        filepath: Path to the .txt incident report file.

    Returns:
        List of non-empty text chunks.
    """
    text = filepath.read_text(encoding="utf-8")
    raw_chunks = re.split(r"\n{2,}", text)
    return [c.strip() for c in raw_chunks if c.strip()]


def infer_anomaly_type(filename_stem: str) -> str:
    """Resolve the anomaly_type label from the filename stem.

    Args:
        filename_stem: File name without extension (e.g. 'bearing_wear_incidents').

    Returns:
        Anomaly type label, or 'unknown' if no match.
    """
    return _ANOMALY_TYPE_MAP.get(filename_stem, "unknown")


def ingest_reports(
    reports_dir: Path = REPORTS_DIR,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, int]:
    """Ingest all .txt report files into ChromaDB.

    Existing documents in the collection are replaced (collection is cleared
    before ingestion to make the operation idempotent).

    Args:
        reports_dir:     Directory containing the .txt incident report files.
        chroma_dir:      Path to the ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection to populate.

    Returns:
        Dict mapping anomaly_type -> chunk count ingested.
    """
    txt_files = sorted(reports_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {reports_dir}")

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    client = chromadb.PersistentClient(path=chroma_dir)

    # Delete and recreate for idempotency
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    summary: dict[str, int] = {}
    global_chunk_idx = 0

    for filepath in txt_files:
        anomaly_type = infer_anomaly_type(filepath.stem)
        chunks = load_and_chunk(filepath)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for local_idx, chunk in enumerate(chunks):
            doc_id = f"{filepath.stem}_chunk_{local_idx:03d}"
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "source_file": filepath.name,
                    "anomaly_type": anomaly_type,
                    "chunk_index": global_chunk_idx,
                    "local_chunk_index": local_idx,
                }
            )
            global_chunk_idx += 1

        # Batch upsert to ChromaDB
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        summary[anomaly_type] = summary.get(anomaly_type, 0) + len(chunks)
        print(
            f"  [{anomaly_type:<20}] {filepath.name:<40} "
            f"{len(chunks):>3} chunks ingested"
        )

    return summary


def _print_summary(summary: dict[str, int]) -> None:
    """Print the ingestion summary table."""
    print("\n" + "=" * 55)
    print("  INCIDENT REPORTS INGESTION SUMMARY")
    print("=" * 55)
    total = 0
    for anomaly_type, count in sorted(summary.items()):
        print(f"  {anomaly_type:<25} {count:>4} chunks")
        total += count
    print(f"  {'TOTAL':<25} {total:>4} chunks")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Stored at:  {Path(CHROMA_DIR).resolve()}")
    print("=" * 55 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Run the incident reports ingestion pipeline."""
    print(f"Ingesting incident reports from {REPORTS_DIR.resolve()} ...\n")
    summary = ingest_reports()
    _print_summary(summary)


if __name__ == "__main__":
    main()
