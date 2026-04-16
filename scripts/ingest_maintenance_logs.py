"""
Ingest maintenance logs into ChromaDB collection 'maintenance_logs'.

Processing pipeline:
    1. Read all .txt files from data/maintenance_logs/
    2. Chunk each file on double newline (one log entry per chunk)
    3. Embed with sentence-transformers all-MiniLM-L6-v2
    4. Store in ChromaDB with metadata: source_file, maintenance_type, chunk_index
    5. Print ingestion summary

ChromaDB persists to: .chroma/maintenance_logs/

Run:
    python scripts/ingest_maintenance_logs.py
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── Config ────────────────────────────────────────────────────────────────────

LOGS_DIR: Path = Path("data/maintenance_logs")
CHROMA_DIR: str = ".chroma"
COLLECTION_NAME: str = "maintenance_logs"
EMBED_MODEL: str = "all-MiniLM-L6-v2"

# Map filename stem -> maintenance_type label
_MAINTENANCE_TYPE_MAP: dict[str, str] = {
    "preventive_maintenance": "preventive",
    "corrective_actions": "corrective",
}


# ── Core functions ────────────────────────────────────────────────────────────


def load_and_chunk(filepath: Path) -> list[str]:
    """Read a maintenance log file and split into entry-level chunks.

    Chunks are separated by one or more blank lines. Empty/whitespace-only
    chunks are discarded.

    Args:
        filepath: Path to the .txt maintenance log file.

    Returns:
        List of non-empty text chunks.
    """
    text = filepath.read_text(encoding="utf-8")
    raw_chunks = re.split(r"\n{2,}", text)
    return [c.strip() for c in raw_chunks if c.strip()]


def infer_maintenance_type(filename_stem: str) -> str:
    """Resolve the maintenance_type label from the filename stem.

    Args:
        filename_stem: File name without extension (e.g. 'preventive_maintenance').

    Returns:
        Maintenance type label, or 'general' if no match.
    """
    return _MAINTENANCE_TYPE_MAP.get(filename_stem, "general")


def ingest_logs(
    logs_dir: Path = LOGS_DIR,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, int]:
    """Ingest all .txt maintenance log files into ChromaDB.

    Existing documents in the collection are replaced (collection is cleared
    before ingestion to make the operation idempotent).

    Args:
        logs_dir:        Directory containing the .txt maintenance log files.
        chroma_dir:      Path to the ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection to populate.

    Returns:
        Dict mapping maintenance_type -> chunk count ingested.
    """
    txt_files = sorted(logs_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {logs_dir}")

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
        maintenance_type = infer_maintenance_type(filepath.stem)
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
                    "maintenance_type": maintenance_type,
                    "chunk_index": global_chunk_idx,
                    "local_chunk_index": local_idx,
                }
            )
            global_chunk_idx += 1

        # Batch upsert to ChromaDB
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        summary[maintenance_type] = summary.get(maintenance_type, 0) + len(chunks)
        print(
            f"  [{maintenance_type:<15}] {filepath.name:<45} "
            f"{len(chunks):>3} chunks ingested"
        )

    return summary


def _print_summary(summary: dict[str, int]) -> None:
    """Print the ingestion summary table."""
    print("\n" + "=" * 55)
    print("  MAINTENANCE LOGS INGESTION SUMMARY")
    print("=" * 55)
    total = 0
    for maintenance_type, count in sorted(summary.items()):
        print(f"  {maintenance_type:<25} {count:>4} chunks")
        total += count
    print(f"  {'TOTAL':<25} {total:>4} chunks")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Stored at:  {Path(CHROMA_DIR).resolve()}")
    print("=" * 55 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Run the maintenance logs ingestion pipeline."""
    print(f"Ingesting maintenance logs from {LOGS_DIR.resolve()} ...\n")
    summary = ingest_logs()
    _print_summary(summary)


if __name__ == "__main__":
    main()
