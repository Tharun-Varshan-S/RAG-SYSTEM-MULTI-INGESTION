import json
import logging
import os
import re
from typing import TypedDict

import faiss
import numpy as np

logger = logging.getLogger(__name__)
STORE_PATH = os.path.abspath(
    os.environ.get(
        "RAG_STORE_PATH",
        os.path.join(os.path.dirname(__file__), "..", "data", "vector_store.json"),
    )
)


def get_keyword_score(query: str, text: str) -> float:
    q_tokens = set(re.findall(r'\w+', query.lower()))
    t_tokens = set(re.findall(r'\w+', text.lower()))
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens.intersection(t_tokens))
    return min(1.0, overlap / len(q_tokens))

class DocumentInfo(TypedDict):
    text: str
    embedding: list[float]
    source_type: str
    source_name: str

# FAISS-backed storage
documents: list[DocumentInfo] = []
faiss_index: faiss.IndexFlatIP | None = None
index_dimension: int | None = None

def _normalize_embedding(embedding: list[float]) -> np.ndarray:
    vector = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm

def _create_faiss_index(dimension: int) -> faiss.IndexFlatIP:
    logger.info("Creating FAISS index with dimension %d", dimension)
    return faiss.IndexFlatIP(dimension)


def save_database() -> None:
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf-8") as handle:
        json.dump({"documents": documents}, handle)


def load_database() -> int:
    global faiss_index, index_dimension, documents

    documents = []
    faiss_index = None
    index_dimension = None

    if not os.path.exists(STORE_PATH):
        logger.info("No persisted vector store found at %s", STORE_PATH)
        return 0

    try:
        with open(STORE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as e:
        logger.warning("Failed to load persisted vector store: %s", e)
        return 0

    loaded_docs = payload.get("documents", []) if isinstance(payload, dict) else []
    for item in loaded_docs:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        source_type = str(item.get("source_type", "text"))
        source_name = str(item.get("source_name", "unknown"))
        embedding = item.get("embedding")

        if not text or not isinstance(embedding, list) or not embedding:
            continue

        normalized_embedding = _normalize_embedding(embedding)
        if index_dimension is None:
            index_dimension = normalized_embedding.shape[0]
            faiss_index = _create_faiss_index(index_dimension)

        if normalized_embedding.shape[0] != index_dimension:
            logger.warning("Skipping persisted chunk with incompatible embedding dimension.")
            continue

        faiss_index.add(normalized_embedding.reshape(1, -1))
        documents.append({
            "text": text,
            "embedding": normalized_embedding.tolist(),
            "source_type": source_type,
            "source_name": source_name,
        })

    logger.info("Loaded %d persisted chunks from %s", len(documents), STORE_PATH)
    return len(documents)


def list_sources() -> list[dict[str, str]]:
    unique_sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for doc in reversed(documents):
        key = (doc["source_name"], doc["source_type"])
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append({"name": doc["source_name"], "type": doc["source_type"]})

    return unique_sources

def add_to_database(
    chunk: str,
    embedding: list[float],
    source_type: str = "text",
    source_name: str = "unknown",
    persist: bool = True,
) -> bool:
    """
    Stores a valid chunk and its embedding into the FAISS index.
    Keeps an index → chunk mapping for retrieval.
    """
    global faiss_index, index_dimension

    if not chunk or not chunk.strip():
        logger.warning("Skipped storing empty chunk.")
        return False

    if not embedding:
        logger.warning("Skipped storing chunk with empty embedding.")
        return False

    if any(doc['text'] == chunk for doc in documents):
        logger.info("Duplicate chunk found, skipping storage.")
        return False

    normalized_embedding = _normalize_embedding(embedding)
    if index_dimension is None:
        index_dimension = normalized_embedding.shape[0]
        faiss_index = _create_faiss_index(index_dimension)

    if normalized_embedding.shape[0] != index_dimension:
        logger.error(
            "Embedding dimension mismatch: expected %d, got %d",
            index_dimension,
            normalized_embedding.shape[0]
        )
        return False

    faiss_index.add(normalized_embedding.reshape(1, -1))
    documents.append({
        "text": chunk,
        "embedding": normalized_embedding.tolist(),
        "source_type": source_type,
        "source_name": source_name,
    })
    if persist:
        save_database()
    logger.info("Chunk added to FAISS index. Index size: %d", faiss_index.ntotal)
    return True

def retrieve(query: str, query_embedding: list[float], top_n: int = 3) -> list[dict[str, str]]:
    """
    Searches the FAISS index for the top K vectors and returns the matching chunks with source metadata.
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to retrieve.")
        return []

    if faiss_index is None or not documents:
        logger.warning("No indexed data available for retrieval.")
        return []

    if not query_embedding:
        logger.warning("Query embedding is empty.")
        return []

    normalized_query = _normalize_embedding(query_embedding)
    if index_dimension is None or normalized_query.shape[0] != index_dimension:
        logger.error(
            "Query embedding dimension mismatch: expected %s, got %d",
            index_dimension,
            normalized_query.shape[0]
        )
        return []

    k = min(top_n, faiss_index.ntotal)
    distances, indices = faiss_index.search(normalized_query.reshape(1, -1), k)

    logger.info("FAISS index size: %d", faiss_index.ntotal)
    results: list[dict[str, str]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(documents):
            continue
        doc = documents[int(idx)]
        results.append({
            "text": doc["text"],
            "source": doc["source_type"],
            "source_name": doc["source_name"],
        })
        logger.info("Search result idx=%d score=%.4f source=%s source_name=%s", int(idx), float(score), doc["source_type"], doc["source_name"])

    if not results:
        return []

    return results


load_database()
