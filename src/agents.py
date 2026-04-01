import json
import logging
from typing import Any

from redis.exceptions import RedisError

from src.embedding import get_embedding
from src.generator import refine_query, generate_response
from src.retriever import get_keyword_score, retrieve

logger = logging.getLogger(__name__)

class AgentPipelineError(Exception):
    pass


def query_agent(question: str) -> dict[str, str]:
    logger.info("Query Agent | validating input")
    if not question or not question.strip():
        raise AgentPipelineError("Question cannot be empty.")

    question = question.strip()
    refined = refine_query(question)
    logger.info("Query Agent | finished refinement")
    return {"question": question, "refined_query": refined}


def retrieval_agent(refined_query: str, top_n: int = 3) -> list[dict[str, str]]:
    logger.info("Retrieval Agent | generating query embedding")
    query_embedding = get_embedding(refined_query)
    if not query_embedding:
        raise AgentPipelineError("Failed to generate query embedding.")

    logger.info("Retrieval Agent | running FAISS retrieval")
    candidate_chunks = retrieve(refined_query, query_embedding, top_n=top_n * 3)
    if not candidate_chunks:
        logger.info("Retrieval Agent | no candidates found")
        return []

    logger.info("Retrieval Agent | applying hybrid ranking")
    ranked: list[tuple[float, dict[str, str]]] = []
    for chunk in candidate_chunks:
        score = get_keyword_score(refined_query, chunk["text"])
        ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)

    selected_sources: set[str] = set()
    final_chunks: list[dict[str, str]] = []
    for score, chunk in ranked:
        if len(final_chunks) >= top_n:
            break
        if chunk["source_name"] not in selected_sources:
            final_chunks.append(chunk)
            selected_sources.add(chunk["source_name"])

    for score, chunk in ranked:
        if len(final_chunks) >= top_n:
            break
        if chunk not in final_chunks:
            final_chunks.append(chunk)

    source_distribution: dict[str, int] = {}
    for chunk in final_chunks:
        source_distribution[chunk["source"]] = source_distribution.get(chunk["source"], 0) + 1

    logger.info(
        "Retrieval Agent | source distribution: %s",
        json.dumps(source_distribution),
    )
    logger.info("Retrieval Agent | returning %d chunks", len(final_chunks))
    return final_chunks


def reasoning_agent(refined_query: str, context_chunks: list[dict[str, str]], level: str = "beginner", assistant_mode: str = "assist") -> dict[str, object]:
    logger.info("Reasoning Agent | generating answer | level=%s | assistant_mode=%s", level, assistant_mode)
    answer = generate_response(refined_query, [chunk["text"] for chunk in context_chunks], level=level, assistant_mode=assistant_mode)
    logger.info("Reasoning Agent | generation complete")
    return answer


def response_agent(payload: dict[str, object], sources: list[dict[str, str]], meta: dict[str, float] | None = None) -> dict[str, Any]:
    logger.info("Response Agent | formatting final output")
    response_payload = payload.copy()
    response_payload["sources"] = sources
    if meta is not None:
        response_payload["meta"] = meta
    return response_payload


def fallback_response(reason: str, meta: dict[str, float] | None = None) -> dict[str, Any]:
    logger.error("Fallback Response | %s", reason)
    payload: dict[str, Any] = {
        "understanding": "I am unable to process this request completely right now.",
        "key_points": [],
        "explanation": "Please try again later.",
        "real_world_example": "",
        "next_steps": [],
        "sources": [],
    }
    if meta is not None:
        payload["meta"] = meta
    return payload
