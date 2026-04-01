import io
import json
import logging
import os
import re
import time
import uuid
from typing import List

from urllib.parse import parse_qs, urlparse

import redis
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from redis.exceptions import RedisError
from youtube_transcript_api import YouTubeTranscriptApi

from src.agents import (
    AgentPipelineError,
    fallback_response,
    query_agent,
    reasoning_agent,
    response_agent,
    retrieval_agent,
)
from src.embedding import get_embedding
from src.ingestion import chunk_text
from src.retriever import add_to_database, list_sources, load_database, save_database

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger("rag-api")

VALID_MODES = {"beginner", "normal", "expert"}
VALID_INTERACTION_MODES = {"teach", "assist"}

app = FastAPI(title="RAG Backend System")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas
class AskRequest(BaseModel):
    question: str
    user_id: str = "default_user"
    mode: str | None = None
    assistant_mode: str | None = None

class SetPreferenceRequest(BaseModel):
    user_id: str
    level: str

class SourceInfo(BaseModel):
    text: str
    source: str
    source_name: str

class AskResponse(BaseModel):
    understanding: str
    key_points: List[str]
    explanation: str
    real_world_example: str
    next_steps: List[str]
    sources: List[SourceInfo]
    meta: dict[str, float]

class IngestRequest(BaseModel):
    text: str
    source_type: str = "text"
    source_name: str = "unknown"

class SourceItem(BaseModel):
    type: str
    content: str
    source_name: str

class BatchIngestRequest(BaseModel):
    sources: List[SourceItem]

class IngestResponse(BaseModel):
    status: str
    chunks_added: int


class SourceCatalogItem(BaseModel):
    name: str
    type: str


class SourceCatalogResponse(BaseModel):
    sources: List[SourceCatalogItem]

user_preferences: dict[str, str] = {}
user_memory: dict[str, dict[str, object]] = {}
redis_client: redis.Redis | None = None


def normalize_mode(mode: str | None) -> str:
    if not mode:
        return "normal"
    normalized = mode.strip().lower()
    if normalized in VALID_MODES:
        return normalized
    return "normal"


def normalize_interaction_mode(mode: str | None) -> str:
    if not mode:
        return "assist"
    normalized = mode.strip().lower()
    if normalized in VALID_INTERACTION_MODES:
        return normalized
    return "assist"


def infer_mode_from_query(question: str) -> str:
    text = question.strip().lower()
    easy_keywords = ["what is", "who is", "define", "explain", "how does", "simple", "beginner", "why", "tell me", "example", "use case"]
    expert_keywords = ["difference between", "implement", "algorithm", "architecture", "mathematical", "optimize", "performance", "scalable", "deep learning", "neural", "security", "cryptography", "statistics", "derive", "proof", "complex", "advanced"]

    if any(token in text for token in expert_keywords):
        return "expert"
    if any(token in text for token in easy_keywords):
        return "beginner"
    if len(text.split()) <= 5:
        return "beginner"
    return "normal"


def get_user_memory(user_id: str) -> dict[str, object]:
    return user_memory.setdefault(user_id, {"queries": [], "preferred_mode": "normal", "preferred_interaction": "assist"})


def resolve_explanation_mode(req_mode: str | None, user_id: str, question: str) -> tuple[str, str]:
    explicit_mode = normalize_mode(req_mode)
    memory = get_user_memory(user_id)
    inferred_mode = infer_mode_from_query(question)
    memory_exists = bool(memory["queries"])

    if req_mode is not None:
        memory["preferred_mode"] = explicit_mode
        return explicit_mode, "explicit"

    if not memory_exists:
        if inferred_mode in {"beginner", "expert"}:
            return inferred_mode, "inferred"
        return "beginner", "default"

    stored_mode = memory.get("preferred_mode", "normal")
    if stored_mode == inferred_mode:
        return stored_mode, "memory"

    return "normal", "conflict"


def infer_interaction_mode(question: str) -> str:
    text = question.strip().lower()
    if any(token in text for token in ["fix", "bug", "error", "debug", "stack trace", "issue", "refactor", "code"]):
        return "assist"
    if any(token in text for token in ["explain", "learn", "understand", "concept", "how does", "what is"]):
        return "teach"
    return "assist"


def resolve_interaction_mode(req_mode: str | None, user_id: str, question: str) -> tuple[str, str]:
    explicit_mode = normalize_interaction_mode(req_mode)
    memory = get_user_memory(user_id)
    inferred_mode = infer_interaction_mode(question)
    memory_exists = bool(memory["queries"])

    if req_mode is not None:
        memory["preferred_interaction"] = explicit_mode
        return explicit_mode, "explicit"

    if not memory_exists:
        return inferred_mode, "inferred"

    stored_interaction = memory.get("preferred_interaction", "assist")
    if stored_interaction == inferred_mode:
        return stored_interaction, "memory"

    return "assist", "conflict"


def is_vague_query(question: str) -> bool:
    text = question.strip().lower()
    if not text or len(text) <= 2:
        return True
    vague_tokens = ["ai", "what", "thing", "something", "explain", "help"]
    if text in vague_tokens or text.endswith("?") and len(text.split()) <= 2:
        return True
    return False


def log_step(request_id: str, user_id: str, step_name: str, duration: float, extra: dict | None = None) -> None:
    payload = {
        "request_id": request_id,
        "user_id": user_id,
        "step": step_name,
        "duration": round(duration, 4),
        "timestamp": time.time(),
    }
    if extra:
        payload.update(extra)
    logger.info(json.dumps(payload))


def log_error(request_id: str, user_id: str, step_name: str, error: Exception, extra: dict | None = None) -> None:
    payload = {
        "request_id": request_id,
        "user_id": user_id,
        "step": step_name,
        "error": str(error),
        "timestamp": time.time(),
    }
    if extra:
        payload.update(extra)
    logger.error(json.dumps(payload))


def _init_redis() -> redis.Redis | None:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        client.ping()
        logger.info("Redis cache is available.")
        return client
    except Exception as e:
        logger.warning("Redis cache unavailable, continuing without cache: %s", e)
        return None


def get_cached_response(key: str) -> dict | None:
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(key)
        if raw:
            logger.info("Cache hit for key: %s", key)
            return json.loads(raw)
        logger.info("Cache miss for key: %s", key)
    except RedisError as e:
        logger.warning("Redis get failed, skipping cache: %s", e)
    return None


def set_cached_response(key: str, value: dict, expire_seconds: int = 300) -> None:
    if redis_client is None:
        return
    try:
        redis_client.set(key, json.dumps(value), ex=expire_seconds)
        logger.info("Stored response in Redis cache for key: %s", key)
    except RedisError as e:
        logger.warning("Redis set failed, skipping cache store: %s", e)


def normalize_source_type(source_type: str | None) -> str:
    normalized = (source_type or "text").strip().lower()
    if normalized == "video":
        return "youtube"
    if normalized in {"document", "code", "pdf", "youtube", "text"}:
        return normalized
    return "text"


def store_ingested_text(text: str, source_type: str, source_name: str) -> int:
    chunks = chunk_text(text, chunk_size=200)
    if not chunks:
        raise HTTPException(status_code=400, detail="Failed to create text chunks from input.")

    chunks_added = 0
    for chunk in chunks:
        try:
            emb = get_embedding(chunk)
            if emb:
                added = add_to_database(
                    chunk,
                    emb,
                    source_type=source_type,
                    source_name=source_name,
                    persist=False,
                )
                if added:
                    chunks_added += 1
        except Exception as e:
            logger.error("Error generating embedding for chunk: %s", e)

    if chunks_added == 0:
        raise HTTPException(status_code=500, detail="Ingestion failed. Ensure API keys or DB are configured correctly.")

    save_database()

    return chunks_added


def extract_youtube_video_id(source_url: str) -> str | None:
    parsed = urlparse(source_url.strip())
    query_params = parse_qs(parsed.query)
    if query_params.get("v"):
        return query_params["v"][0]

    path = parsed.path.strip("/")
    if parsed.netloc in {"youtu.be", "www.youtu.be"} and path:
        return path.split("/")[0]

    match = re.search(r"(?:v=|/shorts/|/embed/|/)([A-Za-z0-9_-]{11})(?:[?&/]|$)", source_url)
    if match:
        return match.group(1)
    return None


def extract_youtube_transcript(source_url: str) -> str:
    video_id = extract_youtube_video_id(source_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Provide a valid YouTube URL.")

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=("en",))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to fetch a YouTube transcript: {e}") from e

    transcript_text = "\n".join(snippet.text for snippet in transcript.snippets).strip()
    if not transcript_text:
        raise HTTPException(status_code=400, detail="The YouTube transcript was empty.")
    return transcript_text


def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read the PDF file: {e}") from e

    page_texts: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            logger.warning("Skipping unreadable PDF page: %s", e)
            page_text = ""
        if page_text.strip():
            page_texts.append(page_text.strip())

    text = "\n\n".join(page_texts).strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from the PDF. If this is a scanned PDF, use OCR first.",
        )
    return text


redis_client = _init_redis()
load_database()

@app.post("/set-preference")
async def set_preference(req: SetPreferenceRequest):
    level = req.level.lower()
    if level not in ["beginner", "normal", "expert"]:
        level = "normal"
    user_preferences[req.user_id] = level
    memory = get_user_memory(req.user_id)
    memory["preferred_mode"] = level
    logger.info("User preference updated | user_id: %s | preferred_mode: %s", req.user_id, level)
    return {"status": "success", "user_id": req.user_id, "level": level}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info("%s %s - %d - Completed in %.4fs", request.method, request.url.path, response.status_code, process_time)
    return response


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text for ingestion cannot be empty.")

    source_type = normalize_source_type(req.source_type)
    logger.info("Ingesting source type=%s source_name=%s", source_type, req.source_name)
    chunks_added = store_ingested_text(req.text, source_type, req.source_name)
    return {"status": "success", "chunks_added": chunks_added}


@app.get("/sources", response_model=SourceCatalogResponse)
async def sources_endpoint():
    return {"sources": list_sources()}


@app.post("/ingest-source", response_model=IngestResponse)
async def ingest_source_endpoint(
    source_type: str = Form(...),
    source_name: str = Form("unknown"),
    source_text: str = Form(""),
    source_url: str = Form(""),
    file: UploadFile | None = File(None),
):
    normalized_type = normalize_source_type(source_type)
    normalized_name = source_name.strip() or "unknown"

    if normalized_type == "pdf":
        if file is None:
            raise HTTPException(status_code=400, detail="Upload a PDF file to ingest this source.")
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
        text = extract_pdf_text(pdf_bytes)
        if normalized_name == "unknown" and file.filename:
            normalized_name = os.path.splitext(file.filename)[0]
    elif normalized_type == "youtube":
        if not source_url.strip():
            raise HTTPException(status_code=400, detail="Provide a YouTube URL to ingest this source.")
        text = extract_youtube_transcript(source_url)
        if normalized_name == "unknown":
            normalized_name = extract_youtube_video_id(source_url) or "youtube source"
    else:
        if not source_text.strip():
            raise HTTPException(status_code=400, detail="Source text cannot be empty for this source type.")
        text = source_text

    logger.info("Ingesting source type=%s source_name=%s", normalized_type, normalized_name)
    chunks_added = store_ingested_text(text, normalized_type, normalized_name)
    return {"status": "success", "chunks_added": chunks_added}


@app.post("/ingest-batch", response_model=IngestResponse)
async def ingest_batch_endpoint(req: BatchIngestRequest):
    if not req.sources:
        raise HTTPException(status_code=400, detail="No sources provided for batch ingestion.")

    total_chunks = 0
    total_added = 0
    for source in req.sources:
        if not source.content or not source.content.strip():
            logger.warning("Skipping empty source %s (%s)", source.source_name, source.type)
            continue

        logger.info("Batch ingesting source type=%s source_name=%s", source.type, source.source_name)
        chunks = chunk_text(source.content, chunk_size=200)
        logger.info("Source chunk count=%d | type=%s | source_name=%s", len(chunks), source.type, source.source_name)

        total_chunks += len(chunks)
        for chunk in chunks:
            try:
                emb = get_embedding(chunk)
                if emb:
                    added = add_to_database(
                        chunk,
                        emb,
                        source_type=source.type,
                        source_name=source.source_name,
                        persist=False,
                    )
                    if added:
                        total_added += 1
            except Exception as e:
                logger.error("Error generating embedding for chunk from %s: %s", source.source_name, e)

    if total_added == 0:
        raise HTTPException(status_code=500, detail="Batch ingestion failed. Ensure valid source content and embeddings.")

    save_database()

    logger.info("Batch ingestion complete | total_sources=%d | total_chunks=%d | chunks_added=%d", len(req.sources), total_chunks, total_added)
    return {"status": "success", "chunks_added": total_added}


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    request_id = uuid.uuid4().hex
    explanation_mode, mode_source = resolve_explanation_mode(req.mode, req.user_id, question)
    assistant_mode, interaction_source = resolve_interaction_mode(req.assistant_mode, req.user_id, question)
    memory = get_user_memory(req.user_id)
    memory["queries"].append(question)
    memory["preferred_mode"] = explanation_mode
    memory["preferred_interaction"] = assistant_mode
    if len(memory["queries"]) > 20:
        memory["queries"] = memory["queries"][-20:]

    decision_path = {
        "mode_source": mode_source,
        "interaction_source": interaction_source,
        "preferred_mode": memory["preferred_mode"],
        "preferred_interaction": memory["preferred_interaction"],
    }

    logger.info(json.dumps({
        "request_id": request_id,
        "user_id": req.user_id,
        "step": "start_request",
        "explanation_mode": explanation_mode,
        "assistant_mode": assistant_mode,
        "decision_path": decision_path,
        "query": question,
        "timestamp": time.time(),
    }))

    cache_key = f"rag_response:{question.lower()}:{explanation_mode}:{assistant_mode}"

    if is_vague_query(question):
        clarification = {
            "understanding": "The query is too broad or vague.",
            "key_points": ["Identify the specific problem or topic you want help with."],
            "explanation": "Please provide a clearer question so I can guide you step-by-step.",
            "real_world_example": "For example, instead of 'AI?', ask 'How does a neural network classify images?'.",
            "next_steps": ["Clarify the topic or problem", "Tell me if you need teaching or practical assistance"],
            "sources": [{"text": "", "source": "general", "source_name": "clarification"}],
            "meta": {"total_time": 0.0, "retrieval_time": 0.0, "reasoning_time": 0.0},
        }
        set_cached_response(cache_key, clarification)
        return clarification

    cache_key = f"rag_response:{question.lower()}:{explanation_mode}:{assistant_mode}"
    cached_result = get_cached_response(cache_key)
    if cached_result:
        return cached_result

    total_start = time.time()
    query_time = 0.0
    retrieval_time = 0.0
    reasoning_time = 0.0
    query_data = {"question": question, "refined_query": question}
    retrieved_chunks: list[dict[str, str]] = []
    output_payload: dict[str, object] = {
        "understanding": "",
        "key_points": [],
        "explanation": "",
        "real_world_example": "",
        "next_steps": [],
    }

    # Query Agent
    try:
        start = time.time()
        query_data = query_agent(question)
        query_time = time.time() - start
        log_step(request_id, req.user_id, "query_agent", query_time)
    except Exception as e:
        query_time = time.time() - start
        log_error(request_id, req.user_id, "query_agent", e, {"query": question})
        query_data = {"question": question, "refined_query": question}

    # Retrieval Agent
    try:
        start = time.time()
        retrieved_chunks = retrieval_agent(query_data["refined_query"], top_n=3)
        retrieval_time = time.time() - start
        log_step(request_id, req.user_id, "retrieval_agent", retrieval_time)
    except Exception as e:
        retrieval_time = time.time() - start
        log_error(request_id, req.user_id, "retrieval_agent", e, {"refined_query": query_data["refined_query"]})
        retrieved_chunks = []

    if not retrieved_chunks:
        output_payload = {
            "understanding": "This appears to be a general concept with no direct matching source.",
            "key_points": ["I could not find a matching source for your exact request", "I will explain the idea using general knowledge"],
            "explanation": "I am giving a fallback explanation based on common knowledge rather than a specific document.",
            "real_world_example": "Think of this as a high-level summary when the exact source is unavailable.",
            "next_steps": ["Provide a more specific query", "Upload related content if available"],
        }
        retrieved_chunks = [{"text": "", "source": "general", "source_name": "general knowledge"}]
    else:
        try:
            start = time.time()
            output_payload = reasoning_agent(
                query_data["refined_query"],
                retrieved_chunks,
                level=explanation_mode,
                assistant_mode=assistant_mode,
            )
            reasoning_time = time.time() - start
            log_step(request_id, req.user_id, "reasoning_agent", reasoning_time, {"assistant_mode": assistant_mode})
        except Exception as e:
            reasoning_time = 0.0
            log_error(request_id, req.user_id, "reasoning_agent", e, {"refined_query": query_data["refined_query"], "sources": len(retrieved_chunks)})
            output_payload = {
                "understanding": "I encountered an error while generating the response.",
                "key_points": [],
                "explanation": "Please try again or simplify the question.",
                "real_world_example": "",
                "next_steps": ["Retry with a simpler question", "Check the source data and try again"],
            }

    total_time = time.time() - total_start
    result = {
        **output_payload,
        "sources": retrieved_chunks,
        "meta": {
            "total_time": round(total_time, 4),
            "retrieval_time": round(retrieval_time, 4),
            "reasoning_time": round(reasoning_time, 4),
        },
    }

    set_cached_response(cache_key, result)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
