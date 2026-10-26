import os
import time
import json
import asyncio
from fastapi import FastAPI, Request, Response, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
from app.model_loader import model_loader
from app.cost_tracker import tracker
from app import llm, auth
from app.auth import current_user, optional_user
from app.courses import store, worker, BUILTIN_ID, VIDEO_EXTS
from app.config import OLLAMA_LLM_MODEL, MAX_UPLOAD_MB, WHISPER_MODEL
from app.config import (
    SLA_LATENCY_MS,
    AWS_LAMBDA_COST_PER_MS,
    AWS_ECS_TASK_COST_PER_SEC
)

app = FastAPI(
    title="Sigma RAG Model Serving API",
    description="Production-grade serving pipeline for RAG AI Teaching assistant with SLA and cost tracking.",
    version="1.0.0"
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Pydantic Schemas
class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of strings to embed.")

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    backend: str

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string.")
    top_k: int = Field(5, description="Number of top similarities to return.")

class SearchResultItem(BaseModel):
    number: str
    title: str
    start: float
    end: float
    text: str
    chunk_id: int
    similarity_score: float
    video_id: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    backend: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="Question to ask the AI teaching assistant.")
    course_id: str = Field(BUILTIN_ID, description="Course to search; defaults to the built-in Sigma course.")

class CourseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)

class SignupRequest(BaseModel):
    name: str = Field(..., max_length=80)
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=256)

class LoginRequest(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=256)

class QueryResponse(BaseModel):
    query: str
    answer: str
    context_chunks: List[SearchResultItem]
    backend: str
    llm: str

# Reject cross-site writes and anonymous uploads before any request body is read
@app.middleware("http")
async def guard_api_writes(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        if origin and origin.split("://", 1)[-1] != request.headers.get("host"):
            return JSONResponse({"detail": "Cross-site request blocked"}, status_code=403)
        if request.url.path.endswith("/videos") and not optional_user(request):
            return JSONResponse({"detail": "Please log in."}, status_code=401)
    return await call_next(request)

# Middleware for Latency & Cost tracking in HTTP headers
@app.middleware("http")
async def add_latency_and_cost_headers(request: Request, call_next):
    # Exclude /health and /stats from SLA tracking to avoid skewing stats
    is_inference_path = request.url.path in ["/embed", "/search", "/query"]
    
    start_time = time.perf_counter()
    response = await call_next(request)
    end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000.0
    
    if is_inference_path:
        tracker.record_inference(latency_ms)
        
        # Calculate instant costs for headers
        billed_duration_ms = max(1.0, round(latency_ms))
        lambda_cost = billed_duration_ms * AWS_LAMBDA_COST_PER_MS
        
        stats = tracker.get_stats()
        total_reqs = stats["total_inferences"]
        uptime = stats["uptime_seconds"]
        
        ecs_total_cost = uptime * AWS_ECS_TASK_COST_PER_SEC
        ecs_cost_per_req = (ecs_total_cost / total_reqs) if total_reqs > 0 else 0.0
        
        # Append headers to the response
        response.headers["X-Inference-Latency-MS"] = f"{latency_ms:.2f}"
        response.headers["X-Inference-Cost-USD-Lambda"] = f"{lambda_cost:.9f}"
        response.headers["X-Inference-Cost-USD-ECS"] = f"{ecs_cost_per_req:.9f}"
        response.headers["X-SLA-Compliance"] = "PASSED" if latency_ms <= SLA_LATENCY_MS else "FAILED"
        
    return response

# Routes - Using async def for highly-performant asynchronous execution on the event loop
@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Simple health check endpoint for AWS Load Balancers."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/stats", tags=["Monitoring"])
async def get_performance_stats():
    """Returns SLA compliance, latencies (avg, p50, p95, p99), and AWS costs (Lambda vs. ECS)."""
    return tracker.get_stats()

@app.post("/embed", response_model=EmbedResponse, tags=["Inference"])
async def get_embeddings(payload: EmbedRequest):
    """Generates embedding vectors for a list of texts."""
    embeddings = model_loader.embed_text(payload.texts).tolist()
    return EmbedResponse(
        embeddings=embeddings,
        backend=model_loader.backend
    )

@app.post("/search", response_model=SearchResponse, tags=["Inference"])
async def search_similar_chunks(payload: SearchRequest):
    """Performs cosine similarity search on embeddings.joblib for a search query."""
    results_list = model_loader.search_similarity(payload.query, top_k=payload.top_k)
    results = [
        SearchResultItem(
            number=str(item['number']),
            title=str(item['title']),
            start=float(item['start']),
            end=float(item['end']),
            text=str(item['text']),
            chunk_id=int(item['chunk_id']),
            similarity_score=float(item['similarity_score'])
        )
        for item in results_list
    ]
    return SearchResponse(
        query=payload.query,
        results=results,
        backend=model_loader.backend
    )

def course_index(course_id: str, user):
    """Search index for a course the user may read: the built-in course or one they own."""
    if course_id != BUILTIN_ID and not (user and store.owned(course_id, user["id"])):
        raise HTTPException(404, "Course not found")
    return store.get_index(course_id)


def owned_course(course_id: str, user: dict) -> dict:
    course = store.owned(course_id, user["id"])
    if not course:
        raise HTTPException(404, "Course not found")
    return course


@app.post("/query", response_model=QueryResponse, tags=["Inference"])
async def query_rag_assistant(payload: QueryRequest, request: Request):
    """Complete RAG endpoint combining vector retrieval and response generation."""
    index = course_index(payload.course_id, optional_user(request))

    def run():
        chunks = index.search_segments(payload.query, top_k=5)
        gen = llm.generate_answer(payload.query, chunks, index, store.course_name(payload.course_id))
        source = next(gen)
        return chunks, source, "".join(gen)

    chunks, source, answer = await asyncio.to_thread(run)
    return QueryResponse(
        query=payload.query,
        answer=answer,
        context_chunks=[SearchResultItem(**c) for c in chunks],
        backend=index.mode,
        llm=source,
    )


@app.post("/api/chat", tags=["Website"])
def chat_stream(payload: QueryRequest, user: dict = Depends(current_user)):
    """Streams newline-delimited JSON: a `sources` event, then `token` events, then `done`."""
    index = course_index(payload.course_id, user)
    course_name = store.course_name(payload.course_id)

    def events():
        if not index.metadata:
            yield json.dumps({"type": "sources", "llm": "none", "sources": []}) + "\n"
            yield json.dumps({"type": "token", "text": "This course has no processed videos yet. Upload a video and wait until it shows **Ready**, then ask away."}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return
        chunks = index.search_segments(payload.query, top_k=5)
        gen = llm.generate_answer(payload.query, chunks, index, course_name)
        source = next(gen)
        relevant = source != "none"
        yield json.dumps({"type": "sources", "llm": source, "sources": chunks if relevant else []}) + "\n"
        for piece in gen:
            yield json.dumps({"type": "token", "text": piece}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/api/status", tags=["Website"])
def status():
    return {
        "retrieval": model_loader.backend,
        "llm": OLLAMA_LLM_MODEL if llm.llm_available() else None,
        "whisper": WHISPER_MODEL,
        "max_upload_mb": MAX_UPLOAD_MB,
        "extensions": sorted(VIDEO_EXTS),
    }


# ---------- Accounts ----------
def start_session(request: Request, user: dict) -> JSONResponse:
    response = JSONResponse(user)
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(
        auth.SESSION_COOKIE, auth.create_session(user["id"]), max_age=auth.SESSION_DAYS * 86400,
        httponly=True, samesite="lax", secure=secure, path="/",
    )
    return response


@app.post("/api/auth/signup", tags=["Accounts"])
def signup(payload: SignupRequest, request: Request):
    try:
        first_user = auth.user_count() == 0
        user = auth.create_user(payload.email, payload.name, payload.password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if first_user:
        store.claim_unowned(user["id"])
    return start_session(request, user)


@app.post("/api/auth/login", tags=["Accounts"])
def login(payload: LoginRequest, request: Request):
    try:
        user = auth.authenticate(payload.email, payload.password, request.client.host if request.client else "")
    except PermissionError as e:
        raise HTTPException(429, str(e))
    except ValueError as e:
        raise HTTPException(401, str(e))
    return start_session(request, user)


@app.post("/api/auth/logout", tags=["Accounts"])
def logout(request: Request):
    token = request.cookies.get(auth.SESSION_COOKIE)
    if token:
        auth.delete_session(token)
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    return response


@app.get("/api/auth/me", tags=["Accounts"])
def me(user: dict = Depends(current_user)):
    return user


# ---------- Courses & uploads (each user sees only their own) ----------
@app.get("/api/courses", tags=["Courses"])
def list_courses(user: dict = Depends(current_user)):
    return store.list_courses(user["id"])


@app.post("/api/courses", tags=["Courses"])
def create_course(payload: CourseCreate, user: dict = Depends(current_user)):
    return store.create_course(payload.name.strip(), user["id"])


@app.get("/api/courses/{course_id}", tags=["Courses"])
def get_course(course_id: str, user: dict = Depends(current_user)):
    if course_id == BUILTIN_ID:
        videos = [dict(v, id=None, status="ready", has_media=False) for v in model_loader.index.list_videos()]
        return {"id": BUILTIN_ID, "name": store.course_name(BUILTIN_ID), "builtin": True, "videos": videos}
    course = owned_course(course_id, user)
    for v in course["videos"]:
        v["has_media"] = True
    return dict(course, builtin=False)


@app.delete("/api/courses/{course_id}", tags=["Courses"])
def delete_course(course_id: str, user: dict = Depends(current_user)):
    owned_course(course_id, user)
    store.delete_course(course_id)
    return {"deleted": course_id}


@app.post("/api/courses/{course_id}/videos", tags=["Courses"])
async def upload_video(course_id: str, file: UploadFile = File(...), translate: bool = Form(True),
                       user: dict = Depends(current_user)):
    owned_course(course_id, user)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(VIDEO_EXTS))}")

    video = store.add_video(course_id, file.filename, translate)
    path = store.media_path(course_id, video)
    limit, written = MAX_UPLOAD_MB * 1024 * 1024, 0
    try:
        with open(path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB")
                out.write(chunk)
    except BaseException:
        store.delete_video(course_id, video["id"])
        raise
    worker.enqueue(course_id, video["id"])
    return store.get_video(course_id, video["id"])


@app.delete("/api/courses/{course_id}/videos/{video_id}", tags=["Courses"])
def delete_video(course_id: str, video_id: str, user: dict = Depends(current_user)):
    owned_course(course_id, user)
    if not store.delete_video(course_id, video_id):
        raise HTTPException(404, "Video not found")
    return {"deleted": video_id}


@app.post("/api/courses/{course_id}/videos/{video_id}/retry", tags=["Courses"])
def retry_video(course_id: str, video_id: str, user: dict = Depends(current_user)):
    owned_course(course_id, user)
    video = store.get_video(course_id, video_id)
    if not video or video["status"] != "failed":
        raise HTTPException(400, "Only failed videos can be retried")
    store.update_video(course_id, video_id, error=None)
    worker.enqueue(course_id, video_id)
    return store.get_video(course_id, video_id)


@app.get("/api/courses/{course_id}/videos/{video_id}/media", tags=["Courses"])
def video_media(course_id: str, video_id: str, user: dict = Depends(current_user)):
    owned_course(course_id, user)
    video = store.get_video(course_id, video_id)
    if not video or not os.path.exists(store.media_path(course_id, video)):
        raise HTTPException(404, "Video not found")
    return FileResponse(store.media_path(course_id, video), filename=video["filename"], content_disposition_type="inline")


@app.on_event("startup")
def start_worker():
    worker.start()
    worker.resume_unfinished()


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
