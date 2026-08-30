from __future__ import annotations

import os
import time
import threading
import asyncio
from typing import TYPE_CHECKING, Optional, Any
from pathlib import Path
from collections import defaultdict

if TYPE_CHECKING:
    from fastapi import FastAPI  # type: ignore
    from fastapi.responses import StreamingResponse  # type: ignore
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore

try:
    from fastapi import FastAPI, HTTPException, Depends, Request  # type: ignore[import-not-found]
    from fastapi.responses import StreamingResponse  # type: ignore[import-not-found]
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]
    from fastapi.security import APIKeyHeader  # type: ignore[import-not-found]
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .runtime import MyAIRuntime, InferenceRequest, InferenceResponse
from ..core.config import ProjectConfig

# ── Rate Limiter ──────────────────────────────────────────────────────────────

class _RateLimiter:
    """Simple in-process sliding-window rate limiter keyed by client IP."""

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0):
        self._max = max_requests
        self._window = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            bucket = self._buckets[client_ip]
            # Expire old timestamps
            self._buckets[client_ip] = [t for t in bucket if t > cutoff]
            if len(self._buckets[client_ip]) >= self._max:
                return False
            self._buckets[client_ip].append(now)
        return True


# ── CORS helpers ──────────────────────────────────────────────────────────────

def _resolve_allowed_origins() -> list[str]:
    """
    Return allowed CORS origins from env var MYAI_ALLOWED_ORIGINS, or
    fall back to localhost-only defaults (never open wildcard with credentials).
    """
    env_origins = os.environ.get("MYAI_ALLOWED_ORIGINS", "")
    if env_origins.strip():
        return [o.strip() for o in env_origins.split(",") if o.strip()]
    return [
        "http://localhost",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
    ]


# ── Pydantic models ───────────────────────────────────────────────────────────

if HAS_FASTAPI:
    class AskRequest(BaseModel):  # type: ignore
        query: str = Field(
            ...,
            min_length=1,
            max_length=8192,
            description="The question to ask (1–8192 chars)"
        )
        temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
        max_tokens: int = Field(256, ge=1, le=1024, description="Maximum tokens to generate")

    class AskResponse(BaseModel):  # type: ignore
        allowed: bool
        score: float
        answer: str
        sources: list[str]
        latency_ms: float

    class HealthResponse(BaseModel):  # type: ignore
        status: str          # "ok" | "degraded"
        model: str
        model_loaded: bool
        mode: str            # "inference" | "retrieval_only"
        knowledge_chunks: int
        uptime_seconds: float

    class InfoResponse(BaseModel):  # type: ignore
        name: str
        model_id: str
        training_method: str
        gate_threshold: float
        version: str


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(
    root: Path,
    api_key: Optional[str] = None,
    max_concurrent: int = 2,
    rate_limit_per_min: int = 60,
) -> "FastAPI":
    """
    Create the FastAPI app with:
      - Optional API key authentication (X-API-Key or Authorization: Bearer)
      - Localhost-only CORS by default (configurable via MYAI_ALLOWED_ORIGINS)
      - Concurrency semaphore to protect GPU/CPU resources
      - Per-IP rate limiting (default 60 req/min)
      - Bounded query size (max 8192 chars)
      - Safe fallback — never leaks raw knowledge chunks on model failure
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "Serving dependencies are missing. Please install them with:\n"
            "  pip install 'myai[serve]'\n"
            "or\n"
            "  pip install fastapi uvicorn"
        )

    # Resolve API key: CLI arg > environment variable
    _api_key = api_key or os.environ.get("MYAI_API_KEY", "").strip() or None

    # Concurrency semaphore
    _semaphore = threading.Semaphore(max_concurrent)

    # Rate limiter
    _rate_limiter = _RateLimiter(max_requests=rate_limit_per_min, window_seconds=60.0)

    # CORS origins
    allowed_origins = _resolve_allowed_origins()

    app = FastAPI(
        title="MYAI Runtime",
        description="Local-first AI model serving API",
        version="0.4.0",
        # Disable docs on open networks when auth is enabled (opt-in locally)
        docs_url="/docs" if not _api_key else None,
        redoc_url="/redoc" if not _api_key else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,  # Never allow credentials with broad origins
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    )

    # Load runtime at startup
    cfg = ProjectConfig.load(root)
    runtime = MyAIRuntime(root)
    runtime.load()

    start_time = time.time()

    # ── Auth dependency ───────────────────────────────────────────────────────

    def _verify_api_key(request: Request) -> None:
        """Verify API key if one is configured. Allows unauthenticated access in dev mode."""
        if _api_key is None:
            return  # No key configured → local dev mode, allow all
        # Check X-API-Key header
        key_header = request.headers.get("X-API-Key", "")
        # Check Authorization: Bearer <token>
        auth_header = request.headers.get("Authorization", "")
        bearer = auth_header.removeprefix("Bearer ").strip() if auth_header.startswith("Bearer ") else ""
        if key_header != _api_key and bearer != _api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # ── Rate-limit + concurrency guard ───────────────────────────────────────

    def _guard(request: Request) -> None:
        """Apply per-IP rate limiting and concurrency cap."""
        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please slow down your requests."
            )

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/")
    def root_endpoint():
        """Root endpoint returning service status and available endpoints."""
        return {
            "app": "MYAI Runtime",
            "version": "0.4.0",
            "project": cfg.name,
            "model_id": cfg.model_id,
            "status": "ok" if runtime.model_loaded else "degraded",
            "mode": "inference" if runtime.model_loaded else "retrieval_only",
            "endpoints": {
                "health": "/health",
                "info": "/info",
                "ask": "POST /ask",
                "ask_stream": "POST /ask-stream",
            }
        }

    @app.get("/health", response_model=HealthResponse)
    def health():
        """Health check endpoint. No authentication required."""
        model_loaded = runtime.model_loaded
        return HealthResponse(
            status="ok" if model_loaded else "degraded",
            model=cfg.model_id,
            model_loaded=model_loaded,
            mode="inference" if model_loaded else "retrieval_only",
            knowledge_chunks=len(runtime.gate.chunks) if runtime.gate else 0,
            uptime_seconds=time.time() - start_time,
        )

    @app.get("/info", response_model=InfoResponse)
    def info(request: Request):
        """Model and project information."""
        _verify_api_key(request)
        return InfoResponse(
            name=cfg.name,
            model_id=cfg.model_id,
            training_method=cfg.training.method.upper(),
            gate_threshold=cfg.gate.threshold,
            version="0.4.0",
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: Request, body: AskRequest):
        """Ask a question with Knowledge Gate enforcement."""
        _verify_api_key(request)
        _guard(request)

        if not runtime.gate:
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        acquired = _semaphore.acquire(blocking=False)
        if not acquired:
            raise HTTPException(
                status_code=503,
                detail="Server is busy. Maximum concurrent requests reached. Please retry shortly."
            )
        try:
            inference_req = InferenceRequest(
                query=body.query,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
            response = runtime.ask(inference_req)
        finally:
            _semaphore.release()

        return AskResponse(
            allowed=response.allowed,
            score=response.score,
            answer=response.answer,
            sources=response.sources,
            latency_ms=response.latency_ms,
        )

    @app.post("/ask-stream")
    def ask_stream(request: Request, body: AskRequest):
        """Stream the answer token-by-token using Server-Sent Events (SSE)."""
        _verify_api_key(request)
        _guard(request)

        if not runtime.gate:
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        acquired = _semaphore.acquire(blocking=False)
        if not acquired:
            raise HTTPException(
                status_code=503,
                detail="Server is busy. Maximum concurrent requests reached. Please retry shortly."
            )

        inference_req = InferenceRequest(
            query=body.query,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )

        def _generate():
            try:
                yield from runtime.stream_answer(inference_req)
            finally:
                _semaphore.release()

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
        )

    return app