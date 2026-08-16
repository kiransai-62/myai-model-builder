import time
from fastapi import FastAPI, HTTPException  # type: ignore[import-not-found]
from fastapi.responses import StreamingResponse  # type: ignore[import-not-found]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path

from .runtime import MyAIRuntime, InferenceRequest, InferenceResponse
from ..core.config import ProjectConfig

class AskRequest(BaseModel):
    query: str = Field(..., description="The question to ask")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(256, ge=1, le=1024, description="Maximum tokens to generate")

class AskResponse(BaseModel):
    allowed: bool
    score: float
    answer: str
    sources: list[str]
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model: str
    knowledge_chunks: int
    uptime_seconds: float

class InfoResponse(BaseModel):
    name: str
    model_id: str
    training_method: str
    gate_threshold: float
    version: str

def create_app(root: Path) -> FastAPI:
    """Create FastAPI app with runtime."""
    app = FastAPI(
        title="MYAI Runtime",
        description="Local-first AI model serving API",
        version="0.4.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Load runtime at startup
    cfg = ProjectConfig.load(root)
    runtime = MyAIRuntime(root)
    runtime.load()
    
    start_time = time.time()

    @app.get("/health", response_model=HealthResponse)
    def health():
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            model=cfg.model_id,
            knowledge_chunks=len(runtime.gate.chunks) if runtime.gate else 0,
            uptime_seconds=time.time() - start_time
        )

    @app.get("/info", response_model=InfoResponse)
    def info():
        """Model and configuration information."""
        return InfoResponse(
            name=cfg.name,
            model_id=cfg.model_id,
            training_method=cfg.training.method.upper(),
            gate_threshold=cfg.gate.threshold,
            version="0.4.0"
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        """Ask a question with Knowledge Gate enforcement."""
        if not runtime.gate:
            raise HTTPException(status_code=503, detail="Runtime not initialized")
        
        inference_req = InferenceRequest(
            query=request.query,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        response = runtime.ask(inference_req)
        
        return AskResponse(
            allowed=response.allowed,
            score=response.score,
            answer=response.answer,
            sources=response.sources,
            latency_ms=response.latency_ms
        )

    @app.post("/ask-stream")
    def ask_stream(request: AskRequest):
        """Stream the answer token-by-token using Server-Sent Events (SSE)."""
        if not runtime.gate:
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        inference_req = InferenceRequest(
            query=request.query,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return StreamingResponse(
            runtime.stream_answer(inference_req),
            media_type="text/event-stream"
        )

    return app