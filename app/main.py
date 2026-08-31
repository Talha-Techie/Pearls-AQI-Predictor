from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from app.api.routes import router
from app.config.settings import get_settings


settings = get_settings()


app = FastAPI(
    title="Pearls AQI Predictor API",
    description=(
        "Real-time AQI prediction API "
        "with 24h, 48h and 72h forecasts."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


# Streamlit will run on port 8501 later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "service": (
            "Pearls AQI Predictor API"
        ),
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
@app.get("/api/health")
def health() -> dict:
    return {
        "status": "healthy",
        "service": (
            "Pearls AQI Predictor API"
        ),
    }


app.include_router(router)
