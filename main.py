"""AI Runner — FastAPI application entry point.

Lancement :
    python main.py
    uvicorn main:app --host 0.0.0.0 --port 8311
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import v1_system, v1_models, v1_chat, v1_comfy, v1_llamacpp, v1_openai, ui
from app.core.config import AppConfig, config, load_config

logger = logging.getLogger("ai-runner")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application."""
    loaded = load_config()
    # Mettre à jour le module-level config
    import app.core.config as cfg
    cfg.config = loaded

    # Créer les dossiers de stockage si nécessaires
    import os
    for d in [loaded.storage.models_dir, loaded.storage.data_dir, loaded.storage.presets_dir]:
        os.makedirs(d, exist_ok=True)

    logger.info(
        f"AI Runner démarré — port {loaded.server.port}, "
        f"modèles: {loaded.storage.models_dir}"
    )
    yield
    logger.info("AI Runner arrêté")


app = FastAPI(
    title="AI Runner",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — permet au frontend (même depuis un autre port) d'accéder à l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routeurs API
app.include_router(v1_system.router, prefix="/api/v1")
app.include_router(v1_models.router, prefix="/api/v1")
app.include_router(v1_chat.router, prefix="/api/v1")
app.include_router(v1_comfy.router, prefix="/api/v1")
app.include_router(v1_llamacpp.router, prefix="/api/v1")
app.include_router(v1_openai.router)  # Pas de prefix : /v1/...

# Routes de l'interface web
app.include_router(ui.router)

# Servir les fichiers statiques (si le dossier existe)
import os as os_mod
static_dir = os_mod.path.join(os_mod.path.dirname(__file__), "app", "web", "static")
if os_mod.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    uvicorn.run(
        "main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=True,
        log_level="info",
    )
