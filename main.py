"""AI Runner — FastAPI application entry point.

Lancement :
    python main.py
    uvicorn main:app --host 0.0.0.0 --port 8311
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import v1_system, v1_models, v1_chat, v1_comfy, v1_llamacpp, v1_openai, v1_benchmark, ui
from app.core.auth import verify_token
from app.core.config import AppConfig, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

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
# Les origines autorisées sont lues depuis la config (app/core/config.py).
# On charge la config ici (avant l'ajout du middleware) pour que les
# origines reflètent config.yaml dès le démarrage.
_loaded = load_config()
import app.core.config as cfg_module
cfg_module.config = _loaded
app.add_middleware(
    CORSMiddleware,
    allow_origins=_loaded.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routeurs API (avec authentification)
_auth_deps = [Depends(verify_token)]

app.include_router(v1_system.router, prefix="/api/v1", dependencies=_auth_deps)
app.include_router(v1_models.router, prefix="/api/v1", dependencies=_auth_deps)
app.include_router(v1_chat.router, prefix="/api/v1", dependencies=_auth_deps)
app.include_router(v1_comfy.router, prefix="/api/v1", dependencies=_auth_deps)
app.include_router(v1_llamacpp.router, prefix="/api/v1", dependencies=_auth_deps)
app.include_router(v1_benchmark.router, prefix="/api/v1", dependencies=_auth_deps)
app.include_router(v1_openai.router, dependencies=_auth_deps)  # Pas de prefix : /v1/... mais auth activée

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
