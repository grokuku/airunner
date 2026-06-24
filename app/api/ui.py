"""Routes pour l'interface Web.

Sert l'application SPA (Single Page Application).
Toutes les routes non-API renvoient index.html.
"""

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(tags=["web"])

# Chemin vers les fichiers statiques
STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
INDEX_HTML = STATIC_DIR / "index.html"


@router.get("/")
async def index():
    """Page principale de l'interface Web."""
    if INDEX_HTML.exists():
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>AI Runner</h1><p>Interface Web en cours de construction (Phase 5).</p>"
    )


@router.get("/app")
async def app_page():
    """Point d'entrée pour l'application SPA."""
    if INDEX_HTML.exists():
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>AI Runner</h1><p>Interface Web en cours de construction (Phase 5).</p>"
    )
