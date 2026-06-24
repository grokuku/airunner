"""API endpoints pour la gestion du binaire llama.cpp.

GET  /api/v1/llamacpp/version  → Version installée + dernière dispo
POST /api/v1/llamacpp/update   → Télécharge et installe la dernière version
"""

import logging
import os

from fastapi import APIRouter, HTTPException

from app.core.llamacpp_manager import (
    check_for_update,
    download_and_install,
)

logger = logging.getLogger("ai-runner")
router = APIRouter(tags=["llamacpp"])


@router.get("/llamacpp/version")
async def get_llamacpp_version():
    """Retourne la version installée et la dernière version disponible.

    Utile pour l'interface web (indicateur de version + bouton update).
    """
    result = await check_for_update()
    return result


@router.post("/llamacpp/update")
async def update_llamacpp():
    """Télécharge et installe la dernière version de llama.cpp.

    Vérifie d'abord qu'une mise à jour est disponible,
    puis télécharge depuis hybridgroup/llama-cpp-builder.
    """
    status = await check_for_update()

    if not status.get("download_url") or not status.get("latest"):
        raise HTTPException(
            status_code=400,
            detail="Impossible de déterminer l'URL de téléchargement. "
                   "Vérifiez la connexion réseau."
        )

    # Télécharger et installer
    result = await download_and_install(
        status["latest"], status["download_url"]
    )

    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Échec du téléchargement")
        )

    # Re-vérifier la version après installation
    new_status = await check_for_update()

    return {
        "status": "updated",
        "previous_version": status.get("installed"),
        "current_version": status.get("latest"),
        "path": result.get("path"),
        "update_available": new_status.get("update_available", False),
    }
