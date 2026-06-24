"""API endpoints pour les informations système.

GET /api/v1/status  → État du serveur (GPU, RAM, CPU, runs)
"""

import logging
from fastapi import APIRouter

from app.core.system_detector import detect
from app.models import SystemStatus

logger = logging.getLogger("ai-runner")
router = APIRouter(tags=["system"])


@router.get("/status", response_model=SystemStatus)
async def get_status():
    """Retourne l'état actuel du système : GPU, RAM, CPU.

    Utile pour savoir si un modèle peut être chargé,
    et pour les intégrations (ComfyUI) qui ont besoin de connaître
    la VRAM disponible.
    """
    status = await detect()
    return status
