"""API endpoints for system information.

GET  /api/v1/status  → Server state (GPU, RAM, CPU, runs)
GET  /api/v1/config   → Current server configuration (auth token masked)
PUT  /api/v1/config   → Update server configuration (cors_origins, auth_token)
"""

import logging
from fastapi import APIRouter

import app.core.config as cfg_module
from app.core.config import save_config
from app.core.system_detector import detect
from app.models import SystemStatus, ServerConfigResponse, ServerConfigUpdate

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


def _mask_token(token: str) -> str:
    """Mask the auth token for safe display."""
    if not token:
        return ""
    if len(token) <= 4:
        return "*" * len(token)
    return token[:2] + "*" * (len(token) - 4) + token[-2:]


@router.get("/config", response_model=ServerConfigResponse)
async def get_config():
    """Return the current server configuration.

    The auth token is masked to avoid leaking secrets to the frontend.
    """
    server = cfg_module.config.server
    return ServerConfigResponse(
        host=server.host,
        port=server.port,
        auth_token=_mask_token(server.auth_token),
        cors_origins=server.cors_origins,
    )


@router.put("/config", response_model=ServerConfigResponse)
async def update_config(payload: ServerConfigUpdate):
    """Update server configuration fields.

    Only `cors_origins` and `auth_token` are mutable via this endpoint.
    Changes are persisted to the config.yaml file.
    """
    server = cfg_module.config.server

    if payload.cors_origins is not None:
        server.cors_origins = payload.cors_origins
    if payload.auth_token is not None:
        server.auth_token = payload.auth_token

    # Persist to disk
    try:
        save_config(cfg_module.config)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist config: %s", exc)

    return ServerConfigResponse(
        host=server.host,
        port=server.port,
        auth_token=_mask_token(server.auth_token),
        cors_origins=server.cors_origins,
    )
