"""API endpoints spécifiques à l'intégration ComfyUI.

Ces endpoints sont conçus pour être appelés par des custom nodes ComfyUI.
Ils sont volontairement simples et prédictibles.

POST /api/v1/comfyui/status   → État actuel (modèle chargé, VRAM dispo)
POST /api/v1/comfyui/prepare  → Charge un modèle pour usage rapide
POST /api/v1/comfyui/release  → Unload forcé, libère la VRAM
"""

from fastapi import APIRouter

from app.core.run_manager import RunStatus, get_run_manager
from app.core.system_detector import detect

router = APIRouter(tags=["comfyui"])


@router.post("/comfyui/status")
async def comfyui_status():
    """Retourne l'état actuel côté ComfyUI.

    Utile pour les custom nodes qui ont besoin de savoir
    si un modèle est chargé et combien de VRAM est disponible.
    """
    system = await detect()
    rm = get_run_manager()
    has_run = rm.server and rm.server.status in (
        RunStatus.RUNNING, RunStatus.LOADING
    )
    return {
        "model_loaded": has_run,
        "model_name": rm.server.model_id if has_run else None,
        "gpu_count": len(system.gpu) if system.gpu else 0,
        "vram_total_gb": sum(g.vram_total_gb for g in system.gpu) if system.gpu else 0,
        "vram_free_gb": sum(g.vram_free_gb for g in system.gpu) if system.gpu else 0,
        "vram_per_gpu": [
            {"index": g.index, "name": g.name, "total_gb": g.vram_total_gb, "free_gb": g.vram_free_gb}
            for g in system.gpu
        ] if system.gpu else [],
        "mode": system.mode,
    }


@router.post("/comfyui/prepare")
async def comfyui_prepare():
    """Indique comment charger un modèle.

    Le chargement est automatique au premier appel de POST /api/v1/chat.
    Cet endpoint est un point d'entrée informatif pour les custom nodes
    ComfyUI — il ne charge pas de modèle lui-même.
    """
    return {
        "status": "ready",
        "message": "Le modèle sera chargé automatiquement lors du premier appel à "
                   "POST /api/v1/chat. Utilisez cet endpoint pour lancer l'inférence.",
    }


@router.post("/comfyui/release")
async def comfyui_release():
    """Libère la VRAM en déchargeant le modèle actuel.

    Appelé par le custom node "AI Runner Unload Model" dans ComfyUI.
    Garantit que la VRAM est libre pour d'autres usages (Stable Diffusion, etc.).
    """
    rm = get_run_manager()
    if rm.server and rm.server.status in (
        RunStatus.RUNNING, RunStatus.LOADING
    ):
        await rm.stop()
        return {"status": "released", "vram_freed": True}
    return {"status": "released", "vram_freed": False}
