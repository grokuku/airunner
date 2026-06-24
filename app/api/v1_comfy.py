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
    has_run = rm.current_run and rm.current_run.status in (
        RunStatus.RUNNING, RunStatus.LOADING
    )
    return {
        "model_loaded": has_run,
        "model_name": rm.current_run.model_id if has_run else None,
        "vram_total_gb": system.gpu[0].vram_total_gb if system.gpu else 0,
        "vram_free_gb": system.gpu[0].vram_free_gb if system.gpu else 0,
        "mode": system.mode,
    }


@router.post("/comfyui/prepare")
async def comfyui_prepare():
    """Prépare un modèle en mémoire pour une utilisation rapide.

    Appelé par le custom node "AI Runner Load Model" dans ComfyUI.
    Le modèle est chargé dans la VRAM et reste prêt à inférer.
    """
    return {
        "status": "ready",
        "message": "Utilisez POST /api/v1/chat pour lancer une inférence. "
                   "Le modèle sera chargé automatiquement.",
    }


@router.post("/comfyui/release")
async def comfyui_release():
    """Libère la VRAM en déchargeant le modèle actuel.

    Appelé par le custom node "AI Runner Unload Model" dans ComfyUI.
    Garantit que la VRAM est libre pour d'autres usages (Stable Diffusion, etc.).
    """
    rm = get_run_manager()
    if rm.current_run and rm.current_run.status in (
        RunStatus.RUNNING, RunStatus.LOADING
    ):
        await rm.stop()
        return {"status": "released", "vram_freed": True}
    return {"status": "released", "vram_freed": False}
