"""API endpoints pour le benchmark et l'auto-config.

POST /api/v1/benchmark/auto  → SSE: benchmark automatique de toutes les configs
GET  /api/v1/benchmark       → Retourne les résultats du dernier benchmark
"""

import json
import logging
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.core.benchmark_runner import run_benchmark
from app.core.gguf_parser import parse_gguf_header, metadata_to_model_meta
from app.core import config as app_config
from app.core.system_detector import detect

logger = logging.getLogger("ai-runner")
router = APIRouter(tags=["benchmark"])


# Cache du dernier benchmark (pour GET /benchmark)
_last_benchmark_results: list[dict] = []
_last_benchmark_best: dict = {}


@router.post("/benchmark/auto")
async def benchmark_auto(
    model_id: str = Query(..., description="ID du modèle à tester"),
    priority: str = Query("speed", description="Critère : 'speed' ou 'quality'"),
    ctx_size: int = Query(8192, description="Taille du contexte (KV cache)"),
):
    """Lance un benchmark automatique pour un modèle.

    Teste différentes configurations (ngl, cache, flash_attn, multi-GPU…)
    et mesure le débit (tok/s), la VRAM et la RAM pour chacune.

    Retourne un SSE stream avec progression en temps réel.
    """
    global _last_benchmark_results, _last_benchmark_best

    # Trouver le fichier modèle
    models_dir = app_config.config.storage.models_dir
    filepath = os.path.join(models_dir, f"{model_id}.gguf")
    if not os.path.isfile(filepath):
        # Chercher par correspondance partielle
        if os.path.isdir(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith(".gguf") and model_id.lower() in f.lower():
                    filepath = os.path.join(models_dir, f)
                    model_id = f.replace(".gguf", "")
                    break
            else:
                raise HTTPException(status_code=404, detail=f"Modèle '{model_id}' introuvable")

    # Parser les métadonnées du modèle
    try:
        metadata = parse_gguf_header(filepath)
        metadata["_filepath"] = filepath
        model_meta = metadata_to_model_meta(
            metadata, model_id=os.path.basename(filepath).replace(".gguf", "")
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Erreur parsing GGUF: {e}")

    # Détection système
    system = await detect()

    async def event_stream():
        global _last_benchmark_results, _last_benchmark_best
        results = []
        best = {}

        async for event in run_benchmark(model_id, model_meta, system, priority, ctx_size=ctx_size):
            yield f"data: {json.dumps(event)}\n\n"

            # Collecter pour le cache GET
            if event.get("type") == "result":
                results.append(event)
            elif event.get("type") == "best":
                best = event

        _last_benchmark_results = results
        _last_benchmark_best = best

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/benchmark")
async def get_benchmark_results():
    """Retourne les résultats du dernier benchmark."""
    return {
        "results": _last_benchmark_results,
        "best": _last_benchmark_best,
    }


@router.post("/benchmark/save-preset")
async def save_benchmark_preset(
    model_id: str = Query(...),
    label: str = Query("optimized"),
):
    """Sauvegarde la meilleure config du dernier benchmark comme preset YAML."""
    global _last_benchmark_best

    if not _last_benchmark_best:
        raise HTTPException(status_code=400, detail="Aucun benchmark à sauvegarder")

    preset = {
        "model_id": model_id,
        "label": label,
        "params": _last_benchmark_best,
        "source": "auto-benchmark",
    }

    # Écrire le fichier preset
    presets_dir = app_config.config.storage.presets_dir
    os.makedirs(presets_dir, exist_ok=True)
    preset_path = os.path.join(presets_dir, f"{model_id}-{label}.json")

    with open(preset_path, "w") as f:
        json.dump(preset, f, indent=2)

    logger.info(f"Preset sauvegardé : {preset_path}")

    return {
        "status": "saved",
        "path": preset_path,
        "preset": preset,
    }
