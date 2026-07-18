"""API endpoints pour la configuration et l'inférence.

POST /api/v1/config/suggest  → Suggestion de configuration optimale
POST /api/v1/chat            → Inférence (stream SSE ou réponse complète)
POST /api/v1/stop            → Arrête le serveur en cours
POST /api/v1/models/unload   → Décharge le modèle (libère VRAM)
GET  /api/v1/history         → Historique des runs
"""

import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.command_builder import build_command
from app.core import config as app_config
from app.core.gguf_parser import metadata_to_model_meta, parse_gguf_header
from app.core.rules_engine import suggest
from app.core.run_manager import RunStatus, get_run_manager
from app.core.security import validate_path_param
from app.core.system_detector import detect
from app.models import ChatRequest, ConfigRequest, ConfigSuggestion

logger = logging.getLogger("ai-runner")
router = APIRouter(tags=["chat"])


def _find_model_file(model_id: str) -> str:
    model_id = validate_path_param(model_id)
    models_dir = app_config.config.storage.models_dir
    filepath = os.path.join(models_dir, f"{model_id}.gguf")
    if os.path.isfile(filepath):
        return filepath
    if not os.path.isdir(models_dir):
        raise HTTPException(status_code=404, detail=f"Dossier des modèles introuvable: {models_dir}")
    for f in os.listdir(models_dir):
        if f.endswith(".gguf") and model_id.lower() in f.lower():
            return os.path.join(models_dir, f)
    raise HTTPException(status_code=404, detail=f"Modèle '{model_id}' introuvable.")


def _get_model_meta(model_id: str):
    filepath = _find_model_file(model_id)
    try:
        metadata = parse_gguf_header(filepath)
        metadata["_filepath"] = filepath
        return metadata_to_model_meta(metadata, model_id=os.path.basename(filepath).replace(".gguf", ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Erreur parsing GGUF: {e}")


@router.post("/config/suggest", response_model=ConfigSuggestion)
async def get_config_suggestion(request: ConfigRequest):
    filepath = _find_model_file(request.model_id)
    model_meta = _get_model_meta(request.model_id)
    system = await detect()
    result = suggest(model_meta, system, request)
    cmd = build_command(filepath, result.params)
    result.command_preview = cmd
    return result


@router.post("/chat")
async def chat(request: ChatRequest):
    """Inférence sur un modèle via llama-server.

    Démarre llama-server si nécessaire, puis proxy la requête.
    """
    filepath = _find_model_file(request.model_id)

    # Obtenir la config suggérée
    try:
        model_meta = _get_model_meta(request.model_id)
        system = await detect()
        from app.models import ConfigRequest as CR
        cfg_req = CR(model_id=request.model_id, ctx_size=request.params.get("ctx_size", 8192),
                     temp=request.params.get("temp", 0.7))
        suggestion = suggest(model_meta, system, cfg_req)
        params = suggestion.params
        params.update(request.params)
    except Exception as e:
        logger.warning(f"Impossible de suggérer une config: {e}")
        params = {"ngl": 99, "override_tensor": [], "cache_type_k": "q8_0",
                  "ctx_size": 8192, "threads": 4, "flash_attn": True,
                  "no_kv_offload": False, "temp": 0.7}

    # Démarrer llama-server (réutilise si même modèle déjà chargé)
    rm = get_run_manager()
    state = await rm.start_server(request.model_id, filepath, params)

    if state.status == RunStatus.ERROR:
        raise HTTPException(status_code=500, detail=state.error_message)

    # Convertir les messages Pydantic en dicts
    messages_dict = [{"role": m.role, "content": m.content} for m in (request.messages or [])]

    # Mode stream (SSE)
    if request.stream:
        return StreamingResponse(
            rm.chat(messages_dict, params),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Mode non-stream : accumuler
    accumulated = ""
    async for event_str in rm.chat(messages_dict, params):
        if event_str.startswith("data: "):
            try:
                event = json.loads(event_str[6:])
                if event.get("type") == "token":
                    accumulated += event.get("text", "")
                elif event.get("type") == "done":
                    break
                elif event.get("type") == "error":
                    raise HTTPException(status_code=500, detail=event.get("message"))
            except (json.JSONDecodeError, KeyError):
                continue

    return {
        "content": accumulated,
        "tokens": rm.server.tokens_generated if rm.server else 0,
        "speed": rm.server.speed_tokens_per_sec if rm.server else 0,
        "elapsed_s": 0,
    }


@router.post("/stop")
async def stop_run():
    rm = get_run_manager()
    if rm.is_running():
        await rm.stop()
        return {"status": "stopped"}
    return {"status": "no_active_run"}


@router.post("/models/unload")
async def unload_model():
    rm = get_run_manager()
    if rm.is_running():
        await rm.stop()
        return {"status": "unloaded", "vram_freed": True}
    return {"status": "no_model_loaded"}


@router.get("/history")
async def get_history():
    try:
        from app.core.run_manager import get_history as get_history_db
        runs = await get_history_db()
        return {"runs": runs}
    except Exception as e:
        logger.warning(f"Historique non disponible: {e}")
        return {"runs": []}