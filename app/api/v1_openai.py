"""API endpoint compatible OpenAI.

POST /v1/chat/completions  → Format OpenAI standard
GET  /v1/models            → Liste des modèles disponibles

Permet d'utiliser n'importe quel outil compatible OpenAI
(SillyTavern, Continue.dev, Open WebUI, etc.) avec AI Runner.
"""

import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core import config as app_config
from app.core.run_manager import RunStatus, get_run_manager

logger = logging.getLogger("ai-runner")
router = APIRouter(tags=["openai"])


@router.get("/v1/models")
async def openai_list_models():
    """Liste les modèles disponibles (format OpenAI)."""
    import os
    models_dir = app_config.config.storage.models_dir
    models = []

    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".gguf"):
                model_id = f.replace(".gguf", "")
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": int(os.path.getmtime(os.path.join(models_dir, f))),
                    "owned_by": "ai-runner",
                })

    return {
        "object": "list",
        "data": models,
    }


@router.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """Endpoint compatible OpenAI.

    Format d'entrée standard :
    {
        "model": "gemma4-coding-Q2_K",
        "messages": [{"role": "user", "content": "Bonjour"}],
        "stream": true,
        "max_tokens": 512,
        "temperature": 0.7
    }

    Format de sortie (stream = true) : SSE avec data: [chunk]
    Format de sortie (stream = false) : JSON OpenAI standard
    """
    body = await request.json()

    model_id = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)

    if not model_id:
        raise HTTPException(status_code=400, detail="Field 'model' is required")
    if not messages:
        raise HTTPException(status_code=400, detail="Field 'messages' is required")

    # Convertir les messages OpenAI en format interne
    # Le format OpenAI utilise role: system/user/assistant
    # Notre format interne est le même
    internal_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Gérer les messages multimodaux (images, etc.)
            # Pour l'instant, on extrait juste le texte
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = "\n".join(text_parts)
        internal_messages.append({"role": role, "content": content})

    # Paramètres d'inférence
    params = {
        "temp": temperature,
        "max_tokens": max_tokens,
        "ctx_size": body.get("max_context", 8192),
    }

    # Trouver le fichier modèle
    import os as os_mod
    models_dir = app_config.config.storage.models_dir
    filepath = os_mod.path.join(app_config.config.storage.models_dir, f"{model_id}.gguf")
    if not os_mod.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    # Démarrer llama-server avec le modèle
    rm = get_run_manager()
    state = await rm.start_server(model_id, filepath, params)

    if state.status == RunStatus.ERROR:
        raise HTTPException(status_code=500, detail=state.error_message)

    # ID unique pour la réponse
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Mode streaming
    if stream:
        return StreamingResponse(
            _openai_stream(rm, internal_messages, params, completion_id, created, model_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Mode non-stream : accumuler
    full_content = ""
    async for event_str in rm.chat(internal_messages, params):
        if event_str.startswith("data: "):
            try:
                event = json.loads(event_str[6:])
                if event.get("type") == "token":
                    full_content += event.get("text", "")
                elif event.get("type") == "done":
                    break
            except (json.JSONDecodeError, KeyError):
                continue

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": full_content.strip(),
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": state.tokens_generated if state else 0,
            "total_tokens": 0,
        },
    }


async def _openai_stream(rm, messages, params, completion_id, created, model_id):
    """Génère un flux SSE au format OpenAI via llama-server."""
    async for event_str in rm.chat(messages, params):
        if event_str.startswith("data: "):
            try:
                event = json.loads(event_str[6:])
                event_type = event.get("type", "")

                if event_type == "token":
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": event.get("text", "")},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                elif event_type == "done":
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                elif event_type == "error":
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "error",
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            except (json.JSONDecodeError, KeyError):
                continue
