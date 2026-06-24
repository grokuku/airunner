"""API endpoints pour la gestion des modèles.

GET    /api/v1/models              → Liste des modèles locaux
POST   /api/v1/models/scan         → Force un re-scan
GET    /api/v1/models/{model_id}   → Détails d'un modèle
DELETE /api/v1/models/{model_id}   → Supprime un modèle
GET    /api/v1/models/hf-search    → Recherche HuggingFace
POST   /api/v1/models/hf-download  → Téléchargement depuis HF

Note : l'analyse GGUF est purement structurelle — pas de base de modèles
connus. Tout est extrait du fichier GGUF en temps réel.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.gguf_parser import parse_gguf_header, metadata_to_model_meta
from app.core import config as app_config
from app.core.run_manager import get_run_manager
from app.models import HfDownloadRequest, ModelMeta

logger = logging.getLogger("ai-runner")
router = APIRouter(tags=["models"])


def _get_models_dir() -> str:
    """Retourne le dossier des modèles depuis la configuration."""
    return app_config.config.storage.models_dir


def _mark_loaded_model(models: list[ModelMeta]) -> list[ModelMeta]:
    """Marque le modèle actuellement chargé comme loaded=True.

    Mute les objets du cache global _models_cache (volontairement).
    Le flag .loaded est éphémère : il reflète l'état au moment de l'appel.
    """
    rm = get_run_manager()
    loaded_id = rm.get_loaded_model_id()
    for m in models:
        m.loaded = loaded_id is not None and m.id == loaded_id
    return models


def _mark_single_model_loaded(model: ModelMeta) -> ModelMeta:
    """Marque un seul modèle comme loaded si c'est le modèle actif."""
    rm = get_run_manager()
    loaded_id = rm.get_loaded_model_id()
    model.loaded = loaded_id is not None and model.id == loaded_id
    return model


def _scan_local_models() -> list[ModelMeta]:
    """Scanne le dossier des modèles et parse les headers GGUF."""
    models_dir = _get_models_dir()
    models: list[ModelMeta] = []

    if not os.path.isdir(models_dir):
        return models

    for filename in os.listdir(models_dir):
        if not filename.endswith(".gguf"):
            continue

        filepath = os.path.join(models_dir, filename)
        if not os.path.isfile(filepath):
            continue

        try:
            metadata = parse_gguf_header(filepath)
            # Ajouter le chemin réel
            metadata["_filepath"] = filepath
            model_id = filename.replace(".gguf", "")
            model = metadata_to_model_meta(metadata, model_id=model_id)
            model.path = filepath
            models.append(model)
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Impossible de parser {filename}: {e}")
            # Créer une entrée minimale
            models.append(ModelMeta(
                id=filename.replace(".gguf", ""),
                path=filepath,
                file_size_gb=round(os.path.getsize(filepath) / 1_000_000_000, 2),
                architecture="inconnue",
                name=filename,
                param_count=0,
            ))

    return models


# Cache des modèles scannés (invalidé par scan)
_models_cache: Optional[list[ModelMeta]] = None


@router.get("/models", response_model=list[ModelMeta])
async def list_models():
    """Liste les modèles GGUF disponibles localement.

    Les métadonnées sont parsées depuis le header GGUF à chaque scan.
    """
    global _models_cache
    if _models_cache is None:
        _models_cache = _scan_local_models()
    return _mark_loaded_model(_models_cache)


@router.post("/models/scan", response_model=list[ModelMeta])
async def scan_models():
    """Force un re-scan complet du dossier des modèles.

    Utile après avoir téléchargé ou supprimé un modèle.
    """
    global _models_cache
    _models_cache = _scan_local_models()
    return _mark_loaded_model(_models_cache)


@router.get("/models/hf-search")
async def hf_search(
    q: str = Query("...", description="Terme de recherche"),
    page: int = Query(1, ge=1),
):
    """Recherche des modèles GGUF sur HuggingFace.

    Retourne les modèles triés par nombre de téléchargements.
    Seuls les modèles avec des fichiers GGUF sont inclus.
    """
    from app.core.huggingface_client import search_models

    results = await search_models(q, page=page)
    return {"results": results, "page": page, "total": len(results)}


@router.post("/models/hf-download")
async def hf_download(request: HfDownloadRequest):
    """Télécharge un fichier GGUF depuis HuggingFace.

    Retourne un StreamingResponse SSE qui émet la progression
    en temps réel (bytes téléchargés, vitesse, ETA).
    """
    from app.core.huggingface_client import download_gguf

    models_dir = _get_models_dir()
    destination = os.path.join(models_dir, request.filename)

    async def event_stream():
        async for event in download_gguf(
            request.repo_id, request.filename, destination
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/models/hf-probe/{repo_id:path}")
async def hf_probe(repo_id: str):
    """Sonde un modèle HuggingFace à distance.

    Télécharge les premiers Mo du GGUF pour analyser
    l'architecture (MoE/dense, nb paramètres, etc.) sans tout télécharger.
    """
    from app.core.huggingface_client import get_model_files, probe_remote_gguf

    files = await get_model_files(repo_id)
    if not files:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun fichier GGUF trouvé pour {repo_id}"
        )

    # Choisir le meilleur fichier pour le probe
    probe_file = None
    for f in files:
        name = f["name"]
        if "-00001-of-" in name or name.startswith("BF16/"):
            continue
        if "Q4_K_M" in name or "Q8_0" in name or "Q5_K_M" in name:
            probe_file = name
            break
    if not probe_file:
        for f in files:
            if "-00001-of-" not in f["name"]:
                probe_file = f["name"]
                break
    if not probe_file:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun fichier GGUF sondable trouvé pour {repo_id}"
        )

    metadata = await probe_remote_gguf(repo_id, probe_file)

    return {
        "repo_id": repo_id,
        "files": files,
        "probe": metadata,
    }


@router.get("/models/{model_id}", response_model=ModelMeta)
async def get_model(model_id: str):
    """Retourne les détails d'un modèle spécifique.

    Le modèle est parsé à la volée depuis son fichier GGUF.
    """
    models_dir = _get_models_dir()
    # Chercher par ID exact
    filepath = os.path.join(models_dir, f"{model_id}.gguf")

    if not os.path.isfile(filepath):
        # Chercher par correspondance partielle
        for f in os.listdir(models_dir):
            if f.endswith(".gguf") and model_id.lower() in f.lower():
                filepath = os.path.join(models_dir, f)
                break
        else:
            raise HTTPException(status_code=404, detail=f"Modèle '{model_id}' introuvable")

    try:
        metadata = parse_gguf_header(filepath)
        metadata["_filepath"] = filepath
        model = metadata_to_model_meta(
            metadata, model_id=os.path.basename(filepath).replace(".gguf", "")
        )
        return _mark_single_model_loaded(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Supprime un modèle GGUF du disque."""
    models_dir = _get_models_dir()
    filepath = os.path.join(models_dir, f"{model_id}.gguf")

    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"Modèle '{model_id}' introuvable")

    os.remove(filepath)
    global _models_cache
    _models_cache = None  # Invalider le cache
    return {"status": "deleted", "model_id": model_id}


@router.post("/models/{model_id}/analyze")
async def analyze_model(model_id: str):
    """Analyse poussée d'un modèle.

    Retourne les métadonnées GGUF complètes, pas seulement le ModelMeta.
    Utile pour le débogage et l'exploration.
    """
    models_dir = _get_models_dir()
    filepath = os.path.join(models_dir, f"{model_id}.gguf")

    if not os.path.isfile(filepath):
        for f in os.listdir(models_dir):
            if f.endswith(".gguf") and model_id.lower() in f.lower():
                filepath = os.path.join(models_dir, f)
                break
        else:
            raise HTTPException(status_code=404, detail=f"Modèle '{model_id}' introuvable")

    try:
        metadata = parse_gguf_header(filepath)
        # Convertir les types non-sérialisables
        return _serialize_metadata(metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _serialize_metadata(metadata: dict) -> dict:
    """Sérialise les métadonnées pour JSON (gère les grands entiers)."""
    result = {}
    for key, value in metadata.items():
        if isinstance(value, int) and value > 2**53:
            result[key] = str(value)
        elif isinstance(value, bytes):
            result[key] = value.hex()
        else:
            result[key] = value
    return result
