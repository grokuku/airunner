"""Client pour l'API HuggingFace.

Recherche de modèles, téléchargement de fichiers GGUF,
et sondage à distance des headers GGUF.

Tout est dynamique — aucune base de modèles statique.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx

from app.core import config as app_config

logger = logging.getLogger("ai-runner")

# URL de base de l'API HuggingFace
HF_API_BASE = "https://huggingface.co"
HF_API_MODELS = f"{HF_API_BASE}/api/models"


def _get_headers() -> dict:
    """Retourne les headers HTTP avec le token si configuré."""
    headers = {"User-Agent": "ai-runner/0.1.0"}
    token = app_config.config.huggingface.token
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def search_models(
    query: str,
    page: int = 1,
    limit: int = 20,
) -> list[dict]:
    """Recherche des modèles sur HuggingFace.

    Args:
        query: Terme de recherche
        page: Numéro de page
        limit: Nombre de résultats par page

    Returns:
        Liste de dictionnaires avec les infos des modèles
    """
    params = {
        "search": query,
        "sort": "downloads",
        "direction": -1,
        "limit": limit,
        "full": "false",
        "config": "false",
    }

    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            resp = await client.get(
                HF_API_MODELS,
                params=params,
                headers=_get_headers(),
                timeout=15,
            )
            resp.raise_for_status()
            models = resp.json()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            logger.warning(f"Erreur recherche HF: {e}")
            return []

    # Filtrer pour ne garder que les modèles avec des fichiers GGUF
    results = []
    for model in models:
        # Vérifier la présence de fichiers GGUF
        sibling_names = [s.get("rfilename", "") for s in model.get("siblings", [])]
        has_gguf = any(n.endswith(".gguf") for n in sibling_names)

        if not has_gguf:
            continue

        # Lister les fichiers GGUF
        gguf_files = [
            {
                "name": name,
                "size": _get_sibling_size(model.get("siblings", []), name),
            }
            for name in sibling_names
            if name.endswith(".gguf")
        ]

        results.append({
            "repo_id": model.get("id", ""),
            "name": model.get("id", "").split("/")[-1],
            "downloads": model.get("downloads", 0),
            "likes": model.get("likes", 0),
            "files": gguf_files,
        })

    return results


def _get_sibling_size(siblings: list[dict], filename: str) -> int:
    """Récupère la taille d'un fichier dans la liste des siblings."""
    for sib in siblings:
        if sib.get("rfilename") == filename:
            return sib.get("size", 0)
    return 0


async def get_model_files(repo_id: str) -> list[dict]:
    """Récupère la liste des fichiers d'un modèle HuggingFace.

    Args:
        repo_id: Identifiant du repo (ex: "bartowski/Qwen3.6-35B-A3B-GGUF")

    Returns:
        Liste des fichiers GGUF avec leurs tailles
    """
    url = f"{HF_API_MODELS}/{repo_id}"

    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            resp = await client.get(url, headers=_get_headers(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            logger.warning(f"Erreur récupération fichiers {repo_id}: {e}")
            return []

    siblings = data.get("siblings", [])
    files = []
    for s in siblings:
        if s.get("rfilename", "").endswith(".gguf"):
            size = s.get("size", 0)
            # Pour les repos Xet (stockage distribué HF), la taille peut être
            # dans le champ "size" ou absente. On fait un HEAD pour la découvrir
            # uniquement si nécessaire (affiché côté client).
            files.append({
                "name": s.get("rfilename", ""),
                "size": size if size else 0,
            })
    return files


async def probe_remote_gguf(repo_id: str, filename: str) -> Optional[dict]:
    """Sonde un fichier GGUF distant en téléchargeant uniquement son header.

    Télécharge les premiers Mo du fichier (jusqu'à 20 Mo pour couvrir
    les matrices d'importance des IQ quants) pour analyser l'architecture
    du modèle sans avoir à le télécharger entièrement.

    Args:
        repo_id: Identifiant du repo HuggingFace
        filename: Nom du fichier GGUF

    Returns:
        Métadonnées parsées du header, ou None si erreur
    """
    from app.core.gguf_parser import parse_gguf_header_from_bytes

    url = f"{HF_API_BASE}/{repo_id}/resolve/main/{filename}"

    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            req_headers = _get_headers()
            # 20 Mo pour couvrir les headers (IQ quants ont matrice d'importance)
            req_headers["Range"] = "bytes=0-20971520"
            resp = await client.get(url, headers=req_headers, timeout=30)
            resp.raise_for_status()
            data = resp.content
        except Exception as e:
            logger.warning(f"Erreur probe distant {filename}: {e}")
            return None

    try:
        metadata = parse_gguf_header_from_bytes(data)
        # Ajouter des infos de contexte
        metadata["_repo_id"] = repo_id
        metadata["_filename"] = filename
        metadata["_remote_size"] = int(
            resp.headers.get("Content-Range", "").split("/")[-1] or 0
        )
        return metadata
    except ValueError as e:
        logger.warning(f"Erreur parsing header distant {filename}: {e}")
        return None


async def download_gguf(
    repo_id: str,
    filename: str,
    destination: str,
) -> AsyncGenerator[dict, None]:
    """Télécharge un fichier GGUF depuis HuggingFace avec progression.

    Génère des événements de progression :
    - {"type": "start", "total_bytes": N}
    - {"type": "progress", "downloaded": N, "speed": X, "eta": Y}
    - {"type": "done", "path": "/app/models/..."}
    - {"type": "error", "message": "..."}

    Args:
        repo_id: Identifiant du repo HF
        filename: Nom du fichier GGUF
        destination: Chemin de destination complet
    """
    url = f"{HF_API_BASE}/{repo_id}/resolve/main/{filename}"

    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            # Vérifier la taille totale avec une requête HEAD
            head_resp = await client.head(url, headers=_get_headers(), timeout=15)
            total_bytes = int(head_resp.headers.get("Content-Length", 0))

            yield {"type": "start", "total_bytes": total_bytes, "filename": filename}

            # Vérifier si le fichier existe déjà partiellement (reprise)
            start_byte = 0
            if os.path.exists(destination):
                start_byte = os.path.getsize(destination)
                if start_byte >= total_bytes:
                    yield {"type": "done", "path": destination}
                    return

            # Téléchargement avec reprise possible
            dl_headers = _get_headers()
            if start_byte > 0:
                dl_headers["Range"] = f"bytes={start_byte}-"

            async with client.stream(
                "GET", url, headers=dl_headers, timeout=300
            ) as resp:
                resp.raise_for_status()

                downloaded = start_byte
                start_time = asyncio.get_event_loop().time()

                with open(destination, "ab" if start_byte > 0 else "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Émettre progression toutes les ~0.5s
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if elapsed > 0.5:
                            speed = downloaded / elapsed  # bytes/s
                            eta = (total_bytes - downloaded) / speed if speed > 0 else 0
                            yield {
                                "type": "progress",
                                "downloaded": downloaded,
                                "total_bytes": total_bytes,
                                "speed_gbps": round(speed / 1_000_000, 2),
                                "eta_s": round(eta),
                                "percent": round(downloaded / total_bytes * 100, 1),
                            }
                            start_time = asyncio.get_event_loop().time()

            yield {"type": "done", "path": destination}

        except httpx.HTTPStatusError as e:
            yield {"type": "error", "message": f"Erreur HTTP {e.response.status_code}: {e.response.text[:200]}"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
