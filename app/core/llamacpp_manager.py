"""Gestionnaire de binaires llama.cpp.

Vérifie la version installée, la dernière version disponible,
et télécharge les mises à jour depuis hybridgroup/llama-cpp-builder.

Le binaire est téléchargé par l'application, pas intégré dans l'image Docker.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import httpx

from app.core import config as app_config

logger = logging.getLogger("ai-runner")

# URL de l'API de version (fournie par ai-dock/llama.cpp-cuda)
VERSION_API_URL = "https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/latest"

# Pattern de l'URL de téléchargement
# Format: llama.cpp-b{version}-cuda-{cuda_version}-{arch}.tar.gz
# Les fichiers sont extraits dans un sous-dossier cuda-{cuda_version}/
DOWNLOAD_URL_TEMPLATE = (
    "https://github.com/ai-dock/llama.cpp-cuda/releases/download/"
    "{version}/llama.cpp-{version}-cuda-12.8-amd64.tar.gz"
)

# Sous-dossier où sont extraits les binaires dans le tar.gz
BIN_SUBDIR = "cuda-12.8"

# Nom du binaire principal (identique à llama-cli dans ce build)
CLI_BINARY_NAME = "llama-cli"


async def get_installed_version() -> Optional[str]:
    """Retourne la version de llama.cpp actuellement installée.

    Lit d'abord le fichier VERSION (écrit au moment du téléchargement).
    Si absent, essaie `llama-cli --version`.
    Retourne None si le binaire n'est pas trouvé.
    """
    binary = _find_llama_cli()
    if not binary:
        return None

    # 1. Lire le fichier VERSION (priorité — le binaire répond "version: 0")
    version_file = Path(binary).parent / "VERSION"
    try:
        if version_file.is_file():
            version = version_file.read_text().strip()
            if version:
                return version
    except Exception:
        pass

    # 2. Essayer --version (fallback)
    try:
        proc = await asyncio.create_subprocess_exec(
            binary, "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        output = stdout.decode().strip() or stderr.decode().strip()

        # Le build hybridgroup répond "version: 0 (unknown)"
        # On cherche un vrai tag bXXXX (4+ chiffres)
        match = re.search(r"b(\d{4,})", output)
        if match:
            return f"b{match.group(1)}"
    except (FileNotFoundError, asyncio.TimeoutError, Exception):
        pass

    # 3. Le binaire existe mais version inconnue
    return "installé (version inconnue)"


async def get_latest_version() -> Optional[dict]:
    """Vérifie la dernière version disponible en ligne.

    Returns:
        Dict avec {tag, html_url, published_at} ou None si erreur
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(VERSION_API_URL, timeout=10)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"Impossible de vérifier la dernière version: {e}")
        return None


async def check_for_update() -> dict:
    """Compare la version installée avec la dernière disponible.

    Returns:
        {
            "installed": "b9765" | None,
            "latest": "b9765" | None,
            "update_available": bool,
            "download_url": str | None,
            "error": str | None
        }
    """
    installed = await get_installed_version()
    latest_data = await get_latest_version()

    result = {
        "installed": installed,
        "latest": None,
        "update_available": False,
        "download_url": None,
        "latest_html_url": None,
        "error": None,
    }

    if latest_data:
        latest_tag = latest_data.get("tag_name", "")
        result["latest"] = latest_tag
        result["latest_html_url"] = latest_data.get("html_url")

        # Construire l'URL de téléchargement (ai-dock)
        if latest_tag:
            result["download_url"] = DOWNLOAD_URL_TEMPLATE.format(
                version=latest_tag
            )

        # Vérifier si une mise à jour est disponible
        if installed and latest_tag:
            installed_num = _extract_version_number(installed)
            latest_num = _extract_version_number(latest_tag)
            if (installed_num is not None and latest_num is not None
                    and latest_num > installed_num):
                result["update_available"] = True

    return result


async def download_and_install(version_tag: str, download_url: str) -> dict:
    """Télécharge et installe une version spécifique de llama.cpp.

    Le binaire est placé dans le dossier configuré par llamacpp.binary_path.

    Returns:
        Dict avec {success, version, path, error}
    """
    result = {"success": False, "version": version_tag, "path": "", "error": None}

    # Déterminer le dossier de destination (parent du binary_path)
    binary_path = Path(app_config.config.llamacpp.binary_path)
    dest_dir = binary_path.parent
    os.makedirs(dest_dir, exist_ok=True)

    # Sous-dossier dans le tar.gz
    subdir = BIN_SUBDIR

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # 1. Télécharger le build CUDA depuis ai-dock
            # Contient tout : llama-server, llama-cli, .so CUDA
            logger.info(f"Téléchargement de llama.cpp CUDA {version_tag}...")
            resp = await client.get(download_url, timeout=300)
            resp.raise_for_status()

            with tempfile.TemporaryDirectory() as tmpdir:
                tar_path = os.path.join(tmpdir, "llama.tar.gz")
                with open(tar_path, "wb") as f:
                    f.write(resp.content)

                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=tmpdir)

                extracted_dir = os.path.join(tmpdir, subdir)
                if not os.path.isdir(extracted_dir):
                    raise FileNotFoundError(
                        f"Sous-dossier '{subdir}' introuvable dans l'archive"
                    )

                # Copier tous les fichiers vers la destination
                for item in os.listdir(extracted_dir):
                    src = os.path.join(extracted_dir, item)
                    dst = os.path.join(dest_dir, item)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                        os.chmod(dst, 0o755)
                    elif os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)

            # Ai-dock inclut déjà tous les binaires (llama-server, llama-cli, etc.)

            # Stocker la version dans un fichier (le binaire --version retourne "0")
            version_file = os.path.join(dest_dir, "VERSION")
            with open(version_file, "w") as f:
                f.write(version_tag)
            logger.info(f"Version {version_tag} écrite dans {version_file}")

            result["success"] = True
            result["path"] = str(dest_dir)
            logger.info(f"llama.cpp {version_tag} installé dans {dest_dir}")

    except Exception as e:
        logger.error(f"Échec du téléchargement/installation: {e}")
        result["error"] = str(e)

    return result


def _find_llama_cli() -> Optional[str]:
    """Trouve le chemin du binaire llama-cli.

    Cherche dans cet ordre :
    1. Chemin configuré (binary_path)
    2. llama-completion dans le même dossier
    3. Cherche dans des dossiers courants
    4. PATH
    """
    paths_to_check = []

    # 1. Chemin configuré
    binary = app_config.config.llamacpp.binary_path
    paths_to_check.append(binary)

    # 2. llama-completion dans le même dossier
    parent = Path(binary).parent
    completion = parent / CLI_BINARY_NAME
    paths_to_check.append(str(completion))

    # 3. Dossiers courants
    for base in ["/app/llama-bin", "/app/bin", "/usr/local/bin"]:
        paths_to_check.append(os.path.join(base, "llama-cli"))
        paths_to_check.append(os.path.join(base, CLI_BINARY_NAME))

    # 4. Tous les fichiers exécutables dans les dossiers courants
    for base in ["/app/llama-bin", "/app/bin"]:
        if os.path.isdir(base):
            for f in os.listdir(base):
                fp = os.path.join(base, f)
                if os.path.isfile(fp) and os.access(fp, os.X_OK):
                    if "llama" in f and "cli" in f:
                        return fp

    for p in paths_to_check:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p

    # 5. PATH
    which = shutil.which("llama-cli")
    if which:
        return which

    return None


def _extract_version_number(tag: str) -> Optional[int]:
    """Extrait le nombre d'une version tag (ex: 'b9765' → 9765)."""
    match = re.search(r"(\d+)", tag)
    if match:
        return int(match.group(1))
    return None
