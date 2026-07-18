"""Validation de sécurité pour les paramètres de chemin.

Fonctions utilitaires pour empêcher les attaques par path traversal
dans les endpoints API qui construisent des chemins fichiers à partir
d'entrées utilisateur.
"""

import logging
import os
import re
from pathlib import Path

from fastapi import HTTPException

logger = logging.getLogger("ai-runner")

# Caractères de contrôle et autres séquences suspectes
_SUSPICIOUS_PATTERN = re.compile(r"[\x00-\x1f\x7f]|\x00")


def validate_path_param(value: str) -> str:
    """Valide un paramètre simple (identifiant) utilisé dans un chemin fichier.

    Rejette si la valeur contient ``..``, ``/``, ``\\`` ou des caractères
    de contrôle (null bytes, etc.) qui pourraient permettre un path traversal.

    :param value: La valeur du paramètre à valider.
    :returns: La valeur validée si elle est sûre.
    :raises HTTPException: 400 si un path traversal est détecté.
    """
    if not value:
        raise HTTPException(
            status_code=400,
            detail="Invalid parameter: path traversal detected",
        )

    # Détection des séquences de traversal et séparateurs de chemin
    if ".." in value or "/" in value or "\\" in value:
        logger.warning("Path traversal détecté dans le paramètre: %r", value)
        raise HTTPException(
            status_code=400,
            detail="Invalid parameter: path traversal detected",
        )

    # Détection de caractères suspects (null bytes, contrôle chars)
    if _SUSPICIOUS_PATTERN.search(value):
        logger.warning("Caractères suspects détectés dans le paramètre: %r", value)
        raise HTTPException(
            status_code=400,
            detail="Invalid parameter: path traversal detected",
        )

    return value


def validate_filepath(base_dir: str, filename: str) -> Path:
    """Construit et valide un chemin fichier dans ``base_dir``.

    Construit ``os.path.join(base_dir, filename)`` puis vérifie avec
    ``os.path.realpath()`` que le chemin final reste bien dans ``base_dir``.

    :param base_dir: Le répertoire de base autorisé.
    :param filename: Le nom du fichier (relatif à ``base_dir``).
    :returns: Le chemin validé (``Path``).
    :raises HTTPException: 400 si le chemin échappe ``base_dir``.
    """
    # Construire le chemin
    full_path = os.path.join(base_dir, filename)

    # Résoudre les chemins réels (élimine les symlinks et ../)
    real_base = os.path.realpath(base_dir)
    real_path = os.path.realpath(full_path)

    # Vérifier que le chemin final est bien dans base_dir
    if not str(real_path).startswith(str(real_base) + os.sep) and real_path != real_base:
        logger.warning(
            "Path traversal détecté: %r échappe %r", filename, base_dir
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid path",
        )

    return Path(real_path)