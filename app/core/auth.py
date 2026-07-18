"""Authentification par token Bearer.

Vérifie l'en-tête ``Authorization: Bearer {token}`` si un ``auth_token``
est configuré dans ``ServerConfig``.  Si le token est vide, l'authentification
est désactivée et toutes les requêtes sont acceptées.
"""

import logging

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import config

logger = logging.getLogger("ai-runner")

# HTTPBearer avec auto_error=False pour ne pas rejeter les requêtes
# sans en-tête Authorization lorsque l'auth est désactivée.
_security_scheme = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Security(_security_scheme),
) -> bool:
    """Dépendance FastAPI de vérification du token d'authentification.

    - Si ``config.server.auth_token`` est vide → authentification désactivée
      (retourne ``True``).
    - Si non vide → vérifie l'en-tête ``Authorization: Bearer {token}`` et
      compare avec ``config.server.auth_token``.
    - Si le token ne matche pas → lève ``HTTPException(401)``.
    """
    expected_token = config.server.auth_token

    # Authentification désactivée : on laisse passer
    if not expected_token:
        return True

    # Aucun en-tête Authorization fourni
    if credentials is None:
        logger.warning("Requête sans en-tête Authorization (auth activée)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing auth token",
        )

    # Vérifier le schéma (doit être "Bearer")
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing auth token",
        )

    # Comparer le token
    if credentials.credentials != expected_token:
        logger.warning("Token d'authentification invalide reçu")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing auth token",
        )

    return True