"""Configuration loader for AI Runner.

Charge la configuration depuis config.yaml et les variables d'environnement.
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8311
    auth_token: str = ""


class StorageConfig(BaseModel):
    models_dir: str = "/app/models"
    data_dir: str = "/app/data"
    presets_dir: str = "/app/presets"


class HuggingFaceConfig(BaseModel):
    token: str = ""


class LlamacppConfig(BaseModel):
    binary_path: str = "/app/llama-bin/llama-cli"
    default_temp: float = 0.7
    default_ctx_size: int = 8192


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    storage: StorageConfig = StorageConfig()
    huggingface: HuggingFaceConfig = HuggingFaceConfig()
    llamacpp: LlamacppConfig = LlamacppConfig()


# Instance globale de configuration (initialisée au démarrage de l'app)
config: AppConfig = AppConfig()


def load_config(path: Optional[str] = None) -> AppConfig:
    """Charge la configuration depuis un fichier YAML.

    Cherche dans cet ordre :
    1. Le chemin explicite passé en paramètre
    2. La variable d'environnement CONFIG_PATH
    3. config/config.yaml dans le répertoire de l'app
    4. config/config.example.yaml (fallback)
    """
    search_paths = []

    if path:
        search_paths.append(Path(path))

    env_path = os.environ.get("CONFIG_PATH")
    if env_path:
        search_paths.append(Path(env_path))

    # Par rapport au répertoire de l'application
    app_dir = Path(__file__).resolve().parent.parent.parent
    search_paths.append(app_dir / "config" / "config.yaml")
    search_paths.append(app_dir / "config" / "config.example.yaml")

    config_data = {}

    for config_path in search_paths:
        if config_path and config_path.exists():
            with open(config_path) as f:
                config_data = yaml.safe_load(f) or {}
            break

    # Surcharge par variables d'environnement
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        if "huggingface" not in config_data:
            config_data["huggingface"] = {}
        config_data["huggingface"]["token"] = env_token

    return AppConfig(**config_data)
