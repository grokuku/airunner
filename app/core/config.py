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
    cors_origins: list[str] = ["http://localhost:8311"]


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

# Chemin du fichier de configuration utilisé au chargement (pour persistance)
_config_path: Optional[Path] = None


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
            global _config_path
            _config_path = config_path
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


def save_config(cfg: AppConfig, path: Optional[str] = None) -> Path:
    """Persiste la configuration dans un fichier YAML.

    Utilise le chemin explicite passé en paramètre, sinon celui mémorisé
    lors du dernier `load_config()`, sinon `config/config.yaml` dans le
    répertoire de l'app.
    """
    global _config_path

    if path:
        target = Path(path)
    elif _config_path:
        target = _config_path
    else:
        app_dir = Path(__file__).resolve().parent.parent.parent
        target = app_dir / "config" / "config.yaml"

    target.parent.mkdir(parents=True, exist_ok=True)

    # On dump le modèle en mode dict natif pour un YAML lisible
    data = cfg.model_dump()
    with open(target, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    _config_path = target
    return target
