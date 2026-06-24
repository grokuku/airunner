"""Constructeur de commande llama.cpp.

Transforme la sortie du moteur de règles en commande shell exécutable.
"""

from typing import Optional

from app.core import config as app_config


def build_command(
    model_path: str,
    params: dict,
    prompt: Optional[str] = None,
) -> str:
    """Construit la commande llama-cli à partir des paramètres.

    Args:
        model_path: Chemin vers le fichier GGUF
        params: Dictionnaire de paramètres (sortie du moteur de règles)
        prompt: Prompt optionnel à passer directement

    Returns:
        Commande shell complète (avec continuation \\n)
    """
    binary = app_config.config.llamacpp.binary_path
    parts = [binary]
    parts.append(f"  -m {_quote(model_path)}")

    # GPU layers
    ngl = params.get("ngl")
    if ngl is not None:
        parts.append(f"  -ngl {ngl}")

    # Override tensor (spécifique aux MoE)
    override_tensor = params.get("override_tensor", [])
    for ot in override_tensor:
        parts.append(f'  --override-tensor "{ot}"')

    # Cache KV
    cache_k = params.get("cache_type_k")
    if cache_k:
        parts.append(f"  --cache-type-k {cache_k}")
        parts.append(f"  --cache-type-v {cache_k}")

    # Contexte
    ctx = params.get("ctx_size")
    if ctx:
        parts.append(f"  --ctx-size {ctx}")

    # Threads
    threads = params.get("threads")
    if threads:
        parts.append(f"  --threads {threads}")
    tbatch = params.get("threads_batch")
    if tbatch:
        parts.append(f"  --threads-batch {tbatch}")

    # Batch
    ubatch = params.get("ubatch_size")
    if ubatch:
        parts.append(f"  --ubatch-size {ubatch}")
    batch = params.get("batch_size")
    if batch:
        parts.append(f"  --batch-size {batch}")

    # Flash attention
    if params.get("flash_attn"):
        parts.append("  --flash-attn")

    # No KV offload
    if params.get("no_kv_offload"):
        parts.append("  --no-kv-offload")

    # IO-uring (automatique si disponible, on l'ajoute seulement si demandé)
    if params.get("io_uring"):
        parts.append("  --io-uring")

    # Temperature
    temp = params.get("temp", 0.7)
    parts.append(f"  --temp {temp}")

    # Prompt
    if prompt:
        parts.append(f"  --prompt {_quote(prompt)}")

    # Concaténer avec retours à la ligne explicites
    return " \\\n".join(parts) + "\n"


def build_chat_command(
    model_path: str,
    params: dict,
    messages: Optional[list[dict]] = None,
) -> str:
    """Construit la commande llama-cli en mode chat.

    Les messages sont passés via --prompt avec le template Jinja2
    ou directement en format chat si supporté.
    """
    binary = app_config.config.llamacpp.binary_path
    parts = [binary]
    parts.append(f"  -m {_quote(model_path)}")

    ngl = params.get("ngl")
    if ngl is not None:
        parts.append(f"  -ngl {ngl}")

    override_tensor = params.get("override_tensor", [])
    for ot in override_tensor:
        parts.append(f'  --override-tensor "{ot}"')

    cache_k = params.get("cache_type_k")
    if cache_k:
        parts.append(f"  --cache-type-k {cache_k}")
        parts.append(f"  --cache-type-v {cache_k}")

    ctx = params.get("ctx_size")
    if ctx:
        parts.append(f"  --ctx-size {ctx}")

    threads = params.get("threads")
    if threads:
        parts.append(f"  --threads {threads}")
    tbatch = params.get("threads_batch")
    if tbatch:
        parts.append(f"  --threads-batch {tbatch}")

    ubatch = params.get("ubatch_size")
    if ubatch:
        parts.append(f"  --ubatch-size {ubatch}")
    batch = params.get("batch_size")
    if batch:
        parts.append(f"  --batch-size {batch}")

    if params.get("flash_attn"):
        parts.append("  --flash-attn")

    if params.get("no_kv_offload"):
        parts.append("  --no-kv-offload")

    temp = params.get("temp", 0.7)
    parts.append(f"  --temp {temp}")

    # Chat mode + Jinja template (support des templates personnalisés GGUF)
    # Pas de --interactive : on veut une réponse unique, pas un prompt interactif
    if messages:
        parts.append("  --jinja")  # Supporte les templates personnalisés dans les GGUF

    # Si on a des messages, convertir en prompt
    if messages:
        # Construction simple d'un prompt à partir des messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}\n")
        prompt_parts.append("<|assistant|>\n")
        combined = "".join(prompt_parts)
        parts.append(f"  --prompt {_quote(combined)}")

    return " \\\n".join(parts) + "\n"


def _quote(s: str) -> str:
    """Quote une chaîne pour le shell, si nécessaire."""
    if " " in s or "(" in s or ")" in s or "*" in s:
        return f'"{s}"'
    return s
