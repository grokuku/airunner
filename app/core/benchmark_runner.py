"""Benchmark runner for auto-config optimization.

Teste automatiquement différentes configurations pour un modèle
et détermine la plus efficace (vitesse, VRAM, score pondéré).
"""

import asyncio
import json
import logging
import time
import os
from typing import AsyncGenerator, Optional

from app.core.run_manager import RunStatus, get_run_manager
from app.core.system_detector import detect
from app.core.config import config as app_config
from app.models import ModelMeta, SystemStatus

logger = logging.getLogger("ai-runner")

# Prompt de test standard (longueur fixe pour comparabilité)
TEST_PROMPT = "Explain the concept of machine learning in detail, including supervised learning, unsupervised learning, and reinforcement learning. Give examples of each."

# Tokens à générer pour chaque test
TEST_GENERATE_TOKENS = 150

# Timeout par config (secondes)
MAX_TIME_PER_CONFIG = 120

# Messages de test
_TEST_MESSAGES = [
    {"role": "user", "content": TEST_PROMPT},
]


def generate_config_grid(
    model_meta: ModelMeta,
    system: SystemStatus,
    ctx_size: int = 8192,
) -> list[dict]:
    """Génère une grille de configurations à tester, de la plus prometteuse à la moins.

    ctx_size est inclus dans chaque config pour que le cache KV
    soit dimensionné correctement.
    """
    n_layers = max(model_meta.block_count, 1)
    multi_gpu = len(system.gpu) > 1 if system.gpu else False
    half_layers = max(n_layers // 2, 1)

    base = {
        "temp": 0.7,
        "max_tokens": TEST_GENERATE_TOKENS,
        "no_kv_offload": False,
        "ctx_size": ctx_size,
    }

    configs = []

    # 1. Full GPU, cache Q8, flash on (config recommandée par défaut)
    configs.append({**base, "ngl": 99, "cache_type_k": "q8_0",
                    "cache_type_v": "q8_0", "flash_attn": True,
                    "label": "Full GPU Q8"})

    # 2. Full GPU, cache Q4 (moins de VRAM)
    configs.append({**base, "ngl": 99, "cache_type_k": "q4_0",
                    "cache_type_v": "q4_0", "flash_attn": True,
                    "label": "Full GPU Q4"})

    # 3. Offloading partiel (moitié des couches)
    configs.append({**base, "ngl": half_layers, "cache_type_k": "q8_0",
                    "cache_type_v": "q8_0", "flash_attn": True,
                    "label": f"Offload {half_layers}/{n_layers}"})

    # 4. Sans flash attention (comparaison)
    configs.append({**base, "ngl": 99, "cache_type_k": "q8_0",
                    "cache_type_v": "q8_0", "flash_attn": False,
                    "label": "Full GPU no flash"})

    # 5. Multi-GPU split si applicable
    if multi_gpu:
        # Calculer tensor_split proportionnel à la VRAM libre
        ratios = [g.vram_free_gb for g in system.gpu]
        total_ratio = sum(ratios)
        ts = ",".join(str(max(1, int(r / total_ratio * 10))) for r in ratios)
        configs.append({**base, "ngl": 99, "cache_type_k": "q8_0",
                        "cache_type_v": "q8_0", "flash_attn": True,
                        "no_kv_offload": True, "split_mode": "layer",
                        "tensor_split": ts, "main_gpu": 0,
                        "label": f"Multi-GPU ({len(system.gpu)}×)"})

    # 6. CPU only
    configs.append({**base, "ngl": 0, "cache_type_k": "q8_0",
                    "cache_type_v": "q8_0", "flash_attn": False,
                    "label": "CPU only"})

    # 7. Offload moitié + cache Q4
    configs.append({**base, "ngl": half_layers, "cache_type_k": "q4_0",
                    "cache_type_v": "q4_0", "flash_attn": True,
                    "label": f"Offload {half_layers} Q4"})

    return configs


def compute_score(
    tok_s: float,
    vram_gb: float,
    vram_total: float,
    priority: str = "speed",
) -> float:
    """Calcule un score de 0 à 100 pour une configuration.

    priority="speed"  → pondère plus la vitesse (tok/s)
    priority="quality" → pondère plus la marge VRAM (pour contexte long)
    """
    if tok_s <= 0:
        return 0.0

    # Score de vitesse : normalisé à 100 pour 100 tok/s
    speed_score = min(tok_s / 100.0 * 100, 100)

    # Score mémoire : pourcentage de VRAM restante
    vram_remaining = max(vram_total - vram_gb, 0)
    memory_score = min(vram_remaining / max(vram_total, 1) * 100, 100)

    if priority == "speed":
        # 70% vitesse, 30% mémoire
        score = speed_score * 0.7 + memory_score * 0.3
    else:
        # 40% vitesse, 60% mémoire (priorité à la marge pour grand contexte)
        score = speed_score * 0.4 + memory_score * 0.6

    return round(score, 1)


async def run_benchmark(
    model_id: str,
    model_meta: ModelMeta,
    system: SystemStatus,
    priority: str = "speed",
    ctx_size: int = 8192,
) -> AsyncGenerator[dict, None]:
    """Exécute le benchmark complet et yield les événements SSE.

    Génère :
      {"type": "start", "total": N, "model_id": "..."}
      {"type": "progress", "current": K, "total": N, "config": {...}}
      {"type": "result", "config": {...}, "tok_s": X, "vram_gb": Y, "ram_gb": Z, "score": S}
      {"type": "best", "config": {...}, "score": S, "label": "..."}
      {"type": "done"}
      {"type": "error", "message": "..."}
    """
    models_dir = app_config.storage.models_dir
    filepath = os.path.join(models_dir, f"{model_id}.gguf")
    if not os.path.isfile(filepath):
        yield {"type": "error", "message": f"Fichier introuvable : {filepath}"}
        return

    configs = generate_config_grid(model_meta, system, ctx_size=ctx_size)
    total = len(configs)
    yield {"type": "start", "total": total, "model_id": model_id}

    results = []
    rm = get_run_manager()

    for idx, cfg in enumerate(configs):
        label = cfg.pop("label", f"Config {idx + 1}")

        # Arrêter le serveur précédent s'il tourne
        if rm.is_running():
            await rm.stop()
            await asyncio.sleep(1)

        yield {
            "type": "progress",
            "current": idx + 1,
            "total": total,
            "config": {"label": label, **cfg},
        }

        try:
            state = await rm.start_server(model_id, filepath, cfg)

            if state.status == RunStatus.ERROR:
                results.append({
                    "label": label,
                    "tok_s": 0,
                    "vram_gb": 0,
                    "ram_gb": 0,
                    "error": state.error_message[:200],
                })
                yield {
                    "type": "result",
                    "config": {"label": label, **cfg},
                    "tok_s": 0,
                    "vram_gb": 0,
                    "ram_gb": 0,
                    "error": state.error_message[:200],
                }
                continue

            # Attendre un peu que le serveur stabilise
            await asyncio.sleep(1)

            # Mesurer la VRAM/RAM de base
            vram_peak = 0.0
            ram_used = 0.0
            token_count = 0
            start_time = time.time()

            # Lancer le chat de test
            async for event_str in rm.chat(_TEST_MESSAGES, cfg):
                if event_str.startswith("data: "):
                    try:
                        event = json.loads(event_str[6:])
                        if event.get("type") == "token":
                            token_count += 1
                            # Mise à jour VRAM pic
                            if rm.server:
                                vram_peak = max(vram_peak, rm.server.vram_used_gb)
                                ram_used = max(ram_used, rm.server.ram_used_gb)
                        elif event.get("type") == "error":
                            logger.warning(f"Erreur pendant le test {label}: {event.get('message')}")
                    except (json.JSONDecodeError, KeyError):
                        continue

            elapsed = time.time() - start_time
            tok_s = round(token_count / max(elapsed, 0.1), 1) if token_count > 0 else 0.0

            # Dernière lecture VRAM/RAM
            if rm.server:
                vram_peak = max(vram_peak, rm.server.vram_used_gb)
                ram_used = max(ram_used, rm.server.ram_used_gb)

            # Calcul du score
            vram_total = system.gpu[0].vram_total_gb if system.gpu else 0
            score = compute_score(tok_s, vram_peak, vram_total, priority)

            results.append({
                "label": label,
                "tok_s": tok_s,
                "vram_gb": vram_peak,
                "ram_gb": ram_used,
            })

            yield {
                "type": "result",
                "config": {"label": label, **cfg},
                "tok_s": tok_s,
                "vram_gb": vram_peak,
                "ram_gb": ram_used,
                "score": score,
            }

        except Exception as e:
            logger.error(f"Erreur benchmark {label}: {e}")
            yield {
                "type": "result",
                "config": {"label": label, **cfg},
                "tok_s": 0,
                "vram_gb": 0,
                "ram_gb": 0,
                "error": str(e)[:200],
            }

    # Arrêter le serveur à la fin
    if rm.is_running():
        await rm.stop()

    # Trouver le meilleur score
    best = max(results, key=lambda r: r.get("tok_s", 0))
    if best.get("tok_s", 0) > 0:
        yield {
            "type": "best",
            "label": best["label"],
            "tok_s": best["tok_s"],
            "vram_gb": best.get("vram_gb", 0),
            "ram_gb": best.get("ram_gb", 0),
        }

    yield {"type": "done"}
