"""Benchmark runner for auto-config optimization.

Teste automatiquement différentes configurations pour un modèle
et détermine la plus efficace (vitesse, VRAM, score pondéré).

Compare la VRAM estimée (règles) vs la VRAM réelle mesurée.
Pour le multi-GPU, teste uniquement le nombre minimum de GPUs
nécessaire pour contenir le modèle, plus éventuellement +1 pour
voir si plus de GPUs apporte un gain de vitesse.
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
from app.core.rules_engine import estimate_model_size, estimate_kv_cache_gb, _get_bits_from_quant
from app.models import ModelMeta, SystemStatus

logger = logging.getLogger("ai-runner")

# Prompt de test standard (longueur fixe pour comparabilité)
TEST_PROMPT = "Explain the concept of machine learning in detail, including supervised learning, unsupervised learning, and reinforcement learning. Give examples of each."
TEST_GENERATE_TOKENS = 150
MAX_TIME_PER_CONFIG = 120
VRAM_OVERHEAD_GB = 0.3

_TEST_MESSAGES = [
    {"role": "user", "content": TEST_PROMPT},
]


def _estimate_vram_for_config(
    model_meta: ModelMeta,
    cfg: dict,
    ctx_size: int,
) -> float:
    """Estime la VRAM nécessaire pour une configuration donnée (en Go).

    Utilise les mêmes formules que le moteur de règles pour permettre
    la comparaison estimé vs réel.
    """
    bits = _get_bits_from_quant(model_meta.quant)
    params_b = model_meta.params_b
    if model_meta.is_moe:
        # MoE: on considère que les experts sont offloadés sur CPU
        # Seule l'attention reste sur GPU → ~MOE_ATTENTION_RATIO des params
        params_b = model_meta.params_b * 0.15

    model_gb = estimate_model_size(params_b, bits)
    n_layers = max(model_meta.block_count, 1)
    hidden_size = model_meta.embedding_length or 4096

    # Type de cache KV (q8_0=1.0, q4_0=0.5)
    kv_bits = 1.0 if cfg.get("cache_type_k", "q8_0") == "q8_0" else 0.5
    kv_gb = estimate_kv_cache_gb(ctx_size, hidden_size, n_layers, kv_bits)

    # Offloading
    ngl = cfg.get("ngl", 99)
    if ngl >= n_layers:
        # Full GPU
        total = model_gb + kv_gb + VRAM_OVERHEAD_GB
    else:
        # Offloading partiel
        total = (model_gb / n_layers) * ngl + kv_gb + VRAM_OVERHEAD_GB

    return round(total, 2)


def _find_min_gpus_for_model(
    model_meta: ModelMeta,
    system: SystemStatus,
    ctx_size: int,
    kv_bits: float = 1.0,
) -> int:
    """Trouve le nombre minimum de GPUs nécessaire pour contenir le modèle.

    Trie les GPUs par VRAM libre décroissante et cherche le plus petit
    ensemble dont la somme de VRAM dépasse la taille du modèle.
    """
    bits = _get_bits_from_quant(model_meta.quant)
    model_gb = estimate_model_size(model_meta.params_b, bits)
    n_layers = max(model_meta.block_count, 1)
    hidden_size = model_meta.embedding_length or 4096
    kv_gb = estimate_kv_cache_gb(ctx_size, hidden_size, n_layers, kv_bits)
    needed = model_gb + kv_gb + VRAM_OVERHEAD_GB

    # Trier les GPUs par VRAM libre décroissante
    sorted_gpus = sorted(system.gpu, key=lambda g: g.vram_free_gb, reverse=True)

    cumulative = 0.0
    for i, gpu in enumerate(sorted_gpus):
        cumulative += gpu.vram_free_gb
        if cumulative >= needed:
            return i + 1  # i+1 GPUs suffisent

    # Ne tient même pas sur tous les GPUs
    return len(system.gpu)


def generate_config_grid(
    model_meta: ModelMeta,
    system: SystemStatus,
    ctx_size: int = 8192,
    fixed_cache_type: Optional[str] = None,
    fixed_flash_attn: Optional[bool] = None,
) -> list[dict]:
    """Génère une grille de configurations à tester.

    Args:
        ctx_size: Taille du contexte (KV cache)
        fixed_cache_type: "q8_0", "q4_0", ou None pour tester les deux
        fixed_flash_attn: True, False, ou None pour tester les deux

    fixed_* permet à l'utilisateur de réduire les tests en fixant
    certains paramètres (ex: toujours Q4 pour le cache KV).
    """
    n_layers = max(model_meta.block_count, 1)
    multi_gpu = len(system.gpu) > 1 if system.gpu else False
    half_layers = max(n_layers // 2, 1)

    # Types de cache KV à tester
    cache_types = ["q8_0"] if fixed_cache_type else ["q8_0", "q4_0"]
    if fixed_cache_type == "q4_0":
        cache_types = ["q4_0"]

    base = {
        "temp": 0.7,
        "max_tokens": TEST_GENERATE_TOKENS,
        "no_kv_offload": False,
        "ctx_size": ctx_size,
    }

    def make_config(**overrides) -> dict:
        """Crée une config et calcule estimate_vram_gb."""
        cfg = {**base, **overrides}
        if "cache_type_k" not in cfg:
            cfg["cache_type_k"] = "q8_0"
            cfg["cache_type_v"] = "q8_0"
        if "flash_attn" not in cfg:
            cfg["flash_attn"] = True
        cfg["estimate_vram_gb"] = _estimate_vram_for_config(model_meta, cfg, ctx_size)
        return cfg

    configs = []

    # ── Full GPU avec chaque type de cache ──
    for ct in cache_types:
        label_base = "Full GPU"
        if ct == "q8_0":
            label_ct = "cache Q8"
        else:
            label_ct = "cache Q4"

        # Avec flash attn
        if fixed_flash_attn is None or fixed_flash_attn is True:
            configs.append(make_config(
                ngl=99, cache_type_k=ct, cache_type_v=ct, flash_attn=True,
                label=f"{label_base} • {label_ct}",
            ))

        # Sans flash attn (si pas fixé)
        if fixed_flash_attn is None:
            configs.append(make_config(
                ngl=99, cache_type_k=ct, cache_type_v=ct, flash_attn=False,
                label=f"{label_base} • {label_ct} • no flash",
            ))

    # ── Offloading progressif (single GPU, ngl croissant) ──
    ct = cache_types[0]
    for ratio, label_suffix in [(0.25, "¼ GPU"), (0.5, "½ GPU"), (0.75, "¾ GPU"), (1.0, "Tout GPU")]:
        ngl_val = max(1, int(n_layers * ratio))
        configs.append(make_config(
            ngl=ngl_val, cache_type_k=ct, cache_type_v=ct, flash_attn=True,
            label=f"Offload {ngl_val}/{n_layers} ({label_suffix})",
        ))

    # ── Multi-GPU intelligent ──
    if multi_gpu:
        min_gpus = _find_min_gpus_for_model(model_meta, system, ctx_size)
        sorted_gpus = sorted(system.gpu, key=lambda g: g.vram_free_gb, reverse=True)

        # Config avec le minimum de GPUs
        selected = sorted_gpus[:min_gpus]
        ratios = [g.vram_free_gb for g in selected]
        total_r = sum(ratios)
        ts = ",".join(str(max(1, int(r / total_r * 10))) for r in ratios)
        configs.append(make_config(
            ngl=99, cache_type_k=ct, cache_type_v=ct, flash_attn=True,
            no_kv_offload=True, split_mode="layer", tensor_split=ts, main_gpu=0,
            label=f"Multi-GPU {min_gpus}× (min)",
        ))

        # Config avec +1 GPU pour comparaison
        if min_gpus < len(system.gpu):
            extra = sorted_gpus[:min_gpus + 1]
            extra_r = [g.vram_free_gb for g in extra]
            extra_t = sum(extra_r)
            extra_ts = ",".join(str(max(1, int(r / extra_t * 10))) for r in extra_r)
            configs.append(make_config(
                ngl=99, cache_type_k=ct, cache_type_v=ct, flash_attn=True,
                no_kv_offload=True, split_mode="layer", tensor_split=extra_ts, main_gpu=0,
                label=f"Multi-GPU {min_gpus + 1}× (+1)",
            ))

    # ── CPU only ──
    configs.append(make_config(
        ngl=0, cache_type_k="q8_0", cache_type_v="q8_0", flash_attn=False,
        label="CPU only",
    ))

    return configs


def compute_score(
    tok_s: float,
    vram_gb: float,
    vram_total: float,
    priority: str = "speed",
) -> float:
    """Calcule un score de 0 à 100 pour une configuration."""
    if tok_s <= 0:
        return 0.0

    speed_score = min(tok_s / 100.0 * 100, 100)
    vram_remaining = max(vram_total - vram_gb, 0)
    memory_score = min(vram_remaining / max(vram_total, 1) * 100, 100)

    if priority == "speed":
        score = speed_score * 0.7 + memory_score * 0.3
    else:
        score = speed_score * 0.4 + memory_score * 0.6

    return round(score, 1)


async def run_benchmark(
    model_id: str,
    model_meta: ModelMeta,
    system: SystemStatus,
    priority: str = "speed",
    ctx_size: int = 8192,
    fixed_cache_type: Optional[str] = None,
    fixed_flash_attn: Optional[bool] = None,
) -> AsyncGenerator[dict, None]:
    """Exécute le benchmark complet et yield les événements SSE.

    Génère :
      {"type": "start", "total": N, "model_id": "..."}
      {"type": "progress", "current": K, "total": N, "config": {...}}
      {"type": "result", "config": {...}, "tok_s": X, "vram_gb": Y,
       "estimate_vram_gb": Z, "ram_gb": W, "score": S, "diff_pct": D}
      {"type": "best", ...}
      {"type": "done"}
      {"type": "error", "message": "..."}
    """
    models_dir = app_config.storage.models_dir
    filepath = os.path.join(models_dir, f"{model_id}.gguf")
    if not os.path.isfile(filepath):
        yield {"type": "error", "message": f"Fichier introuvable : {filepath}"}
        return

    configs = generate_config_grid(
        model_meta, system,
        ctx_size=ctx_size,
        fixed_cache_type=fixed_cache_type,
        fixed_flash_attn=fixed_flash_attn,
    )
    total = len(configs)
    yield {"type": "start", "total": total, "model_id": model_id}

    results = []
    rm = get_run_manager()

    for idx, cfg in enumerate(configs):
        # Extraire les métadonnées avant de pop
        label = cfg.pop("label", f"Config {idx + 1}")
        estimate_vram = cfg.pop("estimate_vram_gb", None)

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
                results.append({"label": label, "tok_s": 0, "vram_gb": 0, "ram_gb": 0,
                               "error": state.error_message[:200]})
                yield {"type": "result", "config": {"label": label, **cfg},
                       "tok_s": 0, "vram_gb": 0, "estimate_vram_gb": estimate_vram,
                       "ram_gb": 0, "diff_pct": 0, "error": state.error_message[:200]}
                continue

            await asyncio.sleep(1)

            vram_peak = 0.0
            ram_used = 0.0
            token_count = 0
            start_time = time.time()

            async for event_str in rm.chat(_TEST_MESSAGES, cfg):
                if event_str.startswith("data: "):
                    try:
                        event = json.loads(event_str[6:])
                        if event.get("type") == "token":
                            token_count += 1
                            if rm.server:
                                vram_peak = max(vram_peak, rm.server.vram_used_gb)
                                ram_used = max(ram_used, rm.server.ram_used_gb)
                        elif event.get("type") == "error":
                            logger.warning(f"Erreur test {label}: {event.get('message')}")
                    except (json.JSONDecodeError, KeyError):
                        continue

            elapsed = time.time() - start_time
            tok_s = round(token_count / max(elapsed, 0.1), 1) if token_count > 0 else 0.0

            if rm.server:
                vram_peak = max(vram_peak, rm.server.vram_used_gb)
                ram_used = max(ram_used, rm.server.ram_used_gb)

            # Écart estimé vs réel
            diff_pct = 0
            if estimate_vram and estimate_vram > 0 and vram_peak > 0:
                diff_pct = round((vram_peak - estimate_vram) / estimate_vram * 100, 1)

            vram_total = system.gpu[0].vram_total_gb if system.gpu else 0
            score = compute_score(tok_s, vram_peak, vram_total, priority)

            results.append({"label": label, "tok_s": tok_s, "vram_gb": vram_peak,
                           "ram_gb": ram_used})

            yield {"type": "result", "config": {"label": label, **cfg},
                   "tok_s": tok_s, "vram_gb": vram_peak,
                   "estimate_vram_gb": estimate_vram, "ram_gb": ram_used,
                   "score": score, "diff_pct": diff_pct}

        except Exception as e:
            logger.error(f"Erreur benchmark {label}: {e}")
            yield {"type": "result", "config": {"label": label, **cfg},
                   "tok_s": 0, "vram_gb": 0, "estimate_vram_gb": estimate_vram,
                   "ram_gb": 0, "diff_pct": 0, "error": str(e)[:200]}

    if rm.is_running():
        await rm.stop()

    best = max(results, key=lambda r: r.get("tok_s", 0))
    if best.get("tok_s", 0) > 0:
        yield {"type": "best", "label": best["label"],
               "tok_s": best["tok_s"],
               "vram_gb": best.get("vram_gb", 0),
               "ram_gb": best.get("ram_gb", 0)}

    yield {"type": "done"}
