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

from app.core.run_manager import RunStatus, get_run_manager, _get_resource_usage
from app.core.system_detector import detect, get_live_snapshot
from app.core.config import config as app_config
from app.core.rules_engine import estimate_model_size, estimate_kv_cache_gb, _get_bits_from_quant, MOE_ATTENTION_RATIO
from app.models import ModelMeta, SystemStatus

logger = logging.getLogger("ai-runner")

# Prompt de test standard (longueur fixe pour comparabilité)
TEST_PROMPT = "Explain the concept of machine learning in detail, including supervised learning, unsupervised learning, and reinforcement learning. Give examples of each."
TEST_GENERATE_TOKENS = 150
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
        params_b = model_meta.params_b * MOE_ATTENTION_RATIO

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
    force_mtp: bool = False,
    include_cpu_only: bool = True,
) -> list[dict]:
    """Génère une grille de configurations à tester.

    Args:
        ctx_size: Taille du contexte (KV cache). Peut être grand — jusqu'au max natif
            du modèle voire plus. Si le KV cache ne tient plus sur GPU, llama.cpp
            déborde en RAM (spill) : c'est le comportement voulu pour tester le vrai
            max au lieu de s'arrêter à l'estimation VRAM du moteur de règles.
        fixed_cache_type: "q8_0", "q4_0", ou None pour tester les deux
        fixed_flash_attn: True, False, ou None pour tester les deux
        include_cpu_only: Si False, n'ajoute pas la config "CPU only" à la grille.

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
        "batch_size": 2048,
        "ubatch_size": 512,
        "cont_batching": True,
        "no_context_shift": True,
        "jinja": True,
        "parallel": 1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
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

    # ── Variations ciblées (batch, cont-batching, threads-batch) ──
    # Uniquement sur le premier cache_type (q8_0 par défaut) pour ne pas
    # exploser le nombre de configs : on cherche le meilleur réglage.
    variation_ct = cache_types[0]
    if variation_ct == "q8_0":
        try:
            cpu_cores = max(system.cpu.cores, 1)
        except Exception:
            cpu_cores = 8
        configs.append(make_config(
            ngl=99, cache_type_k=variation_ct, cache_type_v=variation_ct,
            flash_attn=True, batch_size=1024,
            label="Full GPU • batch 1024",
        ))
        configs.append(make_config(
            ngl=99, cache_type_k=variation_ct, cache_type_v=variation_ct,
            flash_attn=True, batch_size=4096,
            label="Full GPU • batch 4096",
        ))
        configs.append(make_config(
            ngl=99, cache_type_k=variation_ct, cache_type_v=variation_ct,
            flash_attn=True, cont_batching=False,
            label="Full GPU • no cont-batching",
        ))
        configs.append(make_config(
            ngl=99, cache_type_k=variation_ct, cache_type_v=variation_ct,
            flash_attn=True, threads_batch=cpu_cores * 2,
            label="Full GPU • threads-batch 2x",
        ))

    # ── Offloading progressif (GPU, ngl croissant) ──
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
            split_mode="layer", tensor_split=ts, main_gpu=0,
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
                split_mode="layer", tensor_split=extra_ts, main_gpu=0,
                label=f"Multi-GPU {min_gpus + 1}× (+1)",
            ))

    # ── Configurations spécifiques MoE ──
    # Pour les modèles MoE, on place l'attention sur GPU et les experts
    # (ffn_gate/down/up) sur CPU, comme rules_engine.py le fait déjà.
    if model_meta.is_moe:
        moe_configs_data = [
            {
                "label": "MoE: attn GPU, experts CPU",
                "ngl": 99,
                "override_tensor": [
                    ".*attn.*=CUDA0",
                    ".*ffn_gate.*=CPU",
                    ".*ffn_down.*=CPU",
                    ".*ffn_up.*=CPU",
                ],
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "flash_attn": True,
            },
            {
                "label": "MoE: attn GPU, experts CPU • cache Q4",
                "ngl": 99,
                "override_tensor": [
                    ".*attn.*=CUDA0",
                    ".*ffn_gate.*=CPU",
                    ".*ffn_down.*=CPU",
                    ".*ffn_up.*=CPU",
                ],
                "cache_type_k": "q4_0",
                "cache_type_v": "q4_0",
                "flash_attn": True,
            },
        ]

        # Éviter les doublons : une config est identique si elle a le même
        # ngl, cache_type_k et override_tensor.
        existing_keys = set()
        for c in configs:
            key = (c.get("ngl"), c.get("cache_type_k"), tuple(c.get("override_tensor", [])))
            existing_keys.add(key)

        for moe_cfg in moe_configs_data:
            key = (moe_cfg["ngl"], moe_cfg["cache_type_k"], tuple(moe_cfg["override_tensor"]))
            if key not in existing_keys:
                configs.append(make_config(**moe_cfg))
                existing_keys.add(key)

    # ── Configurations MTP (si le modèle supporte Multi-Token Prediction) ──
    if model_meta.mtp or force_mtp:
        configs.append(make_config(
            ngl=99, cache_type_k=ct, cache_type_v=ct, flash_attn=True,
            mtp=True, spec_draft_n_max=2, parallel=1,
            label="Full GPU • MTP • cache " + ("Q8" if ct == "q8_0" else "Q4"),
        ))
        if ct != "q4_0":
            configs.append(make_config(
                ngl=99, cache_type_k="q4_0", cache_type_v="q4_0", flash_attn=True,
                mtp=True, spec_draft_n_max=2, parallel=1,
                label="Full GPU • MTP • cache Q4",
            ))

    # ── CPU only ──
    if include_cpu_only:
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
    force_mtp: bool = False,
    include_cpu_only: bool = True,
    skip_offload_if_full_gpu: bool = False,
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
        force_mtp=force_mtp,
        include_cpu_only=include_cpu_only,
    )
    total = len(configs)
    yield {"type": "start", "total": total, "model_id": model_id}

    results = []
    rm = get_run_manager()

    # Suivi du succès réel du Full GPU : un Full GPU qui a généré des tokens
    # prouve que le modèle tient entièrement en VRAM. On s'appuie sur les
    # résultats réels, pas sur l'estimation VRAM (trop imprécise).
    full_gpu_ok = False

    for idx, cfg in enumerate(configs):
        # Extraire les métadonnées avant de pop
        label = cfg.pop("label", f"Config {idx + 1}")
        estimate_vram = cfg.pop("estimate_vram_gb", None)

        # Si on demande de sauter l'offload ET que le Full GPU a déjà réussi
        # → ignorer cette config d'offload (décision basée sur les résultats réels)
        if skip_offload_if_full_gpu and full_gpu_ok and label.startswith("Offload"):
            yield {"type": "skipped", "config": {"label": label}, "reason": "Full GPU fonctionne"}
            continue

        # Arrêter le serveur précédent : le port doit être libéré avant
        # de lancer la configuration suivante.
        if rm.server is not None:
            await rm.stop()
            await asyncio.sleep(2)

        yield {
            "type": "progress",
            "current": idx + 1,
            "total": total,
            "config": {"label": label, **cfg},
        }

        try:
            # allow_fallback=False : le benchmark doit tester la config exacte
            # sans que le run_manager ne modifie silencieusement les paramètres.
            state = await rm.start_server(model_id, filepath, cfg, allow_fallback=False)

            if state.status == RunStatus.ERROR:
                results.append({"label": label, "tok_s": 0, "vram_gb": 0, "ram_gb": 0,
                               "error": state.error_message[:200]})
                yield {"type": "result", "config": {"label": label, **cfg},
                       "tok_s": 0, "vram_gb": 0, "estimate_vram_gb": estimate_vram,
                       "ram_gb": 0, "diff_pct": 0, "error": state.error_message[:200]}
                continue

            await asyncio.sleep(1)

            # Mesurer la VRAM/RAM réelle juste après le chargement du modèle
            vram_peak, ram_used = await _get_resource_usage()
            vram_peak = vram_peak or 0.0
            ram_used = ram_used or 0.0
            logger.info(
                f"[{label}] Ressources après chargement — "
                f"VRAM: {vram_peak} GB, RAM: {ram_used} GB"
            )

            token_count = 0
            tok_s = 0
            inference_error: Optional[str] = None
            last_monitor = 0.0

            # Au lieu de calculer tok_s nous-même (token_count / elapsed inclut
            # le pré-fill), on utilise la vitesse reportée par le backend dans
            # les events SSE, qui est calculée depuis le premier token.
            async for event_str in rm.chat(_TEST_MESSAGES, cfg):
                # Monitoring temps réel ~1x par seconde pendant l'inférence
                now = time.monotonic()
                if now - last_monitor >= 1.0:
                    try:
                        snapshot = await get_live_snapshot()
                    except Exception as e:
                        logger.warning(f"[monitor] snapshot échoué: {e}")
                        snapshot = {}
                    yield {
                        "type": "monitor",
                        **snapshot,
                        "config": {
                            "ngl": cfg.get("ngl"),
                            "cache_type_k": cfg.get("cache_type_k"),
                            "no_kv_offload": cfg.get("no_kv_offload"),
                            "split_mode": cfg.get("split_mode"),
                            "tensor_split": cfg.get("tensor_split"),
                            "override_tensor": cfg.get("override_tensor", []),
                            "mtp": cfg.get("mtp", False),
                            "label": label,
                        },
                    }
                    last_monitor = now

                if event_str.startswith("data: "):
                    try:
                        event = json.loads(event_str[6:])
                        if event.get("type") == "token":
                            token_count += 1
                            tok_s = event.get("speed", tok_s)
                            if rm.server:
                                vram_peak = max(vram_peak, rm.server.vram_used_gb)
                                ram_used = max(ram_used, rm.server.ram_used_gb)
                        elif event.get("type") == "done":
                            break
                        elif event.get("type") == "error":
                            inference_error = event.get("message", "Erreur inconnue")
                            logger.warning(
                                f"[{label}] Erreur inférence: {inference_error}"
                            )
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue

            # Si l'inférence a échoué silencieusement, yield un event d'erreur
            if inference_error or token_count == 0:
                if inference_error:
                    err_msg = inference_error
                else:
                    # Échec silencieux : 0 token sans event d'erreur. La cause
                    # détaillée (dernières lignes brutes du stream) est journalisée
                    # par run_manager.chat() en warning. On ajoute aussi les
                    # derniers logs stderr de llama-server pour un détail exploitable.
                    stderr = ""
                    if rm.server is not None:
                        stderr = rm.server.recent_stderr(15)
                    detail = "(cause détaillée dans le warning backend [chat])"
                    if stderr:
                        detail = f"Derniers logs llama-server:\n{stderr}"
                    err_msg = f"Inférence: 0 token généré (échec silencieux) — {detail}"
                logger.error(f"[{label}] {err_msg}")
                results.append({
                    "label": label, "tok_s": 0, "vram_gb": vram_peak,
                    "ram_gb": ram_used, "error": err_msg[:200],
                })
                yield {
                    "type": "result",
                    "config": {"label": label, **cfg},
                    "tok_s": 0,
                    "vram_gb": vram_peak,
                    "estimate_vram_gb": estimate_vram,
                    "ram_gb": ram_used,
                    "diff_pct": 0,
                    "score": 0,
                    "error": err_msg[:200],
                }
                continue

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
                           "ram_gb": ram_used, "score": score})

            yield {"type": "result", "config": {"label": label, **cfg},
                   "tok_s": tok_s, "vram_gb": vram_peak,
                   "estimate_vram_gb": estimate_vram, "ram_gb": ram_used,
                   "score": score, "diff_pct": diff_pct}

            # Un vrai config Full GPU (ngl=99, pas de split, pas de MoE) qui a
            # généré des tokens prouve que le full GPU fonctionne.
            if label.startswith("Full GPU") and tok_s > 0:
                full_gpu_ok = True

        except Exception as e:
            logger.error(f"Erreur benchmark {label}: {e}")
            yield {"type": "result", "config": {"label": label, **cfg},
                   "tok_s": 0, "vram_gb": 0, "estimate_vram_gb": estimate_vram,
                   "ram_gb": 0, "diff_pct": 0, "error": str(e)[:200]}

    # Arrêter le serveur après le dernier test
    if rm.server is not None:
        await rm.stop()
        await asyncio.sleep(2)

    if not results:
        yield {"type": "done"}
        return

    best = max(results, key=lambda r: r.get("score", 0))
    if best.get("score", 0) > 0:
        yield {"type": "best", "label": best["label"],
               "tok_s": best["tok_s"],
               "vram_gb": best.get("vram_gb", 0),
               "ram_gb": best.get("ram_gb", 0)}

    yield {"type": "done"}
