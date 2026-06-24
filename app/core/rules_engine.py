"""Moteur de règles pour la sélection optimale des paramètres llama.cpp.

Prend en entrée :
  - Les métadonnées du modèle (depuis GGUF parser)
  - Les ressources système (depuis system_detector)
  - Les préférences utilisateur

Applique des règles dérivées de la vidéo "Everything That Actually Matters
for Local AI" (Codacus) et retourne une configuration optimale.

Aucune base de modèles statique : tout est calculé à partir des données
réelles du GGUF et du système.
"""

from typing import Optional

from app.models import (
    ConfigRequest, ConfigSuggestion, GPUInfo,
    ModelMeta, RamEstimate, VramEstimate, SystemStatus,
)


# ─── Constantes ────────────────────────────────────────

# Quantifications disponibles (du plus précis au moins précis)
# Chaque entrée : (nom, bits_per_weight, description)
AVAILABLE_QUANTS = [
    ("F16",    16.0, "Float 16, précision maximale, très rare"),
    ("Q8_K",   8.5,  "8-bit K-quant, quasi sans perte"),
    ("Q8_0",   8.0,  "8-bit, pas de perte"),
    ("Q6_K",   6.5,  "6-bit K-quant, très bonne qualité"),
    ("Q5_K",   5.5,  "5-bit K-quant (toute variante)"),
    ("Q5_0",   5.0,  "5-bit legacy"),
    ("Q4_K",   4.5,  "4-bit K-quant (toute variante), sweet spot"),
    ("Q4_0",   4.0,  "4-bit legacy"),
    ("IQ4_NL", 4.5,  "4-bit importance quant, nouvel layout"),
    ("IQ4_XS", 4.0,  "4-bit importance quant, ~10% plus petit"),
    ("Q3_K",   3.5,  "3-bit K-quant (toute variante)"),
    ("IQ3_XXS",3.0,  "3-bit importance quant très serré"),
    ("Q2_K",   2.5,  "2-bit K-quant, perte significative"),
]

# Bits par paramètre pour le cache KV
KV_CACHE_OPTIONS = [
    ("q8_0", 1.0,   "1 byte par valeur, quasi sans perte"),
    ("q4_0", 0.5,   "0.5 byte par valeur, bon pour long contexte"),
    ("q4_1", 0.5,   "0.5 byte avec offset, légèrement mieux"),
]

# Overhead fixe estimé (puffers, overhead CUDA, etc.)
VRAM_OVERHEAD_GB = 0.3

# Ratio attention/experts pour MoE (approximation)
# ~15% des paramètres pour l'attention dans un MoE typique
MOE_ATTENTION_RATIO = 0.15


# ─── Fonctions de calcul ──────────────────────────────


def estimate_model_size(params_b: float, bits: float) -> float:
    """Estime la taille d'un modèle en Go selon le nombre de bits par paramètre.

    Formule : params × bits / 8 / 1e9
    Exemple : 35B × 4.5 bits / 8 / 1e9 = 19.7 Go
    """
    return params_b * bits / 8.0


def estimate_kv_cache_gb(
    ctx_size: int,
    hidden_size: int,
    n_layers: int,
    bits_per_value: float,
) -> float:
    """Estime la taille du cache KV en Go.

    Formule : ctx × hidden × layers × 2 (K+V) × bits / 8 / 1e9

    Args:
        ctx_size: Nombre de tokens de contexte
        hidden_size: Taille du vecteur caché (embedding_length)
        n_layers: Nombre de couches
        bits_per_value: Bits par valeur de cache (1.0 pour q8_0, 0.5 pour q4_0)
    """
    return (ctx_size * hidden_size * n_layers * 2 * bits_per_value) / 8.0 / 1e9


def estimate_gpu_bandwidth(gpu: GPUInfo) -> float:
    """Estimation conservative de la bande passante GPU en Go/s.

    Basé sur le nom du GPU. Fallback à 200 Go/s si inconnu.
    """
    name = gpu.name.lower()
    if "rtx 4090" in name:
        return 1008
    elif "rtx 4080" in name:
        return 716
    elif "rtx 4070" in name:
        return 504
    elif "rtx 4060" in name:
        return 272
    elif "rtx 3090" in name:
        return 936
    elif "rtx 3080" in name:
        return 760
    elif "rtx 3070" in name:
        return 448
    elif "rtx 3060" in name:
        return 360
    elif "rtx 3050" in name:
        return 224
    elif "tesla p40" in name or "tesla p100" in name:
        return 250
    else:
        # Valeur conservative pour GPU inconnu
        return 200


def estimate_speed(
    model_meta: ModelMeta,
    strategy: str,
    config_params: dict,
    gpu: Optional[GPUInfo],
    system: Optional[SystemStatus] = None,
) -> str:
    """Estime la vitesse d'inférence en tok/s.

    C'est une estimation approximative basée sur la bande passante.
    Multi-GPU : la bande passante est multipliée par le nombre de GPUs.
    """
    # Si paramètres inconnus, pas d'estimation possible
    if model_meta.active_params_b == 0:
        return "Estimation indisponible (paramètres inconnus)"

    # Si pas de GPU détecté du tout, estimation CPU très conservative
    if not gpu and not (system and system.gpu):
        return "1-5 tok/s (CPU only)"

    # Utiliser le premier GPU disponible pour la bande passante de base
    first_gpu = gpu or system.gpu[0] if system and system.gpu else None
    bandwidth_gbs = estimate_gpu_bandwidth(first_gpu) if first_gpu else 200

    # Multi-GPU : la bande passante totale est la somme de tous les GPUs
    if system and system.gpu and len(system.gpu) > 1:
        total_bandwidth = sum(estimate_gpu_bandwidth(g) for g in system.gpu)
        bandwidth_gbs = total_bandwidth

    if strategy == "dense_full":
        # Pleine vitesse GPU : limité par bande passante mémoire
        # tokens/s ≈ bandwidth / (param_actifs × bytes_per_param)
        bits = _get_bits_from_quant(config_params.get("quant", "Q4_K_M"))
        bytes_per_param = bits / 8.0
        params_b = model_meta.active_params_b
        # Facteur correctif pour overhead de calcul
        raw_tps = bandwidth_gbs / (params_b * bytes_per_param)
        raw_tps *= 0.3  # Facteur d'efficacité réaliste
        return _format_speed(raw_tps)

    elif strategy == "moe_offload":
        # MoE offload : attention sur GPU, experts sur CPU
        # La vitesse est limitée par le plus lent des deux
        # GPU: attention seulement (15% des paramètres)
        bits = _get_bits_from_quant(config_params.get("quant", "Q4_K_M"))
        bytes_per_param = bits / 8.0
        attention_params = model_meta.active_params_b * MOE_ATTENTION_RATIO
        gpu_tps = bandwidth_gbs / (attention_params * bytes_per_param)
        gpu_tps *= 0.3

        # CPU: experts
        cpu_tps = 15  # Estimation conservative ~15 tok/s pour CPU moderne
        raw_tps = min(gpu_tps, cpu_tps)
        return _format_speed(raw_tps)

    elif strategy == "dense_partial":
        # Offloading partiel : une partie sur GPU, une partie sur CPU
        # Estimation conservative : beaucoup plus lent
        bits = _get_bits_from_quant(config_params.get("quant", "Q4_K_M"))
        bytes_per_param = bits / 8.0
        params_b = model_meta.active_params_b

        # Ratio GPU vs CPU
        ngl = config_params.get("ngl", 1)
        n_layers = max(model_meta.block_count, 1)
        gpu_ratio = min(ngl / n_layers, 1.0)

        gpu_tps = bandwidth_gbs / (params_b * bytes_per_param * gpu_ratio)
        gpu_tps *= 0.15  # Pénalité d'offloading
        cpu_tps = 5  # Très lent sur CPU pour dense

        raw_tps = gpu_tps if gpu_ratio > 0.5 else cpu_tps
        return _format_speed(raw_tps)

    return "Inconnue"


def _get_bits_from_quant(quant_name: str) -> float:
    """Extrait le nombre de bits par paramètre depuis un nom de quant."""
    for name, bits, _ in AVAILABLE_QUANTS:
        if quant_name.startswith(name):
            return bits
    return 4.5  # Fallback Q4_K_M


def _format_speed(tps: float) -> str:
    """Formate une estimation de vitesse en chaîne lisible."""
    if tps > 100:
        return f"{tps:.0f}-{tps * 1.3:.0f} tok/s"
    elif tps > 10:
        return f"{tps:.0f}-{tps * 1.5:.0f} tok/s"
    else:
        return f"{max(1, int(tps))}-{max(2, int(tps * 2))} tok/s"


# ─── Moteur de règles principal ───────────────────────


def suggest(
    model_meta: ModelMeta,
    system: SystemStatus,
    request: Optional[ConfigRequest] = None,
) -> ConfigSuggestion:
    """Calcule la configuration optimale pour un modèle et un système donnés.

    C'est le point d'entrée principal du moteur de règles.

    Args:
        model_meta: Métadonnées du modèle (depuis GGUF parser)
        system: Ressources système (depuis system_detector)
        request: Préférences utilisateur optionnelles

    Returns:
        ConfigSuggestion avec stratégie, paramètres, estimations
    """
    ctx_size = request.ctx_size if request else 8192
    quant_preference = request.quant_preference if request else None

    # Récupérer le GPU principal et détecter multi-GPU
    gpu = system.gpu[0] if system.gpu else None
    multi_gpu = len(system.gpu) > 1 if system.gpu else False

    if multi_gpu:
        # VRAM combinée = somme de toutes les VRAM libres (pour quants + multi-GPU)
        vram_total = sum(g.vram_total_gb for g in system.gpu)
        vram_free = sum(g.vram_free_gb for g in system.gpu)
        # VRAM du meilleur GPU individuel (pour décider si un seul GPU suffit)
        vram_free_best_gpu = max(g.vram_free_gb for g in system.gpu)
    else:
        vram_total = gpu.vram_total_gb if gpu else 0.0
        vram_free = gpu.vram_free_gb if gpu else 0.0
        vram_free_best_gpu = vram_free
    ram_available = system.ram.available_gb

    # Paramètres du modèle
    params_b = model_meta.active_params_b
    is_moe = model_meta.is_moe
    n_layers = max(model_meta.block_count, 1)
    hidden_size = model_meta.embedding_length or 4096
    total_params_b = model_meta.params_b

    # ── Étape 1 : Déterminer les quants viables ──

    quants_viable = []
    best_quant = "Q4_K_M"  # Sweet spot par défaut

    for q_name, q_bits, q_desc in AVAILABLE_QUANTS:
        # Taille du modèle à ce quant
        model_gb = estimate_model_size(total_params_b, q_bits)

        # Cache KV en Q8 (estimation)
        kv_gb = estimate_kv_cache_gb(ctx_size, hidden_size, n_layers, 1.0)

        total_needed = model_gb + kv_gb + VRAM_OVERHEAD_GB

        if gpu and total_needed <= vram_free * 0.85:
            quants_viable.append(q_name)

    # Si aucun quant viable en full GPU, on prend Q4_K_M avec offloading
    if not quants_viable:
        quants_viable = ["Q4_K_M"]

    # Choix du quant
    if quant_preference and quant_preference in quants_viable:
        best_quant = quant_preference
    elif quants_viable:
        # Prendre le plus haut bits viable
        for q_name, _, _ in AVAILABLE_QUANTS:
            if q_name in quants_viable:
                best_quant = q_name
                break
        else:
            best_quant = quants_viable[0]

    # ── Étape 2 : Calculer VRAM / RAM ──

    best_bits = _get_bits_from_quant(best_quant)
    model_gb_final = estimate_model_size(total_params_b, best_bits)
    kv_gb = estimate_kv_cache_gb(ctx_size, hidden_size, n_layers, 1.0)

    # ── Étape 3 : Déterminer la stratégie ──

    strategy = "dense_full"
    override_tensor: list[str] = []
    ngl = 99
    warnings: list[str] = []
    ram_weights_gb = 0.0
    vram_weights_gb = model_gb_final
    no_kv_offload = False
    flash_attn = False
    split_mode = "none"
    tensor_split = ""
    main_gpu = 0

    if is_moe and gpu:
        # Stratégie MoE (basée sur la vidéo)
        # On essaie de mettre l'attention sur GPU, les experts sur CPU

        # Estimation : attention ≈ MOE_ATTENTION_RATIO des paramètres actifs
        attention_gb = estimate_model_size(
            total_params_b * MOE_ATTENTION_RATIO, best_bits
        )
        experts_gb = model_gb_final - attention_gb

        total_vram_needed = attention_gb + kv_gb + VRAM_OVERHEAD_GB

        if total_vram_needed <= vram_free * 0.85:
            # L'attention tient sur GPU → offloading optimal
            strategy = "moe_offload"
            if multi_gpu:
                # Répartir l'attention sur plusieurs GPUs
                per_gpu_ratios = [g.vram_free_gb for g in system.gpu]
                total_ratio = sum(per_gpu_ratios)
                tensor_split = ",".join(
                    str(max(1, int(r / total_ratio * 10)))
                    for r in per_gpu_ratios
                )
                split_mode = "layer"
                override_tensor = [
                    ".*attn.*=GPU",
                    ".*ffn_gate.*=CPU",
                    ".*ffn_down.*=CPU",
                    ".*ffn_up.*=CPU",
                ]
            else:
                split_mode = "none"
                override_tensor = [
                    ".*attn.*=CUDA0",
                    ".*ffn_gate.*=CPU",
                    ".*ffn_down.*=CPU",
                    ".*ffn_up.*=CPU",
                ]
            vram_weights_gb = attention_gb
            ram_weights_gb = experts_gb
            no_kv_offload = True
            warnings.append(
                "Experts offloadés sur CPU : assurez-vous d'avoir "
                f"au moins {experts_gb:.1f} Go de RAM libre"
            )
        else:
            # Attention ne tient pas complètement → offloading partiel
            strategy = "moe_offload"  # Même principe, moins de couches GPU
            n_attn_layers = int(
                (vram_free * 0.85 - kv_gb - VRAM_OVERHEAD_GB)
                / (model_gb_final / n_layers)
            )
            n_attn_layers = max(1, min(n_attn_layers, n_layers))
            ngl = n_attn_layers
            if multi_gpu:
                per_gpu_ratios = [g.vram_free_gb for g in system.gpu]
                total_ratio = sum(per_gpu_ratios)
                tensor_split = ",".join(
                    str(max(1, int(r / total_ratio * 10)))
                    for r in per_gpu_ratios
                )
                split_mode = "layer"
                override_tensor = [
                    ".*attn.*=GPU",
                    ".*mlp.*=CPU",
                ]
            else:
                override_tensor = [
                    ".*attn.*=CUDA0",
                    ".*mlp.*=CPU",
                ]
            vram_weights_gb = (model_gb_final / n_layers) * n_attn_layers
            ram_weights_gb = model_gb_final - vram_weights_gb
            no_kv_offload = True
            warnings.append(
                f"VRAM insuffisante pour toute l'attention. "
                f"Seulement {n_attn_layers}/{n_layers} couches sur GPU."
            )

    elif gpu:
        # Modèle dense
        total_needed_full = model_gb_final + kv_gb + VRAM_OVERHEAD_GB

        # 1) Vérifier si le modèle tient sur LE MEILLEUR GPU seul
        if total_needed_full <= vram_free_best_gpu * 0.85:
            strategy = "dense_full"
            split_mode = "none"
            tensor_split = ""
            main_gpu = 0
            ngl = 99
            vram_weights_gb = model_gb_final
            ram_weights_gb = 0.0
            if total_needed_full > vram_free_best_gpu * 0.7:
                warnings.append(
                    f"Utilisation VRAM élevée ({total_needed_full:.1f}/{vram_free_best_gpu:.1f} Go). "
                    "Surveillez les fuites mémoire."
                )

        # 2) Sinon, multi-GPU + VRAM combinée suffisante → split
        elif multi_gpu and total_needed_full <= vram_free * 0.85:
            strategy = "dense_full"
            ngl = n_layers
            per_gpu_ratios = [g.vram_free_gb for g in system.gpu]
            total_ratio = sum(per_gpu_ratios)
            tensor_split = ",".join(
                str(max(1, int(r / total_ratio * 10)))
                for r in per_gpu_ratios
            )
            split_mode = "layer"
            main_gpu = 0
            vram_weights_gb = model_gb_final
            ram_weights_gb = 0.0
            warnings.append(
                f"Modèle dense réparti sur {len(system.gpu)} GPUs "
                f"(split_mode=layer, tensor_split={tensor_split})."
            )

        # 3) Sinon, offloading forcé (multi-GPU si disponible)
        else:
            strategy = "dense_partial"
            if multi_gpu:
                split_mode = "layer"
                per_gpu_ratios = [g.vram_free_gb for g in system.gpu]
                total_ratio = sum(per_gpu_ratios)
                tensor_split = ",".join(
                    str(max(1, int(g.vram_free_gb / max(vram_free, 1) * 10)))
                    for g in system.gpu
                )
                main_gpu = 0
                warning_gpu_info = f" réparti sur {len(system.gpu)} GPUs"
            else:
                split_mode = "none"
                tensor_split = ""
                main_gpu = 0
                warning_gpu_info = ""
            ngl = int(
                (vram_free * 0.85 - kv_gb - VRAM_OVERHEAD_GB)
                / (model_gb_final / n_layers)
            )
            ngl = max(1, min(ngl, n_layers))
            vram_weights_gb = (model_gb_final / n_layers) * ngl
            ram_weights_gb = model_gb_final - vram_weights_gb
            warnings.append(
                f"Modèle dense{warning_gpu_info} trop grand pour la VRAM. "
                f"Offloading forcé : {ngl}/{n_layers} couches sur GPU. "
                "Les performances seront réduites."
            )
    else:
        # Pas de GPU
        strategy = "dense_partial"
        ngl = 0
        vram_weights_gb = 0.0
        ram_weights_gb = model_gb_final
        warnings.append("Aucun GPU détecté. Mode CPU uniquement — très lent.")

    # ── Étape 4 : Configurer le cache KV ──

    vram_after_weights = vram_free - vram_weights_gb - VRAM_OVERHEAD_GB
    cache_kv_bits = 1.0  # q8_0 par défaut
    cache_type = "q8_0"

    if kv_gb > vram_after_weights * 0.9:
        # Pas assez de VRAM pour KV cache en Q8
        kv_gb_q4 = estimate_kv_cache_gb(ctx_size, hidden_size, n_layers, 0.5)
        if kv_gb_q4 <= vram_after_weights * 0.9:
            cache_kv_bits = 0.5
            cache_type = "q4_0"
            warnings.append(
                "Cache KV en Q4 pour économiser la VRAM. "
                "Peut légèrement impacter le raisonnement long."
            )
        else:
            # Réduire le contexte
            max_ctx = int(
                (vram_after_weights * 0.8)
                / (hidden_size * n_layers * 2 * 1.0 / 8.0 / 1e9)
            )
            max_ctx = max(1024, min(max_ctx, ctx_size))
            if max_ctx < ctx_size:
                ctx_size = max_ctx
                warnings.append(
                    f"Contexte réduit à {ctx_size} tokens par manque de VRAM."
                )

    # ── Étape 5 : Configurer threads ──

    cpu_cores = system.cpu.cores
    if strategy == "moe_offload" or (strategy == "dense_partial" and ngl < n_layers):
        # Offloading actif → plus de threads CPU pour les experts
        threads = min(cpu_cores * 2, 16)
        threads_batch = max(1, threads // 2)
    else:
        # Full GPU → threads modérés
        threads = min(cpu_cores, 8)
        threads_batch = threads

    # ── Étape 6 : Options avancées ──

    if gpu and gpu.compute_cap:
        try:
            cc = float(gpu.compute_cap)
            flash_attn = cc >= 7.5  # Turing+ supporte flash attention
        except ValueError:
            flash_attn = False

    # ── Étape 7 : Estimation vitesse ──

    config_params = {
        "quant": best_quant,
        "ngl": ngl,
        "override_tensor": override_tensor,
        "cache_type_k": cache_type,
        "cache_type_v": cache_type,
        "ctx_size": ctx_size,
        "threads": threads,
        "threads_batch": threads_batch,
        "ubatch_size": 256 if vram_after_weights < 1.0 else 512,
        "batch_size": 512 if vram_after_weights < 1.0 else 2048,
        "flash_attn": flash_attn,
        "no_kv_offload": no_kv_offload,
        "temp": request.temp if request else 0.7,
        "split_mode": split_mode,
        "tensor_split": tensor_split,
        "main_gpu": main_gpu,
    }

    speed_estimate = estimate_speed(model_meta, strategy, config_params, gpu, system=system)

    # ── Assembler la réponse ──

    vram_estimate = VramEstimate(
        weights_gb=round(vram_weights_gb, 2),
        cache_kv_gb=round(kv_gb, 2),
        overhead_gb=VRAM_OVERHEAD_GB,
        total_gb=round(vram_weights_gb + kv_gb + VRAM_OVERHEAD_GB, 2),
        available_gb=round(vram_free, 2),
        free_after_gb=round(
            vram_free - vram_weights_gb - kv_gb - VRAM_OVERHEAD_GB, 2
        ),
    )

    ram_estimate = RamEstimate(
        weights_gb=round(ram_weights_gb, 2),
        total_gb=round(ram_weights_gb, 2),
        available_gb=round(ram_available, 2),
    )

    return ConfigSuggestion(
        strategy=strategy,
        quant=best_quant,
        quant_viable=quants_viable,
        vram=vram_estimate,
        ram=ram_estimate if ram_weights_gb > 0 else None,
        params=config_params,
        estimated_speed=speed_estimate,
        warnings=warnings,
        command_preview="",  # Sera rempli par command_builder
    )
