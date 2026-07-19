"""Système de détection des ressources matérielles.

Détecte :
- GPU(s) NVIDIA via nvidia-smi
- RAM via /proc/meminfo
- CPU via /proc/cpuinfo et nproc

Toute la détection est purement structurelle : elle parse des fichiers
système ou des sorties de commandes, pas de base de données matérielle.
"""

import asyncio
import logging
import os
import re
import time
from typing import Optional

from app.models import CPUInfo, GPUInfo, RAMInfo, SystemStatus

logger = logging.getLogger("ai-runner")


# Cache de détection système (TTL : 2 secondes)
_system_cache: Optional[SystemStatus] = None
_system_cache_time: float = 0.0
_SYSTEM_CACHE_TTL = 2.0


# ─── GPU ────────────────────────────────────────────────


async def _run_nvidia_smi(fields: list[str]) -> tuple[Optional[str], Optional[str], int]:
    """Lance nvidia-smi avec les champs demandés et retourne (stdout, stderr, returncode).

    Retourne (None, None, -1) si nvidia-smi n'est pas trouvé ou timeout.
    """
    try:
        query = ",".join(fields)
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        return (
            stdout.decode(errors="replace") if stdout else "",
            stderr.decode(errors="replace") if stderr else "",
            proc.returncode if proc.returncode is not None else -1,
        )
    except FileNotFoundError:
        logger.warning("nvidia-smi non trouvé — GPU non détecté")
        return None, None, -1
    except asyncio.TimeoutError:
        logger.warning("nvidia-smi a dépassé le timeout de 10s")
        return None, None, -1


def _parse_gpu_lines(stdout: str, has_compute_cap: bool) -> list[GPUInfo]:
    """Parse la sortie CSV de nvidia-smi en liste de GPUInfo.

    Args:
        stdout: Sortie texte de nvidia-smi (format csv,noheader,nounits)
        has_compute_cap: True si compute_cap fait partie des champs demandés
    """
    import csv

    gpus: list[GPUInfo] = []
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Utiliser le module csv pour gérer correctement les champs entre guillemets
        # (certains noms de GPU peuvent contenir des virgules)
        try:
            row = next(csv.reader([line]))
        except Exception:
            # Fallback: split simple sur ", "
            row = [p.strip() for p in line.split(", ")]
        parts = [p.strip() for p in row]

        # Nettoyer les unités résiduelles (au cas où nounits n'est pas supporté)
        parts = [p.replace(" MiB", "").replace(" W", "") for p in parts]

        min_parts = 6 if not has_compute_cap else 7
        if len(parts) < min_parts:
            logger.debug(f"Ligne GPU ignorée (pas assez de champs): {line}")
            continue

        try:
            idx = 0
            index = int(parts[idx]); idx += 1
            name = parts[idx]; idx += 1
            vram_total = _mib_to_gb(float(parts[idx])); idx += 1
            vram_free = _mib_to_gb(float(parts[idx])); idx += 1
            vram_used = _mib_to_gb(float(parts[idx])); idx += 1

            if has_compute_cap:
                compute_cap = parts[idx] if parts[idx] != "N/A" else ""
                idx += 1
            else:
                compute_cap = ""

            driver = parts[idx] if idx < len(parts) else ""

            gpu = GPUInfo(
                index=index,
                name=name,
                vram_total_gb=vram_total,
                vram_free_gb=vram_free,
                vram_used_gb=vram_used,
                compute_cap=compute_cap,
                driver=driver,
            )
            gpus.append(gpu)
        except (ValueError, IndexError) as e:
            logger.debug(f"Erreur parsing ligne GPU '{line}': {e}")
            continue

    return gpus


async def detect_gpu() -> list[GPUInfo]:
    """Détecte les GPUs NVIDIA via nvidia-smi.

    Retourne une liste vide si nvidia-smi n'est pas disponible.
    Utilise une requête de secours sans compute_cap si la requête complète
    échoue (compute_cap n'est pas supporté par toutes les versions de nvidia-smi).
    """
    # Tentative 1 : requête complète avec compute_cap
    full_fields = [
        "index", "name", "memory.total", "memory.free",
        "memory.used", "compute_cap", "driver_version",
    ]
    stdout, stderr, rc = await _run_nvidia_smi(full_fields)

    if stdout is not None and rc == 0:
        gpus = _parse_gpu_lines(stdout, has_compute_cap=True)
        if gpus:
            for g in gpus:
                logger.debug(
                    f"GPU détecté : {g.name} — "
                    f"VRAM {g.vram_total_gb} Go (libre: {g.vram_free_gb} Go), "
                    f"compute_cap={g.compute_cap or 'N/A'}, driver={g.driver}"
                )
            logger.debug(f"Detection GPU: {len(gpus)} GPU(s) trouvé(s)")
            return gpus
        else:
            logger.warning("nvidia-smi a réussi mais aucun GPU n'a été parsé")
            if stderr:
                logger.debug(f"nvidia-smi stderr: {stderr.strip()}")
            return []

    # Tentative 2 : requête sans compute_cap (plus compatible)
    if stdout is not None and rc != 0:
        logger.info(
            f"nvidia-smi a échoué (rc={rc}) avec compute_cap, "
            f"réessai sans ce champ"
        )
        if stderr:
            logger.debug(f"nvidia-smi stderr: {stderr.strip()}")

    basic_fields = [
        "index", "name", "memory.total", "memory.free",
        "memory.used", "driver_version",
    ]
    stdout2, stderr2, rc2 = await _run_nvidia_smi(basic_fields)

    if stdout2 is not None and rc2 == 0:
        gpus = _parse_gpu_lines(stdout2, has_compute_cap=False)
        for g in gpus:
            logger.debug(
                f"GPU détecté : {g.name} — "
                f"VRAM {g.vram_total_gb} Go (libre: {g.vram_free_gb} Go), "
                f"driver={g.driver}"
            )
        logger.debug(f"Detection GPU: {len(gpus)} GPU(s) trouvé(s)")
        return gpus

    # nvidia-smi indisponible ou aucune GPU
    if rc == -1 and rc2 == -1:
        logger.info("nvidia-smi indisponible — mode CPU uniquement")
    return []


def _mib_to_gb(mib: float) -> float:
    """Convertit des Mio (mebibytes) en Go (gigabytes)."""
    return round(mib / 1024, 2)


# ─── RAM ────────────────────────────────────────────────


async def detect_ram() -> RAMInfo:
    """Lit /proc/meminfo pour récupérer la RAM totale et libre.

    Retourne des valeurs à 0 si le fichier est inaccessible (Windows, etc.).
    """
    try:
        with open("/proc/meminfo") as f:
            content = f.read()

        total_kb = _parse_meminfo_line(content, "MemTotal")
        free_kb = _parse_meminfo_line(content, "MemFree")
        available_kb = _parse_meminfo_line(content, "MemAvailable")

        return RAMInfo(
            total_gb=round(total_kb / 1_048_576, 2),
            free_gb=round(free_kb / 1_048_576, 2),
            available_gb=round(available_kb / 1_048_576, 2),
        )
    except FileNotFoundError:
        return RAMInfo(total_gb=0.0, free_gb=0.0, available_gb=0.0)


def _parse_meminfo_line(content: str, key: str) -> float:
    """Extrait une valeur numérique de /proc/meminfo.

    Exemple : "MemTotal:       65746812 kB" → 65746812.0
    """
    match = re.search(rf"^{key}:\s+(\d+)\s+kB", content, re.MULTILINE)
    if match:
        return float(match.group(1))
    return 0.0


# ─── CPU ────────────────────────────────────────────────


async def detect_cpu() -> CPUInfo:
    """Détecte le nombre de cœurs, threads et le modèle du CPU."""
    cores = _count_cpu_cores()
    threads = _count_cpu_threads()
    model = _get_cpu_model()

    return CPUInfo(cores=cores, threads=threads, model=model)


def _count_cpu_cores() -> int:
    """Compte le nombre de cœurs physiques via /proc/cpuinfo.

    core id unique = un cœur physique.
    """
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
        core_ids = set(re.findall(r"^core id\s+:\s+(\d+)", content, re.MULTILINE))
        return max(len(core_ids), 1)
    except FileNotFoundError:
        return 1


def _count_cpu_threads() -> int:
    """Compte le nombre de threads (logiques) via nproc."""
    try:
        proc = subprocess_run("nproc")
        return int(proc.strip())
    except Exception:
        # Fallback: compter les "processor" dans /proc/cpuinfo
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read()
            return len(re.findall(r"^processor\s+:", content, re.MULTILINE))
        except FileNotFoundError:
            return 1


def subprocess_run(cmd: str) -> str:
    """Execute une commande simple et retourne stdout."""
    import subprocess
    try:
        return subprocess.check_output(cmd.split(), timeout=5).decode().strip()
    except Exception:
        return ""


def _get_cpu_model() -> str:
    """Récupère le nom du modèle CPU."""
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
        match = re.search(r"^model name\s+:\s+(.+)", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
    except FileNotFoundError:
        pass
    return ""


# ─── Aggregateur ────────────────────────────────────────


async def detect(force: bool = False) -> SystemStatus:
    """Détecte toutes les ressources système en parallèle.

    Utilise un cache de {_SYSTEM_CACHE_TTL}s pour éviter les appels
    répétés à nvidia-smi (coûteux). Passer force=True pour forcer
    une détection fraîche.

    C'est le point d'entrée principal pour la détection système.
    """
    global _system_cache, _system_cache_time

    now = time.time()
    if not force and _system_cache is not None and (now - _system_cache_time) < _SYSTEM_CACHE_TTL:
        return _system_cache

    gpu_task = asyncio.create_task(detect_gpu())
    ram_task = asyncio.create_task(detect_ram())
    cpu_task = asyncio.create_task(detect_cpu())

    gpus, ram, cpu = await asyncio.gather(gpu_task, ram_task, cpu_task)

    mode = "cuda" if gpus else "cpu"

    logger.debug(
        f"Détection système : mode={mode}, GPUs={len(gpus)}, "
        f"RAM dispo={ram.available_gb} Go, CPU cores={cpu.cores}"
    )

    result = SystemStatus(
        gpu=gpus,
        ram=ram,
        cpu=cpu,
        available=bool(gpus) or ram.total_gb > 0,
        mode=mode,
    )

    _system_cache = result
    _system_cache_time = now
    return result
