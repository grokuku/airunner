"""Système de détection des ressources matérielles.

Détecte :
- GPU(s) NVIDIA via nvidia-smi
- RAM via /proc/meminfo
- CPU via /proc/cpuinfo et nproc

Toute la détection est purement structurelle : elle parse des fichiers
système ou des sorties de commandes, pas de base de données matérielle.
"""

import asyncio
import os
import re
import time
from typing import Optional

from app.models import CPUInfo, GPUInfo, RAMInfo, SystemStatus


# Cache de détection système (TTL : 2 secondes)
_system_cache: Optional[SystemStatus] = None
_system_cache_time: float = 0.0
_SYSTEM_CACHE_TTL = 2.0


# ─── GPU ────────────────────────────────────────────────


async def detect_gpu() -> list[GPUInfo]:
    """Détecte les GPUs NVIDIA via nvidia-smi.

    Retourne une liste vide si nvidia-smi n'est pas disponible.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

        if proc.returncode != 0:
            return []

        gpus = []
        for line in stdout.decode().strip().split("\n"):
            line = line.strip().replace(" MiB", "").replace(" W", "")
            parts = [p.strip() for p in line.split(", ")]
            if len(parts) < 7:
                continue

            try:
                gpu = GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_total_gb=_mib_to_gb(float(parts[2])),
                    vram_free_gb=_mib_to_gb(float(parts[3])),
                    vram_used_gb=_mib_to_gb(float(parts[4])),
                    compute_cap=parts[5] if parts[5] != "N/A" else "",
                    driver=parts[6],
                )
                gpus.append(gpu)
            except (ValueError, IndexError):
                continue

        return gpus

    except (FileNotFoundError, asyncio.TimeoutError):
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
