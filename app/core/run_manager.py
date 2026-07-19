"""Gestionnaire de processus llama-server.

Architecture :
  - llama-server est démarré en arrière-plan avec le modèle chargé
  - Il expose une API HTTP OpenAI-compatible sur localhost
  - Notre app proxy les requêtes vers llama-server
  - Le modèle reste chargé entre les messages (rapide)

Avantages vs llama-completion :
  - Pas de mode interactif
  - Streaming propre via HTTP
  - Modèle reste en mémoire entre les requêtes
  - API standard OpenAI
"""

import asyncio
import json
import logging
import os
import signal
import socket
import time
import uuid
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx

from app.core import config as app_config
from app.models import RunHistory

logger = logging.getLogger("ai-runner")

def _find_free_port():
    """Trouve un port libre sur localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Hôte interne pour llama-server (port déterminé dynamiquement)
LLAMA_SERVER_HOST = "127.0.0.1"


async def _drain_stream(stream, buffer: deque, prefix: str = ""):
    """Draine un stream en continu pour éviter le deadlock du pipe.

    Lit les lignes du stream une par une et les stocke dans un deque
    (pour les derniers logs) + les journalise en DEBUG.

    Args:
        stream: Le stream asyncio (stdout ou stderr du process)
        buffer: Un deque(maxlen=...) pour stocker les dernières lignes
        prefix: Préfixe pour les logs (ex: "[llama-server] ")
    """
    while True:
        try:
            line = await stream.readline()
        except Exception:
            break
        if not line:
            break
        decoded = line.decode(errors="replace").rstrip()
        buffer.append(decoded)
        logger.debug(f"{prefix}{decoded}")


class RunStatus(str, Enum):
    CREATED = "created"
    LOADING = "loading"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


class ServerState:
    """État du serveur llama-server."""

    def __init__(self, model_id: str, model_path: str, params: dict):
        self.model_id: str = model_id
        self.model_path: str = model_path
        self.params: dict = params
        self.port: int = _find_free_port()
        self.url: str = f"http://127.0.0.1:{self.port}"
        self.status: RunStatus = RunStatus.CREATED
        self.process: Optional[asyncio.subprocess.Process] = None
        self.tokens_generated: int = 0
        self.speed_tokens_per_sec: float = 0.0
        self.vram_used_gb: float = 0.0
        self.ram_used_gb: float = 0.0
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None
        self.error_message: str = ""
        # Buffers circulaires pour les derniers logs du process
        self._stderr_lines: deque = deque(maxlen=30)
        self._stdout_lines: deque = deque(maxlen=30)
        # Tâches de drain des pipes stdout/stderr
        self._drain_tasks: list[asyncio.Task] = []

    def cancel_drain_tasks(self):
        """Annule les tâches de drain des pipes."""
        for task in self._drain_tasks:
            if not task.done():
                task.cancel()
        self._drain_tasks.clear()

    def recent_stderr(self, n: int = 15) -> str:
        """Retourne les n dernières lignes de stderr pour le debug."""
        return "\n".join(list(self._stderr_lines)[-n:])

    def to_history(self) -> RunHistory:
        return RunHistory(
            id=uuid.uuid4().hex[:12],
            model_id=self.model_id,
            status=self.status.value,
            tokens_generated=self.tokens_generated,
            avg_speed=self.speed_tokens_per_sec,
            params=self.params,
            started_at=self.started_at,
            ended_at=self.ended_at,
        )


async def _get_resource_usage() -> tuple[float, float]:
    """Retourne (VRAM_utilisée_Go, RAM_utilisée_Go)."""
    vram = 0.0
    ram = 0.0
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        if proc.returncode == 0:
            miB = float(stdout.decode().strip().split("\n")[0].strip())
            vram = round(miB / 1024, 2)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            content = f.read()
        import re
        total = re.search(r"MemTotal:\s+(\d+)", content)
        avail = re.search(r"MemAvailable:\s+(\d+)", content)
        if total and avail:
            ram = round((float(total.group(1)) - float(avail.group(1))) / 1_048_576, 2)
    except Exception:
        pass
    return vram, ram


_run_manager: Optional["RunManager"] = None


class RunManager:
    """Gère le serveur llama-server et les requêtes de chat."""

    def __init__(self):
        self.server: Optional[ServerState] = None
        self._monitor_task: Optional[asyncio.Task] = None

    def _find_binary(self) -> Optional[str]:
        """Trouve le binaire llama-server."""
        binary_path = Path(app_config.config.llamacpp.binary_path)
        parent = binary_path.parent
        server_bin = parent / "llama-server"
        if server_bin.is_file() and os.access(str(server_bin), os.X_OK):
            return str(server_bin)
        import shutil
        return shutil.which("llama-server")

    async def start_server(
        self,
        model_id: str,
        model_path: str,
        params: dict,
        load_timeout: int = 120,
        allow_fallback: bool = True,
    ) -> ServerState:
        """Démarre llama-server avec le modèle chargé.

        Si le chargement échoue avec des flags multi-GPU (split_mode/tensor_split),
        réessaie automatiquement sans ces flags avec un ngl réduit.

        Args:
            load_timeout: Timeout (secondes) pour attendre que llama-server
                soit prêt. Les gros modèles (35B+) peuvent nécessiter 2-3 min.
            allow_fallback: Si False, n'essaie pas de fallback automatique
                (utilisé par le benchmark pour tester une config exacte).
        """
        # Si le serveur tourne déjà avec le même modèle, réutiliser
        if (self.server and self.server.status == RunStatus.RUNNING
                and self.server.model_id == model_id):
            return self.server

        # Arrêter le serveur en cours si nécessaire (et annuler l'ancien
        # monitor task pour éviter qu'il continue à poller un serveur mort)
        if self.server and self.server.status in (RunStatus.RUNNING, RunStatus.LOADING):
            await self.stop()
        else:
            # Même si le serveur est en erreur, il peut rester un process
            # zombie (timeout) qu'il faut tuer + annuler le monitor
            self._cancel_monitor_task()
            if self.server and self.server.process:
                await self._kill_process(self.server)

        # Tentative initiale
        state = await self._try_start(model_id, model_path, params, load_timeout)

        if not allow_fallback:
            return state

        # Si échec (souvent OOM ou conflit tensoriel avec split multi-GPU),
        # réessayer en laissant llama-server gérer la distribution automatiquement
        if state.status == RunStatus.ERROR:
            err_msg = state.error_message
            logger.warning(f"Échec, tentative sans split manuel: {err_msg[:100]}")

            # Retirer les flags multi-GPU qui peuvent causer des conflits
            # avec les tenseurs internes du modèle (ex: fused Gated Delta Net)
            fallback = {k: v for k, v in params.items()
                        if k not in ("split_mode", "tensor_split", "main_gpu", "override_tensor")}
            fallback["no_kv_offload"] = False

            # Estimer le nombre de couches qui tiennent sur LE MEILLEUR GPU seul
            # (sans tensor-split, toutes les couches GPU vont sur GPU 0)
            vram_guess = 7.0  # ~7 Go utilisables par GPU après overhead CUDA
            model_gb_guess = 30.0  # estimation conservative
            n_layers_guess = 64  # valeur typique pour un 27-31B
            layer_gb = model_gb_guess / n_layers_guess
            ngl_safe = max(1, int(vram_guess / layer_gb))
            fallback["ngl"] = ngl_safe
            logger.info(f"Fallback: ngl_safe={ngl_safe}")

            await asyncio.sleep(1)
            state = await self._try_start(model_id, model_path, fallback, load_timeout)

            # Si toujours en échec, tenter avec ngl/2
            if state.status == RunStatus.ERROR:
                fallback["ngl"] = max(1, ngl_safe // 2)
                await asyncio.sleep(1)
                state = await self._try_start(model_id, model_path, fallback, load_timeout)

            # Dernier recours : CPU only (ngl=0) — seulement si pas de GPU
            if state.status == RunStatus.ERROR:
                from app.core.system_detector import detect as _detect_system
                _sys = await _detect_system(force=True)
                if _sys.gpu:
                    # GPU disponible : ne pas tomber à ngl=0, garder un minimum
                    fallback["ngl"] = max(1, ngl_safe // 4)
                    logger.warning(
                        f"Échec, mais GPU disponible ({len(_sys.gpu)} GPU(s)) — "
                        f"ngl maintenu à {fallback['ngl']} au lieu de 0"
                    )
                else:
                    logger.warning("Échec, fallback CPU only")
                    fallback["ngl"] = 0
                await asyncio.sleep(1)
                state = await self._try_start(model_id, model_path, fallback, load_timeout)

        return state

    @staticmethod
    async def _kill_process(state: "ServerState") -> None:
        """Tue proprement le process llama-server d'un ServerState.

        Utilisé pour nettoyer les processes zombies lorsqu'une tentative
        de démarrage échoue (timeout/OOM) — sans cela, le process continue
        à occuper la VRAM et provoque des crashes en cascade sur les
        configs suivantes.
        """
        proc = state.process
        if proc is None:
            return
        try:
            if proc.returncode is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.send_signal(signal.SIGKILL)
                    await proc.wait()
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage du process: {e}")

    def _cancel_monitor_task(self) -> None:
        """Annule la tâche de monitoring en arrière-plan si elle existe."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            self._monitor_task = None

    async def _try_start(
        self,
        model_id: str,
        model_path: str,
        params: dict,
    ) -> ServerState:
        """Tente de démarrer llama-server avec les paramètres donnés."""
        binary = self._find_binary()
        if not binary:
            state = ServerState(model_id, model_path, params)
            state.status = RunStatus.ERROR
            state.error_message = (
                f"llama-server introuvable. Téléchargez llama.cpp via "
                f"POST /api/v1/llamacpp/update"
            )
            return state

        state = ServerState(model_id, model_path, params)
        state.status = RunStatus.LOADING
        state.started_at = datetime.now()
        self.server = state

        # Construire la commande llama-server
        port = state.port
        cmd = [binary, "-m", model_path, "--host", LLAMA_SERVER_HOST,
               "--port", str(port)]

        ngl = params.get("ngl", 99)
        cmd.extend(["-ngl", str(ngl)])

        if params.get("cache_type_k"):
            cmd.extend(["--cache-type-k", params["cache_type_k"]])
            cmd.extend(["--cache-type-v", params["cache_type_k"]])

        ctx = params.get("ctx_size", 8192)
        cmd.extend(["--ctx-size", str(ctx)])

        threads = params.get("threads", 4)
        cmd.extend(["--threads", str(threads)])

        tbatch = params.get("threads_batch")
        if tbatch:
            cmd.extend(["--threads-batch", str(tbatch)])

        ubatch = params.get("ubatch_size")
        if ubatch:
            cmd.extend(["--ubatch-size", str(ubatch)])
        batch = params.get("batch_size")
        if batch:
            cmd.extend(["--batch-size", str(batch)])

        if params.get("flash_attn"):
            val = params["flash_attn"]
            if val is True:
                cmd.append("--flash-attn")
                cmd.append("on")
            else:
                cmd.append("--flash-attn")
                cmd.append(str(val))

        # No KV offload (KV cache reste sur GPU)
        if params.get("no_kv_offload"):
            cmd.append("--no-kv-offload")

        # Override tensor pour MoE
        for ot in params.get("override_tensor", []):
            cmd.extend(["--override-tensor", ot])

        # Multi-GPU : split mode, tensor split, main GPU
        split_mode = params.get("split_mode")
        if split_mode and split_mode != "none":
            cmd.extend(["--split-mode", split_mode])
            ts = params.get("tensor_split")
            if ts:
                cmd.extend(["--tensor-split", ts])
            mg = params.get("main_gpu")
            if mg is not None and mg != 0:
                cmd.extend(["--main-gpu", str(mg)])

        # Pas de warmup (utile en fallback quand le warmup échoue)
        if params.get("no_warmup"):
            cmd.append("--no-warmup")

        logger.info(f"Démarrage llama-server: ngl={ngl}, threads={threads}, ctx={ctx}, model={model_path}")
        logger.info(f"Commande complète: {' '.join(cmd)}")

        try:
            state.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            state.status = RunStatus.RUNNING

            # ── Drainer les pipes stdout/stderr en continu ──
            # Sans cette étape, le buffer (~64KB) se remplit avec les logs
            # de llama-server pendant le chargement du modèle, bloque le
            # process (pipe deadlock), et le health check ne répond jamais.
            state._drain_tasks.append(
                asyncio.create_task(
                    _drain_stream(state.process.stderr, state._stderr_lines, "[llama-server] ")
                )
            )
            state._drain_tasks.append(
                asyncio.create_task(
                    _drain_stream(state.process.stdout, state._stdout_lines, "[llama-server/stdout] ")
                )
            )

            # Attendre que llama-server soit prêt (poll health endpoint)
            await self._wait_for_server_ready(state)

            # Sauvegarder dans l'historique SQLite
            await save_run(state)

            # Lancer le monitoring en arrière-plan
            self._monitor_task = asyncio.create_task(self._monitor(state))

        except FileNotFoundError:
            state.cancel_drain_tasks()
            state.status = RunStatus.ERROR
            state.error_message = f"llama-server introuvable: {binary}"
        except Exception as e:
            state.cancel_drain_tasks()
            state.status = RunStatus.ERROR
            state.error_message = str(e)
            logger.error(f"Erreur lancement llama-server: {e}")

        return state

    async def _wait_for_server_ready(self, state: ServerState, timeout: int = 120):
        """Attend que llama-server soit prêt à accepter des requêtes."""
        start = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                if state.process and state.process.returncode is not None:
                    # Le process est terminé (crash)
                    # Utiliser recent_stderr() car le drain task lit déjà stderr
                    stderr = state.recent_stderr(15)
                    state.status = RunStatus.ERROR
                    state.error_message = f"llama-server crashé: {stderr}"
                    raise RuntimeError(state.error_message)
                try:
                    resp = await client.get(f"http://127.0.0.1:{state.port}/health", timeout=2)
                    if resp.status_code == 200:
                        logger.info("llama-server prêt !")
                        return
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(1)
        # Timeout : capturer les logs pour le debug
        # Utiliser recent_stderr() car le drain task lit déjà stderr
        stderr = state.recent_stderr(15)
        state.status = RunStatus.ERROR
        state.error_message = f"Timeout ({timeout}s): llama-server n'a pas démarré. Logs: {stderr}"
        raise RuntimeError(state.error_message)

    async def chat(
        self,
        messages: list[dict],
        params: dict,
    ) -> AsyncGenerator[str, None]:
        """Envoie une requête de chat à llama-server et streame la réponse.

        Génère des événements SSE au format interne :
        - {"type": "token", "text": "...", "speed": X, "tokens": N}
        - {"type": "stats", "vram_gb": X, "ram_gb": Y}
        - {"type": "done", "tokens": N}
        - {"type": "error", "message": "..."}

        Le minuteur de vitesse démarre à l'arrivée du **premier token**,
        pas au début de la méthode. Cela exclut le temps de traitement
        du prompt (chargement, pré-fill) et ne mesure que la vitesse
        d'inférence réelle (génération token par token).
        """
        if not self.server or self.server.status != RunStatus.RUNNING:
            yield f"data: {json.dumps({'type': 'error', 'message': 'llama-server non démarré'})}\n\n"
            return

        temperature = params.get("temp", 0.7)
        max_tokens = params.get("max_tokens", 512)

        request_body = {
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        token_count = 0
        # Démarré seulement à l'arrivée du premier token (exclut le pré-fill)
        first_token_time: Optional[float] = None

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"http://127.0.0.1:{self.server.port}/v1/chat/completions",
                    json=request_body,
                    timeout=300,
                ) as resp:
                    resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                token_count += 1
                                # Démarrer le chrono au premier token
                                if first_token_time is None:
                                    first_token_time = time.time()
                                # Vitesse d'inférence réelle :
                                # on exclut le premier token du décompte car
                                # il inclut le temps de pré-fill du prompt.
                                elapsed = time.time() - first_token_time
                                speed = (token_count - 1) / elapsed if elapsed > 0 else 0
                                self.server.tokens_generated = token_count
                                self.server.speed_tokens_per_sec = speed
                                yield f"data: {json.dumps({'type': 'token', 'text': content, 'speed': round(speed, 1), 'tokens': token_count})}\n\n"
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

            yield f"data: {json.dumps({'type': 'done', 'tokens': token_count})}\n\n"

        except httpx.ConnectError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'llama-server injoignable'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    async def stop(self) -> None:
        """Arrête le serveur llama-server."""
        if not self.server or not self.server.process:
            return

        state = self.server
        state.status = RunStatus.STOPPING
        pid = state.process.pid

        logger.info(f"Arrêt de llama-server (PID {pid})...")

        try:
            state.process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(state.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                state.process.send_signal(signal.SIGKILL)
                await state.process.wait()
        except ProcessLookupError:
            pass

        state.status = RunStatus.STOPPED
        state.ended_at = datetime.now()

        # Sauvegarder dans l'historique SQLite
        await save_run(state)

    async def _monitor(self, state: ServerState) -> None:
        """Surveille le serveur et les ressources."""
        while state.status == RunStatus.RUNNING:
            try:
                vram, ram = await _get_resource_usage()
                state.vram_used_gb = vram
                state.ram_used_gb = ram
            except Exception:
                pass
            await asyncio.sleep(2)

    def is_running(self) -> bool:
        """Vérifie si le serveur tourne."""
        return (self.server is not None
                and self.server.status == RunStatus.RUNNING)

    def get_loaded_model_id(self) -> Optional[str]:
        """Retourne l'ID du modèle actuellement chargé, ou None."""
        if self.server and self.server.status in (
            RunStatus.RUNNING, RunStatus.LOADING
        ):
            return self.server.model_id
        return None


# ─── Persistance SQLite ─────────────────────────────

async def _get_db_path() -> str:
    db_dir = app_config.config.storage.data_dir
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "history.db")


async def _init_db():
    import aiosqlite
    db_path = await _get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                status TEXT NOT NULL,
                tokens_generated INTEGER DEFAULT 0,
                avg_speed REAL DEFAULT 0.0,
                params TEXT DEFAULT '{}',
                started_at TEXT,
                ended_at TEXT
            )
        """)
        await db.commit()


async def save_run(state: ServerState):
    try:
        await _init_db()
        db_path = await _get_db_path()
        import aiosqlite
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO runs
                   (id, model_id, status, tokens_generated, avg_speed, params,
                    started_at, ended_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    uuid.uuid4().hex[:12],
                    state.model_id,
                    state.status.value,
                    state.tokens_generated,
                    round(state.speed_tokens_per_sec, 2),
                    json.dumps(state.params),
                    state.started_at.isoformat() if state.started_at else None,
                    state.ended_at.isoformat() if state.ended_at else None,
                ),
            )
            await db.commit()
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder l'historique: {e}")


async def get_history(limit: int = 50) -> list[dict]:
    try:
        await _init_db()
        db_path = await _get_db_path()
        import aiosqlite
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.warning(f"Impossible de lire l'historique: {e}")
        return []


def get_run_manager() -> RunManager:
    global _run_manager
    if _run_manager is None:
        _run_manager = RunManager()
    return _run_manager