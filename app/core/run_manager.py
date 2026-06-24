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
    ) -> ServerState:
        """Démarre llama-server avec le modèle chargé.

        Si le chargement échoue avec des flags multi-GPU (split_mode/tensor_split),
        réessaie automatiquement sans ces flags avec un ngl réduit.
        """
        # Si le serveur tourne déjà avec le même modèle, réutiliser
        if (self.server and self.server.status == RunStatus.RUNNING
                and self.server.model_id == model_id):
            return self.server

        # Arrêter le serveur en cours si nécessaire
        if self.server and self.server.status in (RunStatus.RUNNING, RunStatus.LOADING):
            await self.stop()

        # Tentative initiale
        state = await self._try_start(model_id, model_path, params)

        # Si échec à cause de flags multi-GPU, réessayer sans
        if state.status == RunStatus.ERROR:
            if params.get("split_mode") and params.get("split_mode") != "none":
                err_msg = state.error_message
                logger.warning(f"Échec avec multi-GPU, tentative sans split: {err_msg[:100]}")
                fallback = {k: v for k, v in params.items() if k not in ("split_mode", "tensor_split", "main_gpu", "override_tensor")}
                fallback["no_kv_offload"] = False
                # Essayer avec ngl réduit progressivement (ngl//2, ngl//4, CPU)
                ngl_vals = [
                    max(1, (params.get("ngl", 99) or 99) // 2),
                    max(1, (params.get("ngl", 99) or 99) // 4),
                    0,  # CPU only en dernier recours
                ]
                for ngl_val in ngl_vals:
                    if ngl_val == (params.get("ngl", 99) or 99):
                        continue  # éviter de tester la même valeur
                    fallback["ngl"] = ngl_val
                    await asyncio.sleep(1)
                    state = await self._try_start(model_id, model_path, fallback)
                    if state.status != RunStatus.ERROR:
                        break

        return state

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

        if params.get("flash_attn"):
            val = params["flash_attn"]
            if val is True:
                cmd.append("--flash-attn")
                cmd.append("on")
            else:
                cmd.append("--flash-attn")
                cmd.append(str(val))

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

        logger.info(f"Démarrage llama-server: {' '.join(cmd[:6])}...")
        logger.debug(f"Commande complète: {' '.join(cmd)}")

        try:
            state.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            state.status = RunStatus.RUNNING

            # Attendre que llama-server soit prêt (poll health endpoint)
            await self._wait_for_server_ready(state)

            # Sauvegarder dans l'historique SQLite
            await save_run(state)

            # Lancer le monitoring en arrière-plan
            self._monitor_task = asyncio.create_task(self._monitor(state))

        except FileNotFoundError:
            state.status = RunStatus.ERROR
            state.error_message = f"llama-server introuvable: {binary}"
        except Exception as e:
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
                    stderr = ""
                    if state.process.stderr:
                        stderr = (await state.process.stderr.read()).decode(errors="replace")[-500:]
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
        stderr = ""
        if state.process and state.process.stderr:
            try:
                stderr = (await state.process.stderr.read()).decode(errors="replace")[-500:]
            except Exception:
                pass
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
        start_time = time.time()

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
                                elapsed = time.time() - start_time
                                speed = token_count / elapsed if elapsed > 0 else 0
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