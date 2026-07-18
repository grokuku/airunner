"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ─── System ────────────────────────────────────────────

class GPUInfo(BaseModel):
    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    vram_used_gb: float
    compute_cap: str = ""
    driver: str = ""


class RAMInfo(BaseModel):
    total_gb: float
    free_gb: float
    available_gb: float


class CPUInfo(BaseModel):
    cores: int
    threads: int
    model: str = ""


class SystemStatus(BaseModel):
    gpu: list[GPUInfo] = []
    ram: RAMInfo
    cpu: CPUInfo
    available: bool
    mode: str = "cpu"  # "cuda", "cpu", "mps"


# ─── Models ────────────────────────────────────────────

class ModelMeta(BaseModel):
    """Métadonnées parsées d'un fichier GGUF."""
    id: str  # filename without extension
    path: str
    file_size_gb: float
    architecture: str = ""
    name: str = ""
    quant: str = ""  # ex: "Q4_K_M"
    param_count: int = 0
    params_b: float = 0.0  # en milliards
    is_moe: bool = False
    expert_count: int = 0
    expert_used_count: int = 0
    active_params_b: float = 0.0
    block_count: int = 0
    context_length: int = 0
    embedding_length: int = 0
    head_count: int = 0
    loaded: bool = False


class HfSearchResult(BaseModel):
    repo_id: str
    name: str
    downloads: int = 0
    likes: int = 0
    files: list[dict] = []  # GGUF files disponibles


class HfDownloadRequest(BaseModel):
    repo_id: str
    filename: str


# ─── Config & Rules ────────────────────────────────────

class ServerConfigResponse(BaseModel):
    """Current server configuration (auth token masked)."""
    host: str
    port: int
    auth_token: str  # masked
    cors_origins: list[str]


class ServerConfigUpdate(BaseModel):
    """Partial update of server configuration.

    All fields are optional; only provided fields are updated.
    """
    cors_origins: Optional[list[str]] = None
    auth_token: Optional[str] = None


class ConfigRequest(BaseModel):
    model_id: str
    ctx_size: int = 8192
    quant_preference: Optional[str] = None  # "Q4_K_M", "Q8_0", etc.
    temp: float = 0.7


class VramEstimate(BaseModel):
    weights_gb: float
    cache_kv_gb: float
    overhead_gb: float = 0.3
    total_gb: float
    available_gb: float
    free_after_gb: float


class RamEstimate(BaseModel):
    weights_gb: float
    total_gb: float
    available_gb: float


class ConfigSuggestion(BaseModel):
    strategy: str  # "moe_offload", "dense_full", "dense_partial"
    quant: str
    quant_viable: list[str] = []
    vram: VramEstimate
    ram: Optional[RamEstimate] = None
    params: dict = {}
    estimated_speed: str = ""
    warnings: list[str] = []
    command_preview: str = ""


# ─── Chat ──────────────────────────────────────────────

class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class ChatRequest(BaseModel):
    model_id: str
    messages: list[Message]
    params: dict = {}
    stream: bool = True


class ChatResponse(BaseModel):
    content: str
    tokens: int = 0
    speed: float = 0.0
    elapsed_s: float = 0.0


class RunHistory(BaseModel):
    id: str
    model_id: str
    status: str
    tokens_generated: int = 0
    avg_speed: float = 0.0
    params: dict = {}
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
