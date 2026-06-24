"""Tests pour le module rules_engine et command_builder."""

import sys
sys.path.insert(0, '.')

from app.core.rules_engine import (
    suggest, estimate_model_size, estimate_kv_cache_gb,
    _get_bits_from_quant,
)
from app.core.command_builder import build_command, build_chat_command
from app.models import ModelMeta, SystemStatus, GPUInfo, RAMInfo, CPUInfo, ConfigRequest


# ─── Fixtures ──────────────────────────────────────────

RTX3060_SYSTEM = SystemStatus(
    gpu=[GPUInfo(
        index=0, name="NVIDIA GeForce RTX 3060",
        vram_total_gb=12.0, vram_free_gb=10.5, vram_used_gb=1.5,
        compute_cap="8.6", driver="535.154.05",
    )],
    ram=RAMInfo(total_gb=64.0, free_gb=52.0, available_gb=55.0),
    cpu=CPUInfo(cores=8, threads=16, model="AMD Ryzen 7"),
    available=True, mode="cuda",
)

CPU_ONLY_SYSTEM = SystemStatus(
    gpu=[],
    ram=RAMInfo(total_gb=32.0, free_gb=24.0, available_gb=28.0),
    cpu=CPUInfo(cores=4, threads=8, model="Intel i5"),
    available=True, mode="cpu",
)

QWEN_MOE = ModelMeta(
    id="qwen-35b", path="/models/qwen-35b.gguf", file_size_gb=18.5,
    architecture="qwen2", name="Qwen 3.6 35B A3B", quant="Q4_K_M",
    param_count=35000000000, params_b=35.0,
    is_moe=True, expert_count=12, expert_used_count=2, active_params_b=8.75,
    block_count=64, context_length=131072, embedding_length=5120, head_count=40,
)

GEMMA_DENSE = ModelMeta(
    id="gemma-12b", path="/models/gemma-12b.gguf", file_size_gb=12.0,
    architecture="gemma2", name="Gemma 4 12B", quant="Q8_0",
    param_count=12000000000, params_b=12.0,
    is_moe=False, expert_count=0, expert_used_count=0, active_params_b=12.0,
    block_count=40, context_length=8192, embedding_length=4096, head_count=32,
)

LLAMA_SMALL = ModelMeta(
    id="llama-3b", path="/models/llama-3b.gguf", file_size_gb=2.0,
    architecture="llama", name="Llama 3.2 3B", quant="Q8_0",
    param_count=3000000000, params_b=3.0,
    is_moe=False, expert_count=0, expert_used_count=0, active_params_b=3.0,
    block_count=28, context_length=8192, embedding_length=2048, head_count=16,
)


# ─── Tests ─────────────────────────────────────────────


class TestEstimations:
    def test_model_size(self):
        assert abs(estimate_model_size(35.0, 4.5) - 19.69) < 0.01
        assert abs(estimate_model_size(12.0, 8.0) - 12.0) < 0.01
        assert abs(estimate_model_size(3.0, 8.0) - 3.0) < 0.01

    def test_kv_cache(self):
        assert abs(estimate_kv_cache_gb(8192, 5120, 64, 1.0) - 0.67) < 0.01
        assert abs(estimate_kv_cache_gb(8192, 4096, 40, 1.0) - 0.34) < 0.01
        assert abs(estimate_kv_cache_gb(8192, 2048, 28, 1.0) - 0.12) < 0.01

    def test_get_bits(self):
        assert _get_bits_from_quant("Q8_0") == 8.0
        assert _get_bits_from_quant("Q4_K_M") == 4.5
        assert _get_bits_from_quant("Q2_K") == 2.5
        assert _get_bits_from_quant("UNKNOWN") == 4.5  # fallback


class TestMoEOffload:
    """Tests pour la stratégie MoE avec offloading (scénario principal de la vidéo)."""

    def test_moe_on_rtx3060(self):
        """Qwen 35B MoE doit utiliser moe_offload sur une RTX 3060 12GB."""
        result = suggest(QWEN_MOE, RTX3060_SYSTEM)
        assert result.strategy == "moe_offload", f"Attendu moe_offload, eu {result.strategy}"
        assert result.quant == "Q4_K_M"
        assert len(result.params["override_tensor"]) > 0
        assert "no_kv_offload" in result.params
        assert result.params["no_kv_offload"] is True
        # VRAM devrait être faible (attention seulement)
        assert result.vram.weights_gb < 5.0, f"VRAM weights trop élevé: {result.vram.weights_gb}"
        # RAM devrait être utilisée pour les experts
        assert result.ram is not None
        assert result.ram.weights_gb > 10.0, f"RAM weights trop faible: {result.ram.weights_gb}"

    def test_moe_with_preferences(self):
        """Les préférences utilisateur doivent être respectées si viables."""
        req = ConfigRequest(model_id="qwen-35b", ctx_size=16384, temp=0.3)
        result = suggest(QWEN_MOE, RTX3060_SYSTEM, req)
        assert result.params["ctx_size"] == 16384
        assert result.params["temp"] == 0.3

    def test_moe_quant_fallback(self):
        """Q8 doit être rejeté si trop gros, fallback vers Q4_K_M."""
        req = ConfigRequest(model_id="qwen-35b", quant_preference="Q8_0")
        result = suggest(QWEN_MOE, RTX3060_SYSTEM, req)
        # Q8 pour 35B = 35Go → ne tient pas → fallback
        assert result.quant != "Q8_0", "Q8 ne devrait pas être viable"
        assert result.quant == "Q4_K_M"


class TestDenseFull:
    """Tests pour les modèles denses qui tiennent en VRAM."""

    def test_gemma_12b_on_rtx3060(self):
        """Gemma 4 12B doit tenir en full GPU."""
        result = suggest(GEMMA_DENSE, RTX3060_SYSTEM)
        assert result.strategy == "dense_full"
        # Pas d'override tensor pour dense
        assert len(result.params["override_tensor"]) == 0
        # VRAM totale raisonnable
        assert result.vram.total_gb < 10.0
        assert result.params["ngl"] == 99

    def test_small_model_full_gpu(self):
        """Un petit modèle 3B doit tenir facilement."""
        result = suggest(LLAMA_SMALL, RTX3060_SYSTEM)
        assert result.strategy == "dense_full"
        # Doit pouvoir utiliser Q8
        assert "Q8_0" in result.quant_viable


class TestCPUOnly:
    """Tests pour le mode sans GPU."""

    def test_cpu_strategy(self):
        """Sans GPU, doit retourner dense_partial avec ngl=0."""
        result = suggest(GEMMA_DENSE, CPU_ONLY_SYSTEM)
        assert result.strategy == "dense_partial"
        assert result.params["ngl"] == 0
        assert any("Aucun GPU" in w for w in result.warnings)

    def test_cpu_speed_estimate(self):
        """L'estimation de vitesse CPU doit être conservative."""
        result = suggest(GEMMA_DENSE, CPU_ONLY_SYSTEM)
        assert "tok/s" in result.estimated_speed


class TestCommandBuilder:
    """Tests pour le constructeur de commande."""

    def test_build_simple(self):
        """Construction de commande basique."""
        params = {
            "ngl": 99,
            "override_tensor": [],
            "cache_type_k": "q8_0",
            "ctx_size": 8192,
            "threads": 8,
            "threads_batch": 4,
            "ubatch_size": 512,
            "batch_size": 2048,
            "flash_attn": True,
            "no_kv_offload": False,
            "temp": 0.7,
        }
        cmd = build_command("/models/test.gguf", params)
        assert "llama-cli" in cmd
        assert "-m /models/test.gguf" in cmd
        assert "-ngl 99" in cmd
        assert "--cache-type-k q8_0" in cmd
        assert "--ctx-size 8192" in cmd
        assert "--flash-attn" in cmd

    def test_build_with_override(self):
        """Commande avec override tensor (MoE)."""
        params = {
            "ngl": 99,
            "override_tensor": [
                ".*attn.*=gpu",
                ".*ffn_gate.*=cpu",
            ],
            "cache_type_k": "q8_0",
            "ctx_size": 8192,
            "threads": 16,
            "threads_batch": 8,
            "ubatch_size": 512,
            "batch_size": 2048,
            "flash_attn": True,
            "no_kv_offload": True,
            "temp": 0.7,
        }
        cmd = build_command("/models/qwen.gguf", params)
        assert '--override-tensor ".*attn.*=gpu"' in cmd
        assert '--override-tensor ".*ffn_gate.*=cpu"' in cmd
        assert "--no-kv-offload" in cmd

    def test_build_chat(self):
        """Construction de commande chat."""
        params = {
            "ngl": 99,
            "override_tensor": [".*attn.*=gpu"],
            "cache_type_k": "q8_0",
            "ctx_size": 8192,
            "threads": 8,
            "threads_batch": 4,
            "ubatch_size": 512,
            "batch_size": 2048,
            "flash_attn": True,
            "no_kv_offload": True,
            "temp": 0.7,
        }
        messages = [
            {"role": "system", "content": "Assistant utile."},
            {"role": "user", "content": "Bonjour !"},
        ]
        cmd = build_chat_command("/models/test.gguf", params, messages)
        assert "--jinja" in cmd
        # Plus de --interactive : on veut une réponse unique
        assert "--interactive" not in cmd
        assert "Assistant utile." in cmd
        assert "Bonjour !" in cmd


class TestEdgeCases:
    """Tests de cas limites."""

    def test_zero_layers(self):
        """Un modèle sans block_count ne doit pas crasher."""
        model = ModelMeta(
            id="unknown", path="/models/x.gguf", file_size_gb=1.0,
            architecture="unknown", name="Unknown",
            quant="Q4_K_M", param_count=1000000000, params_b=1.0,
            is_moe=False, expert_count=0, expert_used_count=0,
            active_params_b=1.0,
            block_count=0, context_length=0, embedding_length=0, head_count=0,
        )
        result = suggest(model, RTX3060_SYSTEM)
        assert result.strategy in ("dense_full", "dense_partial")

    def test_zero_vram(self):
        """VRAM à zéro ne doit pas crasher (cas limite)."""
        no_vram_system = SystemStatus(
            gpu=[GPUInfo(
                index=0, name="GPU", vram_total_gb=0.0, vram_free_gb=0.0,
                vram_used_gb=0.0, compute_cap="", driver="",
            )],
            ram=RAMInfo(total_gb=16.0, free_gb=8.0, available_gb=10.0),
            cpu=CPUInfo(cores=4, threads=8, model="CPU"),
            available=True, mode="cuda",
        )
        result = suggest(GEMMA_DENSE, no_vram_system)
        assert result.strategy == "dense_partial"
        assert result.params["ngl"] >= 0
