"""Tests pour le module gguf_parser."""

import sys
import struct
import io
import os
import tempfile
sys.path.insert(0, '.')

from app.core.gguf_parser import (
    parse_gguf_header, parse_gguf_header_from_bytes,
    metadata_to_model_meta, GGUF_MAGIC,
)


def build_gguf(metadata_pairs):
    """Build a valid GGUF v3 buffer."""
    buf = io.BytesIO()
    buf.write(GGUF_MAGIC)
    buf.write(struct.pack('<I', 3))   # version
    buf.write(struct.pack('<Q', 0))   # tensor_count
    buf.write(struct.pack('<Q', len(metadata_pairs)))  # metadata_count

    for key, value in metadata_pairs:
        key_bytes = key.encode('utf-8')
        buf.write(struct.pack('<Q', len(key_bytes)))
        buf.write(key_bytes)

        if isinstance(value, str):
            buf.write(struct.pack('<I', 8))  # STRING
            val_bytes = value.encode('utf-8')
            buf.write(struct.pack('<Q', len(val_bytes)))
            buf.write(val_bytes)
        elif isinstance(value, int):
            if 0 <= value <= 0xFFFFFFFF:
                buf.write(struct.pack('<I', 4))  # UINT32
                buf.write(struct.pack('<I', value))
            else:
                buf.write(struct.pack('<I', 10))  # UINT64
                buf.write(struct.pack('<Q', value))
        elif isinstance(value, bool):
            buf.write(struct.pack('<I', 7))  # BOOL
            buf.write(struct.pack('B', 1 if value else 0))
        elif isinstance(value, float):
            buf.write(struct.pack('<I', 6))  # FLOAT32
            buf.write(struct.pack('<f', value))
        else:
            raise ValueError(f"Type non supporté: {type(value)}")

    return buf.getvalue()


class TestParseFromBytes:
    def test_basic_moe_model(self):
        data = build_gguf([
            ("general.architecture", "qwen2"),
            ("general.name", "Qwen 3.6 35B A3B"),
            ("general.file_type", 15),
            ("general.parameter_count", 35000000000),
            ("qwen2.block_count", 64),
            ("qwen2.expert_count", 12),
            ("qwen2.expert_used_count", 2),
        ])
        meta = parse_gguf_header_from_bytes(data)
        assert meta["general.architecture"] == "qwen2"
        assert meta["general.parameter_count"] == 35000000000
        assert meta["qwen2.expert_count"] == 12

    def test_dense_model(self):
        data = build_gguf([
            ("general.architecture", "gemma2"),
            ("general.name", "Gemma 4 12B"),
            ("general.file_type", 12),
            ("general.parameter_count", 12000000000),
            ("gemma2.block_count", 40),
        ])
        meta = parse_gguf_header_from_bytes(data)
        assert meta["general.architecture"] == "gemma2"
        assert meta["general.parameter_count"] == 12000000000

    def test_invalid_magic(self):
        data = build_gguf([("general.architecture", "test")])
        bad = b'XXXX' + data[4:]
        try:
            parse_gguf_header_from_bytes(bad)
            assert False, "Devrait lever ValueError"
        except ValueError as e:
            assert "invalide" in str(e)

    def test_unsupported_version(self):
        data = bytearray(build_gguf([]))
        struct.pack_into('<I', data, 4, 99)
        try:
            parse_gguf_header_from_bytes(bytes(data))
            assert False
        except ValueError as e:
            assert "non supportée" in str(e)

    def test_truncated_data(self):
        data = build_gguf([("general.architecture", "test")])
        try:
            parse_gguf_header_from_bytes(data[:10])
            assert False
        except ValueError as e:
            assert "trop court" in str(e)


class TestParseFromFile:
    def test_parse_file(self):
        data = build_gguf([
            ("general.architecture", "llama"),
            ("general.name", "Test Model"),
            ("general.file_type", 10),
            ("general.parameter_count", 8000000000),
            ("llama.block_count", 32),
        ])
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.gguf')
        tmp.write(data)
        tmp_path = tmp.name
        tmp.close()

        try:
            meta = parse_gguf_header(tmp_path)
            assert meta["general.architecture"] == "llama"
            assert meta["general.parameter_count"] == 8000000000
            assert meta["file_size"] > 0  # Taille en bytes (le fichier est petit, ~200 octets)
        finally:
            os.unlink(tmp_path)

    def test_file_not_found(self):
        try:
            parse_gguf_header("/tmp/nonexistent.gguf")
            assert False
        except FileNotFoundError:
            pass


class TestMetadataToModelMeta:
    def test_moe_conversion(self):
        data = build_gguf([
            ("general.architecture", "qwen2"),
            ("general.name", "Qwen 35B"),
            ("general.file_type", 15),
            ("general.parameter_count", 35000000000),
            ("qwen2.block_count", 64),
            ("qwen2.expert_count", 12),
            ("qwen2.expert_used_count", 2),
        ])
        meta = parse_gguf_header_from_bytes(data)
        model = metadata_to_model_meta(meta, model_id="qwen-test")
        assert model.is_moe
        assert model.params_b == 35.0
        assert model.expert_count == 12
        assert model.expert_used_count == 2
        assert model.active_params_b < model.params_b  # MoE a moins d'actifs

    def test_dense_conversion(self):
        data = build_gguf([
            ("general.architecture", "gemma2"),
            ("general.name", "Gemma 12B"),
            ("general.file_type", 12),
            ("general.parameter_count", 12000000000),
            ("gemma2.block_count", 40),
        ])
        meta = parse_gguf_header_from_bytes(data)
        model = metadata_to_model_meta(meta, model_id="gemma-test")
        assert not model.is_moe
        assert model.params_b == 12.0
        assert model.active_params_b == 12.0  # Dense: actifs = total
