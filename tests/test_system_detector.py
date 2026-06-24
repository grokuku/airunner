"""Tests pour le module system_detector."""

import sys
sys.path.insert(0, '.')

from app.core.system_detector import (
    detect, detect_gpu, detect_ram, detect_cpu,
    _count_cpu_cores, _count_cpu_threads, _get_cpu_model,
    _parse_meminfo_line, _mib_to_gb,
)


class TestMeminfo:
    def test_parse_meminfo_line(self):
        content = "MemTotal:       65746812 kB\nMemFree:        12345678 kB\nMemAvailable:   50000000 kB\n"
        assert _parse_meminfo_line(content, "MemTotal") == 65746812.0
        assert _parse_meminfo_line(content, "MemFree") == 12345678.0
        assert _parse_meminfo_line(content, "MemAvailable") == 50000000.0

    def test_parse_missing_key(self):
        content = "MemTotal: 1000 kB\n"
        assert _parse_meminfo_line(content, "MemFree") == 0.0


class TestConversion:
    def test_mib_to_gb(self):
        assert _mib_to_gb(1024) == 1.0
        assert _mib_to_gb(2048) == 2.0
        assert _mib_to_gb(0) == 0.0
        assert _mib_to_gb(512) == 0.5


class TestCPU:
    def test_count_cores(self):
        cores = _count_cpu_cores()
        assert cores >= 1, f"Devrait avoir au moins 1 core, eu {cores}"

    def test_count_threads(self):
        threads = _count_cpu_threads()
        assert threads >= 1, f"Devrait avoir au moins 1 thread, eu {threads}"

    def test_cpu_model(self):
        model = _get_cpu_model()
        assert isinstance(model, str)
        assert len(model) > 0, "Le modèle CPU ne devrait pas être vide"
