# ─── AI Runner — Dockerfile ──────────────────────────
#
# llama.cpp est téléchargé par l'application au premier lancement
# via POST /api/v1/llamacpp/update.
#
# L'image de base nvidia/cuda fournit les libs CUDA runtime
# (libcudart.so, libcublas.so, etc.) nécessaires pour que
# libggml-cuda.so puisse utiliser le GPU.

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/app/config/config.yaml
ENV LD_LIBRARY_PATH=/app/llama-bin:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu

# Python + dépendances système
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier l'application
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt && \
    pip3 install --no-cache-dir --break-system-packages httpx aiofiles aiosqlite

COPY . .

# Créer les dossiers de volumes (llama-bin pour les binaires téléchargés)
RUN mkdir -p /app/models /app/data /app/config /app/presets /app/llama-bin

EXPOSE 8311

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8311/health || exit 1

CMD ["python3", "main.py"]