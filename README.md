# 🚀 AI Runner

> Gestion complète de LLMs en local via llama.cpp — Web UI + API REST + Docker.

```
┌─────────────────────────────────────────────────────────┐
│                    AI Runner                             │
│                                                         │
│  🌐 Web UI          📡 API REST        🐳 Docker       │
│  Dashboard          /api/v1/status     GPU + CUDA      │
│  Models (local+HF)  /api/v1/models     Volumes persist │
│  Config auto        /api/v1/chat       Isolation       │
│  Terminal chat      /api/v1/config     Compilé from src│
│                     /api/v1/comfyui                     │
└─────────────────────────────────────────────────────────┘
```

## ⚡ Démarrage rapide

```bash
# 1. Cloner et lancer
git clone <url> && cd ai-runner
chmod +x setup.sh && ./setup.sh

# 2. Démarrer
docker compose up -d

# 3. Ouvrir le navigateur
open http://localhost:8311
```

## 🧠 Fonctionnalités

### Configuration intelligente
Le moteur de règles analyse automatiquement :
- Les métadonnées du modèle GGUF (architecture, MoE/dense, paramètres)
- Les ressources système (VRAM, RAM, CPU)
- Applique les règles optimales de la [vidéo Codacus](https://www.youtube.com/watch?v=SsUKTFSQoGM)

### Stratégies supportées
| Stratégie | Quand ? | Résultat |
|-----------|---------|----------|
| **Full GPU** | Modèle dense tient en VRAM | Performances maximales |
| **MoE Offload** | Modèle MoE (experts → CPU, attention → GPU) | ~30 tok/s sur RTX 3060 |
| **Dense Offload** | Modèle dense trop grand | Utilisable mais ralenti |
| **CPU Only** | Pas de GPU | Lent mais fonctionnel |

### Modèles
- Recherche et téléchargement depuis HuggingFace
- Analyse à distance (probe du header avant download)
- Barre de progression en temps réel (SSE)
- Scan des modèles locaux

### API REST
- Versionnée (`/api/v1/`)
- Contrôle distant pour ComfyUI, scripts, etc.
- Streaming SSE pour l'inférence en temps réel

## 📋 Prérequis

- **Docker** et **docker-compose**
- **NVIDIA Container Toolkit** (pour GPU)
- **Espace disque** : 10-50 Go selon les modèles

## 🚀 Déploiement

```bash
# Installation complète
./setup.sh

# Ou manuellement :
docker compose build
docker compose up -d

# Avec token HuggingFace (pour modèles gated)
export HF_TOKEN="hf_..."
docker compose up -d

# Logs
docker compose logs -f
```

## 🖥️ Interface Web

| Page | Description |
|------|-------------|
| **Dashboard** | Vue d'ensemble GPU/RAM/CPU, modèles locaux |
| **Modèles** | Gestion locale + recherche/téléchargement HF |
| **Configuration** | Suggestion automatique + paramètres modifiables |
| **Terminal** | Chat en streaming avec monitoring temps réel |

## 📡 API

```bash
# État du système
curl http://serveur:8311/api/v1/status

# Lister les modèles
curl http://serveur:8311/api/v1/models

# Suggestion de configuration
curl -X POST http://serveur:8311/api/v1/config/suggest \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "mon-modele", "ctx_size": 8192}'

# Chat (streaming SSE)
curl -X POST http://serveur:8311/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "mon-modele", "messages": [{"role": "user", "content": "Bonjour !"}], "stream": true}'

# Libérer la VRAM (avant ComfyUI)
curl -X POST http://serveur:8311/api/v1/models/unload
```

## 🧩 Intégration ComfyUI

Des custom nodes ComfyUI sont disponibles (Phase 7 - en cours) :
- **AI Runner Status** : VRAM libre, modèle chargé
- **AI Runner Load Model** : Charge un modèle
- **AI Runner Unload Model** : Libère la VRAM
- **AI Runner Chat** : Prompt → réponse

## 📁 Structure

```
ai-runner/
├── main.py              # Point d'entrée FastAPI
├── Dockerfile           # Build avec llama.cpp + CUDA
├── docker-compose.yml   # GPU + volumes
├── app/
│   ├── api/             # Routes API REST
│   ├── core/            # Moteur de règles, parser GGUF
│   └── web/static/      # Interface Web (SPA)
├── models/              # Modèles GGUF (volume)
├── data/                # Base SQLite (volume)
└── config/              # Configuration (volume)
```

## 🔧 Configuration

```yaml
# config/config.yaml
server:
  port: 8311
  host: "0.0.0.0"
  # auth_token: "mon-token"  # Bearer token optionnel

storage:
  models_dir: "/app/models"
  data_dir: "/app/data"
  presets_dir: "/app/presets"

huggingface:
  token: ""  # Ou via HF_TOKEN env var

llamacpp:
  binary_path: "/app/bin/llama-cli"
  default_temp: 0.7
  default_ctx_size: 8192
```

## 🛣️ Roadmap

- [x] **Phase 1** — Core : parser GGUF + règles + détection système
- [x] **Phase 2** — API REST v1
- [x] **Phase 3** — Téléchargement HuggingFace
- [x] **Phase 4** — Gestion des processus llama.cpp
- [x] **Phase 5** — Interface Web (SPA)
- [x] **Phase 6** — Docker + déploiement
- [ ] **Phase 7** — Intégration ComfyUI
- [ ] **Phase 8** — Presets, optimisations avancées

## 📜 Licence

MIT

---

*Fait avec ❤️ en s'inspirant de [Codacus](https://www.youtube.com/watch?v=SsUKTFSQoGM) et de toute la communauté local AI.*
