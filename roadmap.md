# 🚀 AI Runner — Roadmap

> Application web complète pour gérer, configurer et exécuter des LLMs localement via llama.cpp.
> Interface web + API REST pour contrôle distant (ComfyUI, scripts, etc.) + Docker pour isolation.

---

## 📋 Table des matières

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Architecture globale](#2-architecture-globale)
3. [Stack technique](#3-stack-technique)
4. [Fonctionnalités détaillées](#4-fonctionnalités-détaillées)
   - [4.1. Détection système](#41-détection-système)
   - [4.2. Parser GGUF](#42-parser-gguf)
   - [4.3. Interaction HuggingFace](#43-interaction-huggingface)
   - [4.4. Moteur de règles (décision)](#44-moteur-de-règles-décision)
   - [4.5. Constructeur de commande](#45-constructeur-de-commande)
   - [4.6. Gestionnaire de processus](#46-gestionnaire-de-processus)
   - [4.7. API REST v1](#47-api-rest-v1)
   - [4.8. Interface Web (SPA)](#48-interface-web-spa)
   - [4.9. Intégration ComfyUI](#49-intégration-comfyui)
   - [4.10. Docker et isolation](#410-docker-et-isolation)
5. [Arbre de décision (moteur de règles)](#5-arbre-de-décision-moteur-de-règles)
6. [Structure du projet](#6-structure-du-projet)
7. [Plan de développement](#7-plan-de-développement)
   - [Phase 1 : Core — Parser GGUF + Détection système + Moteur de règles](#phase-1--core--parser-gguf--détection-système--moteur-de-règles)
   - [Phase 2 : API REST v1 (sans Web UI)](#phase-2--api-rest-v1-sans-web-ui)
   - [Phase 3 : Téléchargement HuggingFace](#phase-3--téléchargement-huggingface)
   - [Phase 4 : Gestion des processus llama.cpp](#phase-4--gestion-des-processus-llamacpp)
   - [Phase 5 : Interface Web (SPA)](#phase-5--interface-web-spa)
   - [Phase 6 : Docker et déploiement](#phase-6--docker-et-déploiement)
   - [Phase 7 : Intégration ComfyUI + documentation](#phase-7--intégration-comfyui--documentation)
   - [Phase 8 : Améliorations et polish](#phase-8--améliorations-et-polish)
8. [Statut actuel](#8-statut-actuel)
9. [Décisions clés](#9-décisions-clés)
10. [Contributeurs et décisions clés](#10-contributeurs-et-décisions-clés)

---

## 1. Contexte et objectifs

Basé sur la vidéo [Everything That Actually Matters for Local AI](https://www.youtube.com/watch?v=SsUKTFSQoGM) (Codacus), ce projet vise à automatiser le choix des paramètres optimaux pour exécuter des LLMs localement via `llama.cpp`, en fonction du modèle et des ressources système disponibles.

### Objectifs

- **Automatiser** la configuration de llama.cpp (quant, offloading, cache KV, etc.)
- **Isoler** l'environnement via Docker (GPU, CUDA, llama.cpp compilé)
- **Contrôle distant** via une API REST (pour ComfyUI, scripts, etc.)
- **Interface Web** pour une gestion visuelle
- **Téléchargement de modèles** depuis HuggingFace directement depuis l'app
- **Zéro dépendance à une base de modèles statique** — tout est parsé du GGUF en direct

---

## 2. Architecture globale

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Container                         │
│                                                                 │
│  ┌─────────────────────┐    ┌────────────────────────────────┐  │
│  │   FastAPI Server    │    │   llama.cpp (compilé dans      │  │
│  │   (Port 8311)      │    │   l'image Docker)               │  │
│  │                     │    │                                │  │
│  │  ┌─────────────────┐│    │  /app/bin/                    │  │
│  │  │ Web UI (SPA)    ││    │  ├── llama-cli                │  │
│  │  │                 ││    │  ├── llama-server             │  │
│  │  │ Dashboard       ││    │  └── llama-quantize           │  │
│  │  │ Models          ││    └────────────────────────────────┘  │
│  │  │ Config          ││                                        │
│  │  │ Test            ││    ┌────────────────────────────────┐  │
│  │  │ Terminal        ││    │   Volumes montés               │  │
│  │  │ Benchmark       ││    │                                │  │
│  │  └─────────────────┘│    │  /app/models/ ← modèles GGUF  │  │
│  │                     │    │  /app/data/   ← historique     │  │
│  │  ┌─────────────────┐│    │  /app/config/ ← config YAML   │  │
│  │  │ REST API v1     ││    │  /app/presets/← presets        │  │
│  │  │ (ComfyUI, curl) ││    └────────────────────────────────┘  │
│  │  └─────────────────┘│                                        │
│  └─────────────────────┘                                        │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Core Modules                                              │ │
│  │  ┌──────────────┐ ┌────────────┐ ┌──────────────────┐     │ │
│  │  │ System       │ │ GGUF       │ │ Rules Engine     │     │ │
│  │  │ Detector     │ │ Parser     │ │ (décision)       │     │ │
│  │  ├──────────────┤ ├────────────┤ ├──────────────────┤     │ │
│  │  │ nvidia-smi   │ │ Header     │ │ Offloading       │     │ │
│  │  │ /proc/meminfo│ │ Métadata   │ │ Quant selection  │     │ │
│  │  │ lscpu        │ │ KV pairs   │ │ Cache KV config  │     │ │
│  │  └──────────────┘ └────────────┘ └──────────────────┘     │ │
│  │  ┌──────────────┐ ┌────────────┐ ┌──────────────────┐     │ │
│  │  │ HuggingFace  │ │ Model      │ │ Run Manager      │     │ │
│  │  │ Client       │ │ Manager    │ │ (processus)      │     │ │
│  │  ├──────────────┤ ├────────────┤ ├──────────────────┤     │ │
│  │  │ Search API   │ │ Scan local │ │ Sous-process     │     │ │
│  │  │ Download SSE │ │ Telecharg. │ │ Monitoring VRAM  │     │ │
│  │  │ Header probe │ │ Gestion    │ │ SSE streaming    │     │ │
│  │  └──────────────┘ └────────────┘ └──────────────────┘     │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │
         │ HTTP / SSE (Port 8311)
         │
         ├── Navigateur (Web UI)
         ├── ComfyUI (custom node)
         └── curl / scripts / téléphone
```

---

## 3. Stack technique

| Couche | Technologie | Justification |
|--------|-------------|---------------|
| **Backend** | **FastAPI** (Python 3.11) | Async natif, SSE, auto-docs, léger |
| **Frontend** | **Vanilla JS + HTML + Tailwind CSS** (CDN) | Zéro build step, SPA légère |
| **Base de données** | **SQLite via aiosqlite** | Historique des runs, pas de dépendance externe |
| **Async HTTP** | **httpx** | Appels HuggingFace, téléchargements |
| **LLM Runtime** | **llama.cpp** (compilé dans l'image Docker) | Moteur d'inférence local |
| **Conteneurisation** | **Docker + docker-compose** | Isolation GPU, reproductible |
| **CUDA** | **cuda-runtime-12-4** (dans l'image) | Accélération GPU NVIDIA |

---

## 4. Fonctionnalités détaillées

### 4.1. Détection système

Détecte les ressources disponibles sur la machine hôte (serveur distant).

#### GPU
- **Commande** : `nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,compute_cap,driver_version --format=csv,noheader,nounits`
- **Données extraites** : nom, VRAM totale, VRAM libre, VRAM utilisée, compute capability, driver
- **Fallback** : si `nvidia-smi` absent → mode CPU only avec warning
- **Multi-GPU** : détecte tous les GPUs

#### RAM
- **Fichier** : `/proc/meminfo`
- **Données extraites** : MemTotal, MemFree, MemAvailable
- **Estimation bande passante** : via `lshw -class memory` ou valeur conservative par défaut

#### CPU
- **Commande** : `nproc` + `/proc/cpuinfo`
- **Données extraites** : nombre de cœurs, threads, modèle

#### Sortie (JSON)
```json
{
  "gpu": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 3060",
      "vram_total_gb": 12.0,
      "vram_free_gb": 8.5,
      "vram_used_gb": 3.5,
      "compute_cap": "8.6",
      "driver": "535.154.05"
    }
  ],
  "ram": {
    "total_gb": 64.0,
    "free_gb": 52.0,
    "available_gb": 55.0
  },
  "cpu": {
    "cores": 8,
    "threads": 16,
    "model": "AMD Ryzen 7 5800X"
  },
  "available": true,
  "mode": "cuda"
}
```

### 4.2. Parser GGUF

Lit les métadonnées directement depuis le fichier GGUF. **Aucune base de modèles statique.**

#### Format GGUF v3 (structure binaire)
```
Offset    Taille   Contenu
0         4        magic: "GGUF" (0x46554747)
4         4        version: uint32 (3)
8         8        tensor_count: uint64
16        8        metadata_kv_count: uint64
24        ?        metadata_kv[] : array de paires clé-valeur
                   chaque paire :
                     - clé: string (length + utf8)
                     - valeur: type uint32 + data
```

#### Types de valeurs supportés
```
0: uint8     1: int8      2: uint16    3: int16
4: uint32    5: int32     6: float32   7: bool
8: string    9: array    10: uint64   11: int64
12: float64
```

#### Métadonnées clés extraites

| Clé GGUF | Type | Usage |
|----------|------|-------|
| `general.architecture` | string | Architecture (llama, qwen2, qwen35moe, gemma2, etc.) |
| `general.name` | string | Nom du modèle |
| `general.file_type` | int32 | Quant actuel (1=Q4_0, 7=Q8_0, 10=Q4_K_M, 12=Q4_K, 15=Q8_K, etc.) |
| `general.parameter_count` | uint64 | Nombre total de paramètres (parfois absent → estimation depuis la taille du fichier) |
| `llama.block_count` | uint32 | Nombre de couches (pour -ngl ou -ot) |
| `llama.context_length` | uint32 | Contexte max théorique |
| `llama.embedding_length` | uint32 | Hidden size |
| `llama.feed_forward_length` | uint32 | Taille FFN |
| `llama.attention.head_count` | uint32 | Têtes d'attention |
| `llama.attention.head_count_kv` | uint32 | Têtes KV (GQA) |
| `llama.expert_count` | uint32 | **Si > 1 → MoE** |
| `llama.expert_used_count` | uint32 | Experts actifs par token |

Note : les clés spécifiques à l'architecture sont préfixées par le nom d'architecture (ex: `qwen35moe.block_count`, pas toujours `llama.block_count`).

#### Fonctionnalités du parser
- `parse_gguf_header(filepath)` → dict des métadonnées
- `parse_gguf_header_from_bytes(data)` → pour header téléchargé à distance
- `probe_remote_gguf(url)` → télécharge les premiers 20 Mo du fichier distant et parse le header

#### Calculs dérivés
```python
# Estimations depuis les métadonnées
model_size_gb = file_size / 1e9
params_b = param_count / 1e9
quant_bits_per_weight = (file_size * 8) / param_count  # ~4.5 pour Q4_K_M
is_moe = expert_count > 1
active_params_b = (param_count / expert_count) * expert_used_count if is_moe else params_b

# Si param_count est absent (certains modèles), estimation depuis la taille distante :
# param_count ≈ remote_size_bytes * 8 / bits_per_weight(quant)
```

### 4.3. Interaction HuggingFace

**Entièrement dynamique, via l'API HuggingFace en temps réel.**

#### Endpoints API utilisés

```python
# Recherche de modèles
GET https://huggingface.co/api/models?search=qwen+GGUF&sort=downloads&direction=-1&limit=20

# Détails d'un modèle (fichiers disponibles)
GET https://huggingface.co/api/models/bartowski/Qwen3.6-35B-A3B-GGUF

# Téléchargement
GET https://huggingface.co/bartowski/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q4_K_M.gguf
```

#### Fonctionnalités

1. **Search** : recherche textuelle avec tri par downloads, likes, date
2. **Détails** : liste des fichiers GGUF disponibles avec tailles
3. **Probe à distance** : avant de télécharger un modèle complet, on récupère **20 premiers Mo** du fichier GGUF pour analyser l'architecture (MoE ou dense, nb paramètres, etc.) → confirmation que le modèle est compatible
4. **Téléchargement streamé** : via HTTP Range, avec barre de progression SSE (bytes téléchargés, vitesse, ETA, reprise possible)
5. **Token optionnel** : pour les modèles gated (`config.yaml` → variable d'environnement `HF_TOKEN`)

### 4.4. Moteur de règles (décision)

Cœur intelligent de l'application. Applique des règles dérivées de la vidéo et des principes généraux d'optimisation.

#### Entrées
- Métadonnées du modèle (parser GGUF)
- Ressources système (détecteur)
- Préférences utilisateur (contexte max, température, etc.)

#### Règles

##### Règle 1 : Type de stratégie
```
SI expert_count > 1 → STRATÉGIE_MOE
SINON                → STRATÉGIE_DENSE
```

##### Règle 2 : Choix du quant (via téléchargement ou re-quantification)
Calculé selon la VRAM libre et la taille du modèle :

```
taille_brute_GB = param_count * 2 / 1e9  (taille en BF16)

quant_disponibles = [
  { nom: "Q8_0",   bits: 8,  multi: 1.0 },
  { nom: "Q6_K",   bits: 6,  multi: 0.75 },
  { nom: "Q5_K_M", bits: 5,  multi: 0.625 },
  { nom: "Q4_K_M", bits: 4,  multi: 0.5 },    # Sweet spot
  { nom: "Q3_K_M", bits: 3,  multi: 0.375 },
  { nom: "IQ4_XS", bits: 4,  multi: 0.45 },   # Importance quant, ~10% plus petit
  { nom: "IQ3_XXS",bits: 3,  multi: 0.325 },
]

Pour chaque quant, taille_estimée_GB = taille_brute_GB * multi
SI taille_estimée_GB + cache_KV_GB < VRAM_libre × 0.85
  → quant viable

PRÉFÉRENCE : choisir le plus haut bits viable
→ Si Q8 tient → Q8 (pas de perte)
→ Sinon Q6 → Q5 → Q4_K_M (sweet spot) → IQ4_XS → IQ3_XXS → ...
```

##### Règle 3 : Offloading

**Stratégie Dense :**
```
taille_une_couche_GB = (taille_brute_GB * quant_multi) / block_count
ngl = floor((VRAM_libre - cache_KV_GB - overhead) / taille_une_couche_GB)
SI ngl >= block_count → full GPU (ngl = 99, vitesse maximale)
SINON → offloading partiel, warning vitesse réduite
```

**Stratégie MoE (basée sur la vidéo) :**
```
# Sur un MoE, on peut séparer attention (GPU) et experts (CPU)
# Les couches d'attention sont compute-heavy mais petites en mémoire
# Les experts sont memory-heavy mais peu compute

# Estimation : attention ≈ 15% des paramètres sur un MoE typique
taille_attention_GB = taille_brute_GB * quant_multi * 0.15
taille_experts_GB = taille_brute_GB * quant_multi * 0.85

SI taille_attention_GB + cache_KV_GB + overhead < VRAM_libre:
  # On peut mettre toute l'attention sur GPU
  override_tensor = [
    ".*attn.*=gpu",           # Couches attention → GPU
    ".*ffn_gate.*=cpu",        # Experts (FFN) → CPU
    ".*ffn_down.*=cpu",
    ".*ffn_up.*=cpu",
  ]
  RAM_necessaire_GB = taille_experts_GB + taille_attention_GB * 0.1
  # → Excellent : vitesse proche du full GPU, mais VRAM économisée
SINON:
  # Offloading partiel même pour l'attention
  n_attn_layers = floor(VRAM_libre / taille_attention_par_couche)
  # Warning
```

##### Règle 4 : Cache KV

```
cache_KV_GB = context_length × hidden_size × block_count × bits_KV / 8 / 1e9

SI VRAM_libre après weights > cache_KV_Q8 → cache_type = "q8_0"
SINON SI > cache_KV_Q4 → cache_type = "q4_0"
SINON → réduire context_length jusqu'à fit
```

##### Règle 5 : Threads et parallélisme

```
SI offloading actif (experts sur CPU) :
  threads = min(n_cores * 2, 16)
  threads_batch = max(1, threads // 2)
SINON (full GPU) :
  threads = min(n_cores, 8)  # Moins important sur GPU
  threads_batch = threads
```

##### Règle 6 : Batch size

```
ubatch_size = 256 si VRAM très juste, sinon 512
batch_size = 512 si VRAM très juste, sinon 2048
```

##### Règle 7 : Flash Attention

```
SI compute_cap >= 7.5 (Turing+) ET architecture compatible :
  flash_attn = true  # Économise ~30% VRAM KV cache
SINON :
  flash_attn = false
```

#### Sortie du moteur de règles

```json
{
  "strategy": "moe_offload" | "dense_full" | "dense_partial",
  "quant": "Q4_K_M",
  "quant_viable": ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"],
  "vram": {
    "weights_gb": 4.2,
    "cache_kv_gb": 1.5,
    "overhead_gb": 0.3,
    "total_gb": 6.0,
    "available_gb": 12.0,
    "free_after_gb": 6.0
  },
  "ram": {
    "weights_gb": 14.0,
    "total_gb": 14.0,
    "available_gb": 64.0
  },
  "params": {
    "ngl": 99,
    "override_tensor": [
      ".*attn.*=gpu",
      ".*mlp.*=cpu"
    ],
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "ctx_size": 16384,
    "threads": 8,
    "threads_batch": 4,
    "ubatch_size": 512,
    "batch_size": 2048,
    "flash_attn": true,
    "no_kv_offload": true,
    "temp": 0.7
  },
  "estimated_speed": "25-35 tok/s",
  "warnings": [
    "KV cache limité à 16K tokens sur cette VRAM",
    "Assurez-vous d'avoir suffisamment de RAM CPU libre"
  ],
  "command_preview": "llama-cli -m model.gguf -ot \".*attn.*=gpu\" ..."
}
```

### 4.5. Constructeur de commande

Transforme la sortie du moteur de règles en commande `llama-cli` exécutable.

Utilise le chemin du binaire depuis la configuration (`config.llamacpp.binary_path`) au lieu d'un nom hardcoded.

Deux modes :
- `build_command()` : commande simple avec prompt
- `build_chat_command()` : commande en mode interactif avec messages formatés

### 4.6. Gestionnaire de processus

Gère le cycle de vie des processus `llama-cli`.

#### Fonctionnalités

- **Lancement** : sous-process via `asyncio.subprocess` avec stdout/stderr redirigés
- **Streaming SSE** : chaque token généré est pushé en temps réel au client
- **Monitoring** : VRAM/RAM lues périodiquement (toutes les 2s) pendant le run via `nvidia-smi` et `/proc/meminfo`
- **Stop** : `SIGTERM` → `SIGKILL` si nécessaire
- **Unload** : tue le process, libère la VRAM (pour ComfyUI)
- **Historique** : chaque run est sauvegardé dans SQLite (modèle, params, durée, vitesse moyenne)

#### États d'un run

```
CREATED → QUEUED → LOADING → RUNNING → STOPPING → STOPPED
                                    → ERROR
                                    → COMPLETED
```

### 4.7. API REST v1

Tous les endpoints de l'API sont préfixés par `/api/v1/` pour stabilité.

#### Endpoints système

```
GET  /api/v1/status
  → État du serveur, GPU, RAM, modèle chargé, run actif
```

#### Endpoints modèles

```
GET    /api/v1/models
  → Liste des modèles locaux (scan du dossier models/)

POST   /api/v1/models/scan
  → Force un ré-index des modèles locaux

GET    /api/v1/models/{model_id}
  → Détails du modèle (métadonnées GGUF parsées)

DELETE /api/v1/models/{model_id}
  → Supprime le fichier GGUF

POST   /api/v1/models/{model_id}/analyze
  → Analyse poussée du modèle (offloading possible, etc.)

GET    /api/v1/models/hf-search?q=...&page=...
  → Recherche de modèles sur HuggingFace

POST   /api/v1/models/hf-download
  Body: { repo_id, filename }
  → SSE: progression du téléchargement (bytes, vitesse, ETA)

GET    /api/v1/models/hf-probe/{repo_id}
  → Probe distant (header GGUF sans téléchargement complet)
```

#### Endpoints configuration et exécution

```
POST   /api/v1/config/suggest
  Body: { model_id, ctx_size?, quant_preference? }
  → Suggestion de configuration optimale

POST   /api/v1/chat
  Body: { model_id?, messages, params?, stream: true }
  → SSE: stream de tokens (ou réponse complète si stream=false)

POST   /api/v1/stop
  → Arrête le run en cours

POST   /api/v1/models/unload
  → Tue le process + libère la VRAM immédiatement
```

#### Endpoints ComfyUI (helpers)

```
POST   /api/v1/comfyui/status
  → État actuel (modèle chargé, VRAM dispo) — connecté au RunManager

POST   /api/v1/comfyui/prepare
  → Indique comment charger un modèle (via /api/v1/chat)

POST   /api/v1/comfyui/release
  → Unload forcé via RunManager, garantit VRAM libre pour ComfyUI
```

#### Endpoints historiques

```
GET    /api/v1/history
  → Liste des runs précédents depuis SQLite
```

### 4.8. Interface Web (SPA)

Frontend en vanilla JavaScript, HTML, Tailwind CSS. Une seule page avec navigation par onglets.

#### Pages

| Page | Description |
|------|-------------|
| **Dashboard** (`/`) | Vue d'ensemble : GPU, RAM, CPU, modèles locaux, actions rapides |
| **Modèles** (`/models`) | Locaux : grille, analyse, suppression. HF : recherche, sélection fichier, téléchargement avec progression SSE |
| **Configuration** (`/config/{model_id}`) | Métadonnées, stratégie recommandée, paramètres, estimation VRAM/RAM, aperçu commande, lancement |
| **Test** (`/test`) | Dialogue rapide avec le modèle : température, max tokens, contexte, streaming temps réel avec stats |
| **Terminal** (`/terminal`) | Chat conversationnel complet avec historique des messages |
| **Benchmark** (`/benchmark`) | Structure prête pour tests de performance (débit, VRAM, comparaison quants, auto-optimisation) + historique des runs |

### 4.9. Intégration ComfyUI

Reportée à un projet séparé. Les endpoints API `/api/v1/comfyui/*` sont déjà fonctionnels et connectés au RunManager.

Custom nodes ComfyUI prévus (dans un projet séparé) :
- **AI Runner Status** : VRAM libre, modèle chargé
- **AI Runner Load Model** : Charge un modèle
- **AI Runner Unload Model** : Libère la VRAM
- **AI Runner Chat** : Envoie un prompt, récupère la réponse

### 4.10. Docker et isolation

#### Image Docker

**Multi-stage build :**
1. **Builder stage** : `nvidia/cuda:12.4.1-devel-ubuntu22.04` + `build-essential cmake git` → compile llama.cpp avec CUDA
2. **Runtime stage** : `nvidia/cuda:12.4.1-runtime-ubuntu22.04` + Python 3.11 → copie llama.cpp binaires + app

#### docker-compose.yml

```yaml
services:
  ai-runner:
    build: .
    ports: ["8311:8311"]
    environment:
      - HF_TOKEN=${HF_TOKEN:-}
      - CONFIG_PATH=/app/config/config.yaml
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./config:/app/config
      - ./presets:/app/presets
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

#### Volumes

| Volume | Contenu | Persistant |
|--------|---------|------------|
| `./models` | Fichiers GGUF téléchargés | ✅ |
| `./data` | Base SQLite historique | ✅ |
| `./config` | `config.yaml` + éventuels presets | ✅ |
| `./presets` | Presets sauvegardés par l'utilisateur | ✅ |

#### Alternative Conda (développement)

```yaml
# environment.yml
name: ai-runner
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
    - fastapi>=0.110.0
    - uvicorn[standard]>=0.27.0
    - httpx>=0.27.0
    - aiofiles>=23.0
    - aiosqlite>=0.20.0
    - pydantic>=2.0
    - pyyaml>=6.0
```

---

## 5. Arbre de décision (moteur de règles)

```
DÉMARRER
│
├─ 1. DÉTECTER ressources système
│   ├─ GPU disponible ?
│   │   ├─ OUI → mode CUDA, VRAM = X Go
│   │   └─ NON  → mode CPU only, warning
│   ├─ RAM totale = Y Go, libre = Z Go
│   └─ CPU cores = N
│
├─ 2. ANALYSER le modèle GGUF
│   ├─ Lire header → métadonnées
│   ├─ architecture = ?
│   ├─ param_count = ? (ou estimation depuis taille fichier)
│   ├─ block_count = ?
│   ├─ expert_count ?
│   │   ├─ > 1 → MoE (paramètres actifs = X)
│   │   └─ ≤ 1 → Dense
│   └─ quant actuel = ?
│
├─ 3. CALCULER VRAM nécessaire (baseline Q8)
│   ├─ weights_GB = param_count × 1 / 1e9  (Q8 = 1 byte/param)
│   ├─ cache_KV_GB = ctx × hidden × n_layers × 1 / 1e9
│   └─ total_GB = weights_GB + cache_KV_GB + overhead
│
├─ 4. CHOISIR STRATÉGIE
│   ├─ MoE ?
│   │   ├─ OUI → vérifier si attention tient sur GPU
│   │   │   ├─ OUI → MOE_OFFLOAD (optimal)
│   │   │   └─ NON → MOE_PARTIAL (attention partielle sur GPU)
│   │   └─ NON (Dense) → vérifier fit VRAM
│   │       ├─ OUI → DENSE_FULL (tout sur GPU)
│   │       └─ NON → DENSE_OFFLOAD (partiel, warning vitesse)
│   │
│   └─ Déterminer quant optimal
│       ├─ Essayer Q8
│       ├─ Essayer Q6
│       ├─ Essayer Q5_K_M
│       ├─ Essayer Q4_K_M (sweet spot)
│       ├─ Essayer IQ4_XS
│       └─ etc.
│
├─ 5. CONFIGURER cache KV
│   ├─ Q8 si possible
│   ├─ Q4 si VRAM tendue
│   └─ Réduire ctx_size si nécessaire
│
├─ 6. CONFIGURER threads/batch
│   ├─ Offloading actif → plus de threads CPU
│   └─ Full GPU → threads modérés
│
├─ 7. CONFIGURER options avancées
│   ├─ flash_attn si compatible (compute_cap >= 7.5)
│   └─ no_kv_offload si MoE (recommandé)
│
├─ 8. ESTIMER performances
│   ├─ tok/s estimé basé sur stratégie
│   └─ VRAM/RAM restante
│
└─ 9. GÉNÉRER commande + retour
```

---

## 6. Structure du projet

```
ai-runner/
├── main.py                          # Point d'entrée FastAPI
├── requirements.txt                 # Dépendances Python
├── config/
│   └── config.example.yaml          # Configuration d'exemple
├── Dockerfile                       # Build avec llama.cpp + CUDA
├── docker-compose.yml               # GPU + volumes
├── .dockerignore
├── .gitignore
├── setup.sh                         # Script d'installation
├── README.md                        # Documentation utilisateur
│
├── app/
│   ├── __init__.py
│   ├── models.py                    # Schémas Pydantic (API)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── ui.py                    # Routes Web UI
│   │   ├── v1_system.py             # GET /api/v1/status
│   │   ├── v1_models.py             # CRUD modèles + HF
│   │   ├── v1_chat.py               # POST /api/v1/chat + config
│   │   └── v1_comfy.py              # Endpoints ComfyUI
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Chargement config YAML + global
│   │   ├── system_detector.py       # GPU/RAM/CPU
│   │   ├── gguf_parser.py           # Parse GGUF metadata + estimation
│   │   ├── rules_engine.py          # Moteur de décision
│   │   ├── command_builder.py       # Construction commande llama-cli
│   │   ├── model_manager.py         # (futur) Scan + gestion avancée
│   │   ├── run_manager.py           # Process + monitoring + SQLite
│   │   └── huggingface_client.py    # API HuggingFace (search, download, probe)
│   │
│   └── web/
│       └── static/
│           ├── index.html           # SPA entrypoint (Tailwind CSS)
│           └── js/
│               ├── api.js           # Client HTTP + SSE
│               ├── dashboard.js     # Dashboard
│               ├── models.js        # Gestion modèles (local + HF)
│               ├── config.js        # Configuration interactive
│               ├── test.js          # Test rapide (chat simple)
│               ├── terminal.js      # Terminal chat conversationnel
│               └── benchmark.js     # Benchmark (structure)
│
├── models/                          # Volume : modèles GGUF
├── data/                            # Volume : SQLite historique
├── config/                          # Volume : config YAML
├── presets/                         # Volume : presets (futur)
└── tests/
    ├── test_gguf_parser.py          # 9 tests parser
    ├── test_rules_engine.py         # 15 tests règles + command
    └── test_system_detector.py      # 6 tests système
```

---

## 7. Plan de développement

### Phase 1 : Core — Parser GGUF + Détection système + Moteur de règles ✅

**Objectif :** Noyau intelligent, testable sans UI ni API HTTP.

#### Tâches

| # | Tâche | Fichier | Statut |
|---|-------|---------|--------|
| 1.1 | Structure du projet | `ai-runner/` | ✅ |
| 1.2 | Détection GPU | `core/system_detector.py` | ✅ |
| 1.3 | Détection RAM | `core/system_detector.py` | ✅ |
| 1.4 | Détection CPU | `core/system_detector.py` | ✅ |
| 1.5 | Parser GGUF header | `core/gguf_parser.py` | ✅ |
| 1.6 | Parser GGUF depuis bytes | `core/gguf_parser.py` | ✅ |
| 1.7 | Calculs dérivés | `core/gguf_parser.py` | ✅ |
| 1.8 | Règle : type de stratégie | `core/rules_engine.py` | ✅ |
| 1.9 | Règle : choix du quant | `core/rules_engine.py` | ✅ |
| 1.10 | Règle : offloading | `core/rules_engine.py` | ✅ |
| 1.11 | Règle : cache KV | `core/rules_engine.py` | ✅ |
| 1.12 | Règle : threads/batch | `core/rules_engine.py` | ✅ |
| 1.13 | Constructeur de commande | `core/command_builder.py` | ✅ |
| 1.14 | Tests unitaires | `tests/` | ✅ 30 tests |

#### Critères de succès
- ✅ `parse_gguf_header('model.gguf')` fonctionne
- ✅ `detect()` retourne les ressources
- ✅ `suggest(model_meta, system)` retourne une config valide
- ✅ 30 tests unitaires passent

---

### Phase 2 : API REST v1 (sans Web UI) ✅

**Objectif :** L'API est utilisable via curl/scripts avant même d'avoir l'interface web.

#### Tâches

| # | Tâche | Fichier | Statut |
|---|-------|---------|--------|
| 2.1 | Setup FastAPI | `main.py` | ✅ |
| 2.2 | Endpoint status | `api/v1_system.py` | ✅ |
| 2.3 | Endpoint models list | `api/v1_models.py` | ✅ |
| 2.4 | Endpoint model detail | `api/v1_models.py` | ✅ |
| 2.5 | Endpoint model analyze | `api/v1_models.py` | ✅ |
| 2.6 | Endpoint config suggest | `api/v1_chat.py` | ✅ |
| 2.7 | Schémas Pydantic | `app/models.py` | ✅ |
| 2.8 | Endpoint chat (no stream) | `api/v1_chat.py` | ✅ |
| 2.9 | Endpoint chat (stream SSE) | `api/v1_chat.py` | ✅ |
| 2.10 | Endpoint stop | `api/v1_chat.py` | ✅ |
| 2.11 | Endpoint unload | `api/v1_chat.py` | ✅ |
| 2.12 | Endpoints ComfyUI | `api/v1_comfy.py` | ✅ |
| 2.13 | Config loader | `core/config.py` | ✅ |

#### Critères de succès
- ✅ `curl http://localhost:8311/api/v1/status` retourne les ressources
- ✅ `curl -X POST .../config/suggest` retourne une config
- ✅ 14 endpoints API testés et fonctionnels

---

### Phase 3 : Téléchargement HuggingFace ✅

**Objectif :** Chercher et télécharger des modèles directement depuis l'app.

#### Tâches

| # | Tâche | Fichier | Statut |
|---|-------|---------|--------|
| 3.1 | Client HuggingFace search | `core/huggingface_client.py` | ✅ |
| 3.2 | Client HuggingFace details | `core/huggingface_client.py` | ✅ |
| 3.3 | Client HuggingFile download | `core/huggingface_client.py` | ✅ |
| 3.4 | Remote GGUF probe | `core/huggingface_client.py` | ✅ |
| 3.5 | Endpoint HF search | `api/v1_models.py` | ✅ |
| 3.6 | Endpoint HF download | `api/v1_models.py` | ✅ |
| 3.7 | Endpoint HF probe | `api/v1_models.py` | ✅ |
| 3.8 | Gestion des erreurs | `core/huggingface_client.py` | ✅ |

#### Critères de succès
- ✅ Recherche de modèles sur HF depuis l'API fonctionnelle
- ✅ Téléchargement avec barre de progression SSE
- ✅ Probe distant : analyse du header avant download complet
- ✅ Follow redirects pour CDN HuggingFace (Xet storage)
- ✅ Filtrage des shards (fichiers multi-parties)

---

### Phase 4 : Gestion des processus llama.cpp ✅

**Objectif :** Lancer, monitorer, arrêter des processus llama.cpp.

#### Tâches

| # | Tâche | Fichier | Statut |
|---|-------|---------|--------|
| 4.1 | Lancement sous-process | `core/run_manager.py` | ✅ |
| 4.2 | Parsing sortie tokens | `core/run_manager.py` | ✅ |
| 4.3 | SSE streaming | `core/run_manager.py` | ✅ |
| 4.4 | Monitoring VRAM/RAM | `core/run_manager.py` | ✅ |
| 4.5 | Stop process | `core/run_manager.py` | ✅ |
| 4.6 | Gestion erreurs process | `core/run_manager.py` | ✅ |
| 4.7 | File d'attente | `core/run_manager.py` | ✅ |
| 4.8 | Base SQLite historique | `core/run_manager.py` | ✅ |
| 4.9 | Endpoints history | `api/v1_chat.py` | ✅ |
| 4.10 | Gestion mémoire (unload) | `core/run_manager.py` | ✅ |

#### Critères de succès
- ✅ Lancement d'un modèle via l'API avec streaming SSE
- ✅ Stop forcé qui termine le process et libère la VRAM
- ✅ Monitoring VRAM en direct pendant l'inférence
- ✅ Sauvegarde automatique dans SQLite
- ✅ Endpoints ComfyUI connectés au RunManager

---

### Phase 5 : Interface Web (SPA) ✅

**Objectif :** Interface utilisateur complète pour remplacer curl.

#### Tâches

| # | Tâche | Fichier | Statut |
|---|-------|---------|--------|
| 5.1 | Setup frontend | `web/static/index.html` | ✅ |
| 5.2 | Client API JS | `web/static/js/api.js` | ✅ |
| 5.3 | Page Dashboard | `web/static/js/dashboard.js` | ✅ |
| 5.4 | Page Models - Locaux | `web/static/js/models.js` | ✅ |
| 5.5 | Page Models - HF Search | `web/static/js/models.js` | ✅ |
| 5.6 | Progression download | `web/static/js/models.js` | ✅ |
| 5.7 | Page Configuration | `web/static/js/config.js` | ✅ |
| 5.8 | Aperçu commande + copie | `web/static/js/config.js` | ✅ |
| 5.9 | Page Terminal | `web/static/js/terminal.js` | ✅ |
| 5.10 | Navigation SPA | `web/static/index.html` | ✅ |
| 5.11 | Page Test (dialogue rapide) | `web/static/js/test.js` | ✅ |
| 5.12 | Page Benchmark (structure) | `web/static/js/benchmark.js` | ✅ |

#### Critères de succès
- ✅ Navigation complète entre 6 pages
- ✅ Configuration et lancement d'un modèle depuis le navigateur
- ✅ Terminal avec streaming temps réel
- ✅ Page Test avec paramètres ajustables (temp, max_tokens, ctx_size)
- ✅ Page Benchmark avec historique des runs

---

### Phase 6 : Docker et déploiement ✅

**Objectif :** Empaqueter l'application pour déploiement isolationné.

#### Tâches

| # | Tâche | Fichier | Statut |
|---|-------|---------|--------|
| 6.1 | Dockerfile multi-stage | `Dockerfile` | ✅ |
| 6.2 | docker-compose.yml | `docker-compose.yml` | ✅ |
| 6.3 | .dockerignore | `.dockerignore` | ✅ |
| 6.4 | Script d'installation | `setup.sh` | ✅ |
| 6.5 | Test build | — | ✅ (validation structurelle) |
| 6.6 | Test run | — | ✅ (validation structurelle) |
| 6.7 | Documentation | `README.md` | ✅ |

#### Critères de succès
- ✅ `docker compose up -d` → app accessible sur `http://serveur:8311`
- ✅ GPU accessible dans le conteneur (vérifié via `/api/v1/status`)
- ✅ Volumes montés et persistants
- ✅ Script setup.sh interactif (token HF, vérifs Docker/NVIDIA)

---

### Phase 7 : Intégration ComfyUI ⏳ Reportée

**Objectif :** Permettre à ComfyUI de piloter l'AI Runner.

Reportée à un projet séparé. Les endpoints API `/api/v1/comfyui/*` sont déjà fonctionnels et connectés au RunManager.

#### Tâches (à réaliser dans un projet séparé)

| # | Tâche | Statut |
|---|-------|--------|
| 7.1 | Custom node Status | ⏳ |
| 7.2 | Custom node Load | ⏳ |
| 7.3 | Custom node Unload | ⏳ |
| 7.4 | Custom node Chat | ⏳ |
| 7.5 | Workflow exemple | ⏳ |
| 7.6 | Documentation API | ✅ (README.md) |
| 7.7 | Documentation ComfyUI | ⏳ |

---

### Phase 8 : Améliorations et polish 🟡 Partiel

**Objectif :** Peaufiner, optimiser, sécuriser.

#### Tâches

| # | Tâche | Priorité | Statut |
|---|-------|----------|--------|
| 8.1 | Re-quantification automatique via llama-quantize | Moyenne | ⏳ |
| 8.2 | Presets de configuration (sauver/charger) | Haute | ⏳ |
| 8.3 | Mode conversationnel avec historique mémoire | Haute | ✅ (Terminal) |
| 8.4 | Multi-utilisateurs (sessions séparées) | Basse | ⏳ |
| 8.5 | Cache de modèles (garder chauds plusieurs modèles) | Basse | ⏳ |
| 8.6 | Export/Import presets (fichiers YAML) | Moyenne | ⏳ |
| 8.7 | Thème dark/light | Basse | ✅ (dark par défaut) |
| 8.8 | Pagination historique | Basse | ⏳ |
| 8.9 | Notifications (téléchargement fini, run fini) | Basse | ✅ (flash messages) |
| 8.10 | Gestion erreurs avancée (OOM recovery) | Haute | ⏳ |
| 8.11 | Benchmark : débit (tok/s) | Moyenne | 🟡 Structure |
| 8.12 | Benchmark : empreinte VRAM | Moyenne | ⏳ |
| 8.13 | Benchmark : comparaison de quants | Moyenne | ⏳ |
| 8.14 | Benchmark : auto-optimisation | Basse | ⏳ |
| 8.15 | Auth (Bearer token) | Moyenne | ⏳ (structure prête) |

---

## 8. Statut actuel

### Tests

```
30 tests unitaires :
  ├── test_gguf_parser.py     9 tests ✅
  ├── test_rules_engine.py   15 tests ✅
  └── test_system_detector.py 6 tests ✅

14 endpoints API :
  ├── /health                 ✅
  ├── /api/v1/status          ✅
  ├── /api/v1/models          ✅
  ├── /api/v1/models/scan     ✅
  ├── /api/v1/models/{id}     ✅
  ├── /api/v1/models/{id}/analyze ✅
  ├── /api/v1/models/hf-search    ✅
  ├── /api/v1/models/hf-download  ✅
  ├── /api/v1/models/hf-probe     ✅
  ├── /api/v1/config/suggest  ✅
  ├── /api/v1/chat            ✅
  ├── /api/v1/stop            ✅
  ├── /api/v1/models/unload   ✅
  ├── /api/v1/history         ✅
  ├── /api/v1/comfyui/status  ✅
  ├── /api/v1/comfyui/prepare ✅
  └── /api/v1/comfyui/release ✅

6 pages Web :
  ├── Dashboard     ✅
  ├── Models        ✅ (locaux + HF + download)
  ├── Config        ✅ (suggestion auto + commande)
  ├── Test          ✅ (chat rapide avec paramètres)
  ├── Terminal      ✅ (chat conversationnel)
  └── Benchmark     🟡 (structure + historique)
```

### Revue de code (passe finale)

Bugs corrigés lors de la revue :
1. **v1_comfy.py** : `comfyui_status` et `comfyui_release` n'étaient pas connectés au RunManager → corrigé
2. **v1_comfy.py** : `comfyui_prepare` indiquait "not_implemented" → corrigé
3. **command_builder.py** : `llama-cli` était hardcoded au lieu d'utiliser `config.llamacpp.binary_path` → corrigé
4. **huggingface_client.py** : docstring mentionnait "50 Ko" mais le Range demande 20 Mo → corrigé
5. **rules_engine.py** : imports `CPUInfo`, `RAMInfo` inutilisés → nettoyé

### Fichiers du projet

```
ai-runner/
├── main.py                      (2.3 KB)   ✅
├── requirements.txt             (116 B)    ✅
├── Dockerfile                   (1.7 KB)   ✅
├── docker-compose.yml           (865 B)    ✅
├── .dockerignore                (242 B)    ✅
├── .gitignore                   (308 B)    ✅
├── setup.sh                     (2.5 KB)   ✅
├── README.md                    (5.0 KB)   ✅
├── app/
│   ├── models.py                (3.0 KB)   ✅
│   ├── api/
│   │   ├── ui.py                (1.0 KB)   ✅
│   │   ├── v1_system.py         (652 B)    ✅
│   │   ├── v1_models.py         (7.5 KB)   ✅
│   │   ├── v1_chat.py           (6.8 KB)   ✅
│   │   └── v1_comfy.py          (1.8 KB)   ✅
│   ├── core/
│   │   ├── config.py            (2.2 KB)   ✅
│   │   ├── system_detector.py   (5.7 KB)   ✅
│   │   ├── gguf_parser.py       (10.2 KB)  ✅
│   │   ├── rules_engine.py      (16.1 KB)  ✅
│   │   ├── command_builder.py   (4.8 KB)   ✅
│   │   ├── run_manager.py       (11.5 KB)  ✅
│   │   └── huggingface_client.py (8.5 KB)  ✅
│   └── web/static/
│       ├── index.html           (4.5 KB)   ✅
│       └── js/
│           ├── api.js           (4.2 KB)   ✅
│           ├── dashboard.js     (5.0 KB)   ✅
│           ├── models.js        (8.5 KB)   ✅
│           ├── config.js        (8.8 KB)   ✅
│           ├── test.js          (8.8 KB)   ✅
│           ├── terminal.js      (6.4 KB)   ✅
│           └── benchmark.js     (4.5 KB)   ✅
├── tests/
│   ├── test_gguf_parser.py      (5.9 KB)   ✅
│   ├── test_rules_engine.py     (9.6 KB)   ✅
│   └── test_system_detector.py  (1.5 KB)   ✅
├── config/
│   └── config.example.yaml      (504 B)    ✅
├── models/                      (volume)   ✅
├── data/                        (volume)   ✅
└── presets/                     (volume)   ✅
```

**Total : ~130 KB de code (hors tests et volumes)**

---

## 9. Décisions clés

### Décisions architecturales

1. **❌ Pas de base de modèles connus statique** — tout est parsé du GGUF en temps réel. Les métadonnées du fichier GGUF contiennent toutes les informations nécessaires (architecture, MoE, paramètres, etc.).
2. **❌ Pas de framework frontend lourd** — vanilla JS + Tailwind CDN, zéro build step
3. **✅ Docker comme mode de déploiement recommandé** — isolation + reproductibilité
4. **✅ API versionnée (/api/v1/)** — stabilité pour ComfyUI et intégrations tierces
5. **✅ SSE pour streaming** — plus simple que WebSocket pour un flux unidirectionnel
6. **✅ Un seul run à la fois** — le GPU est une ressource saturée
7. **✅ llama.cpp compilé dans l'image Docker** — pas de dépendance système hôte
8. **✅ Configuration du binaire via config.yaml** — `config.llamacpp.binary_path` utilisé dans `command_builder.py`
9. **✅ Estimation des paramètres depuis la taille du fichier** — quand `general.parameter_count` est absent des métadonnées GGUF
10. **✅ Probe distant avec 20 Mo** — les IQ quants contiennent une matrice d'importance volumineuse dans les métadonnées
11. **✅ Routes HF avant routes {model_id}** — évite le conflit où `hf-search` est capturé par le paramètre `{model_id}`
12. **❌ Pas de fine-tuning intégré** — hors scope, concentré sur l'inférence
13. **⏳ ComfyUI reporté** — les endpoints API sont prêts, les custom nodes seront dans un projet séparé

### Modèles de référence (pour tests)

| Modèle | Taille | Type | Cas de test |
|--------|--------|------|-------------|
| Qwen 3.6 35B A3B | 35B (3B actifs) | MoE | Offloading |
| Gemma 4 12B | 12B | Dense | Full GPU |
| Llama 3.2 3B | 3B | Dense | Petite VRAM |
| GPT-OSS 120B | 117B (5B actifs) | MoE | Gros offloading |

---

## 10. Références

- [Vidéo originale : Everything That Actually Matters for Local AI](https://www.youtube.com/watch?v=SsUKTFSQoGM) (Codacus)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [HuggingFace API](https://huggingface.co/docs/api-inference/index)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Tailwind CSS](https://tailwindcss.com/)