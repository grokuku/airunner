#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────
# AI Runner — Script d'installation et de démarrage
# ──────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 AI Runner — Installation"
echo "=========================="
echo ""

# Vérifier Docker
if ! command -v docker &>/dev/null; then
    echo "❌ Docker n'est pas installé."
    echo "   Installez-le d'abord : https://docs.docker.com/engine/install/"
    exit 1
fi

# Vérifier docker-compose
if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null 2>&1; then
    echo "❌ docker-compose n'est pas installé."
    exit 1
fi

# Vérifier NVIDIA Container Toolkit
if command -v nvidia-smi &>/dev/null; then
    echo "✅ GPU NVIDIA détecté"
    if ! docker info 2>/dev/null | grep -q nvidia; then
        echo "⚠️  Le NVIDIA Container Toolkit n'est pas installé."
        echo "   Les GPUs ne seront pas accessibles dans le conteneur."
        echo "   Installez-le : https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        echo ""
    fi
fi

# Créer les dossiers de volumes
mkdir -p models data config presets

# Créer config.yaml si inexistant
if [ ! -f config/config.yaml ]; then
    cp config/config.example.yaml config/config.yaml
    echo "✅ config/config.yaml créé depuis l'exemple"
fi

# Configurer le token HuggingFace optionnel
if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "🔑 Token HuggingFace (optionnel)"
    echo "   Requis pour les modèles gated (Llama, Gemma, etc.)"
    read -rp "   Token HF (laisser vide pour passer) : " hf_token
    if [ -n "$hf_token" ]; then
        export HF_TOKEN="$hf_token"
        echo "   Token enregistré dans la session"
    fi
fi

echo ""
echo "🐳 Build de l'image Docker..."
echo "   (Premier build : 5-15 minutes selon votre connexion et CPU)"
echo ""

# Build l'image
docker compose build

echo ""
echo "✅ Build terminé !"
echo ""
echo "📋 Pour démarrer :"
echo "   cd $SCRIPT_DIR"
echo "   docker compose up -d"
echo ""
echo "🌐 Interface : http://localhost:8311"
echo "📡 API       : http://localhost:8311/api/v1/status"
echo ""
echo "📂 Modèles   : $SCRIPT_DIR/models/"
echo "📊 Données   : $SCRIPT_DIR/data/"
echo ""
echo "💡 Pour utiliser un token HF :"
echo "   export HF_TOKEN='votre_token' && docker compose up -d"
echo ""
echo "🛑 Pour arrêter :"
echo "   docker compose down"
