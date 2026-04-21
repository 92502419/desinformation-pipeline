#!/bin/bash
# scripts/start.sh — Démarrage complet du pipeline de désinformation
# KOMOSSI Sosso — Master 2 BIG DATA IA — UCAO-UUT 2025-2026
# Usage : bash scripts/start.sh
# ─────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "  Pipeline Désinformation — KOMOSSI Sosso — Master 2 BIG DATA IA"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── 0. Mémoire disponible ───────────────────────────────────────
echo ""
echo "[0/6] Ressources système..."
MEM_FREE=$(free -m | awk '/^Mem:/{print $7}')
echo "   RAM disponible : ${MEM_FREE} MB"
if [ "$MEM_FREE" -lt 3000 ]; then
    echo "   ⚠  AVERTISSEMENT : RAM faible (< 3GB libre). Fermer les applications en cours."
fi

# ── 1. Vérifications préliminaires ─────────────────────────────
echo ""
echo "[1/6] Vérifications..."
command -v docker  >/dev/null || { echo "ERREUR : Docker non installé"; exit 1; }
command -v python3 >/dev/null || { echo "ERREUR : Python 3 non installé"; exit 1; }

if [ ! -f "models/onnx/model_quantized.onnx" ]; then
    echo "ERREUR : Modèle ONNX introuvable (models/onnx/model_quantized.onnx)"
    echo "Exécuter d'abord :"
    echo "  source venv_main/bin/activate"
    echo "  bash scripts/download_datasets.sh"
    echo "  python scripts/preprocess_data.py"
    echo "  python scripts/train_model.py"
    echo "  python scripts/export_onnx.py"
    exit 1
fi
ONNX_SIZE=$(du -sh models/onnx/model_quantized.onnx | cut -f1)
echo "   OK Modèle ONNX trouvé (${ONNX_SIZE})"

if [ ! -f "models/pretrained/config.json" ]; then
    echo "ERREUR : Modèle pretrained introuvable (models/pretrained/)"
    exit 1
fi
echo "   OK Modèle pretrained trouvé"

# ── 2. Création des répertoires nécessaires ─────────────────────
echo ""
echo "[2/6] Création des répertoires..."
mkdir -p logs checkpoints models/checkpoints
echo "   OK Répertoires créés"

# ── 3. Nettoyage des anciens containers si nécessaire ───────────
echo ""
echo "[3/6] Nettoyage des anciens containers..."
# Arrêter et supprimer les containers avec --remove-orphans pour nettoyer les orphelins
docker compose down --remove-orphans 2>/dev/null || true
echo "   OK Containers précédents supprimés"

# ── 4. Démarrage Docker Compose ─────────────────────────────────
echo ""
echo "[4/6] Démarrage des services Docker (mode optimisé mémoire)..."
docker compose up -d --remove-orphans
echo "   OK Services lancés"

# ── 5. Attente de la santé des services ─────────────────────────
echo ""
echo "[5/6] Attente de la disponibilité des services (180s max)..."
MAX_WAIT=180
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(curl -sf http://localhost:8000/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "démarrage")
    if [ "$STATUS" = "ok" ]; then
        echo "   OK API FastAPI opérationnelle (${ELAPSED}s)"
        break
    fi
    printf "   ... (%ds) API: %s\r" $ELAPSED "$STATUS"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo ""
    echo "   ATTENTION Timeout — vérifier : docker compose logs api"
fi

# ── 6. Résumé ───────────────────────────────────────────────────
echo ""
echo "[6/6] Etat des containers :"
docker compose ps
echo ""
echo "============================================================"
echo "  Pipeline opérationnel !"
echo "============================================================"
echo ""
echo "  Interface             URL                          Accès"
echo "  ─────────────────────────────────────────────────────────"
echo "  Grafana (dashboards)  http://localhost:3000        admin / admin2025"
echo "  FastAPI (docs)        http://localhost:8000/docs   —"
echo "  Kafdrop (Kafka UI)    http://localhost:9000        —"
echo "  Elasticsearch         http://localhost:9200        —"
echo ""
echo "  Santé   : curl http://localhost:8000/health"
echo "  Logs    : docker compose logs -f spark-app"
echo "  Arrêt   : docker compose down"
echo ""
