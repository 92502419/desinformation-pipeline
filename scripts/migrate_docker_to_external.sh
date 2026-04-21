#!/bin/bash
# scripts/migrate_docker_to_external.sh
# Déplace les données Docker (images, containers, volumes) vers le disque externe
# pour libérer de l'espace sur le disque système et éviter de saturer la RAM du SSD.
#
# PRÉREQUIS : exécuter avec sudo
# Usage : sudo bash scripts/migrate_docker_to_external.sh
# ─────────────────────────────────────────────────────────────────────

EXTERNAL_DRIVE="/media/florent/Disque local2"
DOCKER_NEW_ROOT="${EXTERNAL_DRIVE}/docker-data"
DAEMON_JSON="/etc/docker/daemon.json"

echo "============================================================"
echo "  Migration Docker vers disque externe"
echo "  Destination : ${DOCKER_NEW_ROOT}"
echo "============================================================"
echo ""

# Vérifier qu'on tourne en root
if [ "$(id -u)" -ne 0 ]; then
    echo "ERREUR : Ce script doit être exécuté avec sudo"
    echo "Usage  : sudo bash scripts/migrate_docker_to_external.sh"
    exit 1
fi

# Vérifier que le disque externe est monté
if [ ! -d "${EXTERNAL_DRIVE}" ]; then
    echo "ERREUR : Disque externe non monté : ${EXTERNAL_DRIVE}"
    exit 1
fi

# Espace disponible
SPACE=$(df -BG "${EXTERNAL_DRIVE}" | awk 'NR==2{print $4}' | tr -d 'G')
echo "Espace disponible sur disque externe : ${SPACE}GB"
if [ "${SPACE}" -lt 20 ]; then
    echo "ERREUR : Espace insuffisant (< 20GB)"
    exit 1
fi

echo "[1/5] Arrêt du service Docker..."
systemctl stop docker
systemctl stop docker.socket
echo "   OK Docker arrêté"

echo ""
echo "[2/5] Création du répertoire de destination..."
mkdir -p "${DOCKER_NEW_ROOT}"
chmod 711 "${DOCKER_NEW_ROOT}"
echo "   OK Répertoire créé : ${DOCKER_NEW_ROOT}"

echo ""
echo "[3/5] Migration des données Docker (peut prendre plusieurs minutes)..."
DOCKER_SIZE=$(du -sh /var/lib/docker 2>/dev/null | cut -f1 || echo "inconnu")
echo "   Taille à migrer : ${DOCKER_SIZE}"
rsync -aP --delete /var/lib/docker/ "${DOCKER_NEW_ROOT}/"
echo "   OK Migration terminée"

echo ""
echo "[4/5] Mise à jour de la configuration Docker daemon..."
# Sauvegarder l'ancien daemon.json s'il existe
if [ -f "${DAEMON_JSON}" ]; then
    cp "${DAEMON_JSON}" "${DAEMON_JSON}.bak"
    echo "   OK Ancien daemon.json sauvegardé en ${DAEMON_JSON}.bak"
fi

# Écrire le nouveau daemon.json
cat > "${DAEMON_JSON}" << EOF
{
  "data-root": "${DOCKER_NEW_ROOT}",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF
echo "   OK daemon.json mis à jour"
echo "   data-root = ${DOCKER_NEW_ROOT}"

echo ""
echo "[5/5] Redémarrage de Docker..."
systemctl start docker
sleep 3
if systemctl is-active --quiet docker; then
    echo "   OK Docker redémarré avec succès"
else
    echo "   ERREUR Docker n'a pas démarré — restauration..."
    if [ -f "${DAEMON_JSON}.bak" ]; then
        cp "${DAEMON_JSON}.bak" "${DAEMON_JSON}"
    else
        rm -f "${DAEMON_JSON}"
    fi
    systemctl start docker
    echo "   Configuration restaurée. Vérifier les logs : journalctl -u docker"
    exit 1
fi

echo ""
echo "============================================================"
echo "  Migration terminée avec succès !"
echo "  Docker root : ${DOCKER_NEW_ROOT}"
echo "  Vérification : docker info | grep 'Docker Root Dir'"
echo ""
echo "  Vous pouvez maintenant supprimer l'ancien répertoire :"
echo "  sudo rm -rf /var/lib/docker"
echo "  (Seulement APRÈS avoir vérifié que tout fonctionne !)"
echo "============================================================"
