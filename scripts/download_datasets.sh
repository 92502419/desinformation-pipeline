#!/bin/bash
# scripts/download_datasets.sh — 4 datasets (Fakeddit retiré, inaccessible en 2025)
# Usage DEPUIS LA RACINE DU PROJET : bash scripts/download_datasets.sh


set -e
BASE_DIR="$(pwd)/data/raw"


# Vérification qu'on est bien à la racine du projet
if [ ! -f 'scripts/download_datasets.sh' ]; then
    echo 'ERREUR : Lancer depuis la racine du projet (cd ~/desinformation-pipeline)'
    exit 1
fi


# Vérifier que venv_main est activé
if [ -z "$VIRTUAL_ENV" ]; then
    echo 'INFO : Activation de venv_main...'
    source venv_main/bin/activate
fi


echo '=== Téléchargement des 4 datasets ==='
echo '(Fakeddit retiré : inaccessible sur toutes les sources en 2025)'


# ── 1. WELFake ────────────────────────────────────────────────
echo '[1/4] WELFake (Zenodo)...'
mkdir -p "$BASE_DIR/welfake" && cd "$BASE_DIR/welfake"
curl -L https://zenodo.org/records/4561253/files/WELFake_Dataset.csv -o WELFake_Dataset.csv
echo "WELFake : OK ($(wc -l < WELFake_Dataset.csv) lignes)"
cd -  # Retour à la racine


# ── 2. ISOT ───────────────────────────────────────────────────
echo '[2/4] ISOT (Kaggle API)...'
mkdir -p "$BASE_DIR/isot" && cd "$BASE_DIR/isot"
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset --unzip 2>/dev/null || \
  curl -L 'https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/News-_dataset.zip' -o isot.zip && unzip -o isot.zip
echo 'ISOT : OK'
cd -


# ── 3. LIAR ───────────────────────────────────────────────────
echo '[3/4] LIAR (UCSB)...'
mkdir -p "$BASE_DIR/liar" && cd "$BASE_DIR/liar"
curl -L https://www.cs.ucsb.edu/~william/data/liar_dataset.zip -o liar.zip && unzip -o liar.zip || \
  python3.12 -c "from datasets import load_dataset; ds=load_dataset('liar'); [ds[s].to_pandas().to_csv(f'{\'valid\' if s==\'validation\' else s}.tsv',sep='\\t',index=False) for s in ds]"
echo 'LIAR : OK'
cd -


# ── 4. FakeNewsNet (git clone → CSV dans dataset/) ────────────
echo '[4/4] FakeNewsNet (GitHub clone)...'
mkdir -p "$BASE_DIR/fakenewsnet" && cd "$BASE_DIR/fakenewsnet"
git clone https://github.com/KaiDMML/FakeNewsNet.git . 2>/dev/null || echo 'Déjà cloné'
# Vérifier que les CSV sont présents
ls dataset/*.csv > /dev/null 2>&1 && echo 'FakeNewsNet : OK (CSV trouvés)' || echo 'FakeNewsNet : ATTENTION - vérifier dataset/'
cd -


echo ''
echo '=== Tous les datasets prêts ==='
echo "Espace utilisé : $(du -sh data/raw | cut -f1)"
echo 'Prochaine étape : python scripts/preprocess_data.py'

