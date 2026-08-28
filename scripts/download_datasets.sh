#!/bin/bash
# scripts/download_datasets.sh — Téléchargement des 4 datasets du corpus
# (Fakeddit retiré du projet : inaccessible sur toutes les sources depuis 2025 —
#  voir Documentation_Technique_Pipeline_KOMOSSI_Sosso_v9.docx, Erreur fréquente §4)
#
# Usage OBLIGATOIRE depuis la RACINE du projet :
#   cd ~/desinformation-pipeline && bash scripts/download_datasets.sh
#
# Reconstruit le 2026-08-27 après un plantage disque qui a vidé data/raw/,
# scripts/ et models/checkpoints/ (voir README.md, section "Incident & reprise").

set -e
BASE_DIR="$(pwd)/data/raw"

if [ ! -f 'scripts/download_datasets.sh' ]; then
  echo 'ERREUR : Lancer depuis la racine du projet (cd ~/desinformation-pipeline)'
  exit 1
fi

if [ -z "$VIRTUAL_ENV" ]; then
  echo 'INFO : Activation de venv_main...'
  source venv_main/bin/activate
fi

echo '=== Téléchargement des 4 datasets (~153 000 exemples, ~500 Mo) ==='
echo '(Fakeddit retiré : inaccessible sur toutes les sources en 2025)'

# ── 1. WELFake (Zenodo, 72 134 exemples) ────────────────────────
echo '[1/4] WELFake (Zenodo)...'
mkdir -p "$BASE_DIR/welfake" && cd "$BASE_DIR/welfake"
if [ ! -f WELFake_Dataset.csv ]; then
  curl -L --fail https://zenodo.org/records/4561253/files/WELFake_Dataset.csv -o WELFake_Dataset.csv
fi
echo "WELFake : OK ($(wc -l < WELFake_Dataset.csv) lignes)"
cd - > /dev/null

# ── 2. ISOT (Kaggle API ou copie UVic, 44 898 exemples) ─────────
echo '[2/4] ISOT (Kaggle API)...'
mkdir -p "$BASE_DIR/isot" && cd "$BASE_DIR/isot"
if [ ! -f Fake.csv ] || [ ! -f True.csv ]; then
  kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset --unzip 2>/dev/null || \
    (curl -L --fail 'https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/News-_dataset.zip' -o isot.zip && unzip -o isot.zip)
fi
echo 'ISOT : OK'
cd - > /dev/null

# ── 3. LIAR (UCSB ou copie HuggingFace, 12 836 exemples) ────────
echo '[3/4] LIAR (UCSB)...'
mkdir -p "$BASE_DIR/liar" && cd "$BASE_DIR/liar"
if [ ! -f train.tsv ]; then
  curl -L --fail https://www.cs.ucsb.edu/~william/data/liar_dataset.zip -o liar.zip && unzip -o liar.zip || \
    python3 -c "
from datasets import load_dataset
ds = load_dataset('liar')
for split in ['train','validation','test']:
    df = ds[split].to_pandas()
    fname = 'valid' if split == 'validation' else split
    df.to_csv(f'{fname}.tsv', sep='\t', index=False)
print('LIAR téléchargé via HuggingFace')
"
fi
echo 'LIAR : OK'
cd - > /dev/null

# ── 4. FakeNewsNet (git clone → CSV déjà prêts dans dataset/) ───
echo '[4/4] FakeNewsNet (GitHub clone)...'
mkdir -p "$BASE_DIR/fakenewsnet" && cd "$BASE_DIR/fakenewsnet"
if [ ! -d dataset ]; then
  git clone --depth 1 https://github.com/KaiDMML/FakeNewsNet.git . 2>/dev/null || echo 'Déjà cloné / clone impossible'
fi
ls dataset/*.csv > /dev/null 2>&1 && echo 'FakeNewsNet : OK (CSV trouvés)' || echo 'FakeNewsNet : ATTENTION - vérifier dataset/'
cd - > /dev/null

echo ''
echo '=== Tous les datasets prêts ==='
echo "Espace utilisé : $(du -sh data/raw | cut -f1)"
echo 'Prochaine étape : python scripts/download_african_datasets.py puis python scripts/preprocess_data.py'
