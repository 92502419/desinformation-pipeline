# notebooks/

Le contenu original de ce dossier (notebooks d'exploration — EDA du corpus, analyse
des erreurs de classification, visualisations pour le mémoire) a été perdu dans
l'incident du 26/08/2026 (voir README.md racine, section "Incident & reprise") et
n'a pas pu être reconstruit à partir de la documentation technique (celle-ci ne
contient que le code des scripts `.py`, pas les notebooks Jupyter).

À reconstruire si besoin pour la soutenance :
- `01_eda_corpus.ipynb` — distribution des labels/sources, longueur des textes,
  langues détectées (utiliser `data/processed/preprocessing_report.json`).
- `02_error_analysis.ipynb` — matrice de confusion, exemples mal classés
  (charger `models/pretrained/` + `data/processed/test/test.csv`).
- `03_training_curves.ipynb` — courbes loss/F1 par epoch depuis
  `models/pretrained/training_history.json`.
