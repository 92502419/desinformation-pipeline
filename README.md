# Pipeline Big Data de Monitoring de la Désinformation en Temps Réel

**KOMOSSI Sosso — Master 2 IBDIA, UCAO-UUT, 2025-2026**

Pipeline temps réel de détection de désinformation combinant ingestion Kafka,
traitement en streaming Spark, un modèle Transformer en **apprentissage continu**
(Continual-DistilBERT), un tri-détecteur de dérive de concept (ADWIN + KSWIN +
PageHinkley), et deux interfaces de visualisation (Grafana + Streamlit).

> ⚠️ **Avant de lire les chiffres de performance de ce projet (mémoire, soutenance),
> lisez la section [Incident du 26/08/2026 et reprise](#-incident-du-26082026-et-reprise)
> ci-dessous.** Ce README documente l'état **réellement vérifié** du dépôt à la date du
> dernier entraînement listé, pas un objectif.

---

## 🚨 Incident du 26/08/2026 et reprise

Le 26 août 2026 à 17h41, le disque principal de la machine de développement a subi un
plantage système (`EXT4-fs: orphan cleanup on readonly fs`, remount forcé en lecture
seule) — vraisemblablement déclenché par un entraînement lancé localement sur un GPU
sous-dimensionné (NVIDIA T600, 4 Go de VRAM) qui a saturé la mémoire de la machine.

**Conséquences constatées :**
- Répertoires entièrement vidés : `.git` (tout l'historique de commits), `data/`,
  `scripts/`, `notebooks/`, `config/`, `logs/`, `checkpoints/`, `tests/`,
  `venv_main/`, `venv_producer/`, `venv_spark/`, `streamlit-dashboard/`, `.streamlit/`.
- Fichiers corrompus (contenu binaire illisible) : `README.md`, `.env`,
  `requirements.txt`, `requirements_main.txt`, `runtime.txt`, `docker-compose.yml`,
  `.gitignore`, et dans `models/pretrained/` : `config.json`, `training_history.json`,
  `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`, `tokenizer.json`.
- Le checkpoint de reprise `models/pretrained/_resume_checkpoint.pt` (886 Mo) était
  tronqué en cours d'écriture au moment du plantage — **pas un fichier PyTorch valide**.
  C'est la cause directe de l'observation « l'entraînement s'est arrêté à une seule
  époque » : l'historique qui aurait permis de le vérifier précisément a lui-même été
  corrompu par le même incident.
- Un disque externe de sauvegarde (exFAT) a subi des erreurs I/O le même soir — il ne
  contenait de toute façon que les `.docx` (mémoire, doc technique), pas de sauvegarde
  du code ni des données.

**Ce qui a survécu intact et a servi de base à la reprise :**
- Le code source de `api/`, `producer/`, `spark-app/` (FastAPI, producteur Kafka,
  job Spark Structured Streaming, classifieur ONNX, moniteur de drift, entraînement
  online) — entièrement relu et vérifié, aucune correction nécessaire.
- `models/pretrained/model.safetensors` (poids d'un entraînement antérieur) — non
  réutilisé pour la reprise ci-dessous par prudence (son `config.json` associé étant
  corrompu, son état exact ne peut être garanti) ; remplacé par un ré-entraînement propre.
- Les 3 documents `.docx` (mémoire, documentation technique exhaustive, canevas), qui
  contiennent l'intégralité du code source documenté et ont servi de plan de
  reconstruction fidèle pour tout ce qui manquait (`scripts/`, `docker-compose.yml`,
  `streamlit-dashboard/`, fichiers de config).

**Reconstruit le 27/08/2026 :** `scripts/` (téléchargement, prétraitement,
entraînement avec checkpointing atomique + reprise automatique, export ONNX),
`docker-compose.yml`, `config/` (MongoDB, Grafana), `streamlit-dashboard/`
(dashboard 6 pages), `tests/`, `.gitignore`, `.env.example`, environnement `venv_main`.
Un nouvel historique Git a été initialisé (`git log` ci-dessous fait foi de ce qui a
été réellement exécuté, avec horodatage).

**⚠️ Point d'attention pour la soutenance :** le mémoire (`Memoire_Master2_IBDIA_..._v7.docx`)
annonce des résultats finaux précis (F1-macro = 98,49 %, AUC-ROC = 99,89 %, etc.). Ces
chiffres ne peuvent pas être re-dérivés des fichiers locaux actuels (historique
d'entraînement corrompu). **Les résultats reportés dans la section
[Résultats de l'entraînement](#-résultats-de-lentraînement-vérifiés-localement)
ci-dessous sont ceux, et uniquement ceux, obtenus par le ré-entraînement du
27/08/2026 sur cette machine — à comparer et concilier avec le texte du mémoire
avant la soutenance.**

---

## 🏗️ Architecture

```
RSS/GDELT ──▶ Kafka (raw-news-stream) ──▶ Spark Structured Streaming (micro-batch 5s)
                                                │
                                                ├─▶ Continual-DistilBERT (ONNX INT8, ~5-6 ms/article)
                                                ├─▶ Tri-détecteur de drift (ADWIN+KSWIN+PageHinkley)
                                                ├─▶ Online learning (SGD momentum, reservoir replay équilibré)
                                                │
                                                ├─▶ MongoDB (documents)      ─┐
                                                └─▶ Elasticsearch (full-text) ─┼─▶ FastAPI ─▶ Streamlit (8501)
                                                                               └─▶ Grafana (3000)
```

| Composant | Rôle | Répertoire |
|---|---|---|
| Kafka + Zookeeper | Ingestion distribuée | `docker-compose.yml` |
| RSS/GDELT Producer | Scraping RSS + GDELT → Kafka | `producer/` |
| Spark Structured Streaming | Traitement micro-batch, orchestration inférence/drift/online learning | `spark-app/` |
| Continual-DistilBERT | Classification fake/réel (ONNX INT8) + apprentissage continu | `spark-app/src/nlp_classifier.py` |
| Tri-détecteur de drift | ADWIN + KSWIN + PageHinkley (River) | `spark-app/src/drift_monitor.py` |
| MongoDB / Elasticsearch | Stockage documents / indexation full-text | `docker-compose.yml`, `config/` |
| API FastAPI | Endpoints REST (`/health`, `/api/v1/stats`, `/api/v1/articles`, `/api/v1/drift`, `/api/v1/search/web`) | `api/` |
| Grafana | Tableaux de bord métriques temps réel | `config/grafana/` |
| Streamlit | Dashboard interactif 6 pages (exploration, drift, sources, recherche web) | `streamlit-dashboard/` |

---

## 📂 Structure du projet

```
desinformation-pipeline/
├── scripts/                      # download_datasets.sh, download_african_datasets.py,
│                                  # preprocess_data.py, train_model.py, export_onnx.py
├── data/
│   ├── raw/                      # Datasets bruts téléchargés (non versionné, voir .gitignore)
│   └── processed/                # train/val/test.csv + preprocessing_report.json
├── models/
│   ├── pretrained/                # Modèle HuggingFace (config.json, tokenizer, safetensors)
│   ├── checkpoints/                # Checkpoints intermédiaires (online learning)
│   └── onnx/                       # Modèle quantifié INT8 pour l'inférence
├── producer/                      # Producteur Kafka (RSS + GDELT)
├── spark-app/                     # Job Spark Structured Streaming (NLP + drift + online learning)
├── api/                            # API FastAPI
├── streamlit-dashboard/            # Dashboard Streamlit multi-pages
├── config/                         # Config MongoDB (init.js), Grafana (provisioning, dashboards)
├── tests/                          # Tests unitaires + tests d'intégration API
├── notebooks/                      # Notebooks d'exploration (EDA)
├── reports/                        # Figures + rapport d'analyse (généré par generate_reports.py)
│   ├── RAPPORT_ANALYSE.md          # Courbes, matrices, corrélations + interprétations
│   ├── analyse_erreurs.md          # Faux positifs / faux négatifs qualitatifs
│   ├── metrics_summary.json        # Métriques finales brutes (jeu de test)
│   └── figures/                    # 8 figures PNG
├── docker-compose.yml
└── .env.example                    # Copier en .env avant `docker compose up -d`
```

---

## 🚀 Reproduire le pipeline de bout en bout

### 1. Environnement

```bash
cd ~/desinformation-pipeline
python3 -m venv venv_main && source venv_main/bin/activate
pip install -r requirements_main.txt
```

> Cette machine tourne sous **Python 3.14** (le `python3.12-venv` documenté à l'origine
> n'a pas pu être installé sans droits root) : `torch`/`transformers`/etc. sont donc
> installés en versions récentes non épinglées plutôt qu'aux versions exactes listées
> dans la documentation technique (`torch==2.4.0`, `transformers==4.44.0`...). Fonctionnel,
> mais à harmoniser avec Python 3.12 si l'environnement de soutenance l'exige.

### 2. Données

```bash
bash scripts/download_datasets.sh          # ISOT + WELFake + FakeNewsNet + LIAR (~500 Mo)
python scripts/download_african_datasets.py # Enrichissement MasakhaNEWS (11 langues africaines)
python scripts/preprocess_data.py           # Fusion, nettoyage, équilibrage 50/50, split 60/20/20
```

### 3. Entraînement (avec reprise automatique en cas de plantage + anti-surapprentissage)

```bash
python scripts/train_model.py --epochs 10 --batch_size 24 --lr 2e-5 \
    --africa_boost 5 --save_every 200 --patience 2
```

Le script sauvegarde un checkpoint de reprise **atomique** (`_resume_checkpoint.pt`)
toutes les `--save_every` batches et à chaque fin d'époque : en cas de plantage, il
suffit de relancer exactement la même commande — l'entraînement reprend automatiquement
là où il s'était arrêté (`--resume` est activé par défaut). Le modèle n'est écrit sur
disque (`model.save_pretrained`) que lorsque le F1 de validation s'améliore : les
époques suivantes ne peuvent donc jamais dégrader le modèle livré. `--patience 2`
(défaut) arrête automatiquement l'entraînement après 2 époques sans amélioration du
Val F1 — early stopping standard anti-surapprentissage, conforme au mémoire (§OS 2.3).
`batch_size=24` est le réglage validé sur GPU T600 4 Go (voir section Résultats).

### 4. Analyse des résultats

```bash
python scripts/generate_reports.py
```

Génère `reports/RAPPORT_ANALYSE.md` (courbes d'apprentissage, matrice de confusion,
ROC, précision-rappel, matrice de corrélation, performance par source, distribution
de confiance — chacune avec son interprétation) + `reports/analyse_erreurs.md`
(faux positifs/négatifs qualitatifs), à partir du **jeu de test** (jamais vu à
l'entraînement).

### 5. Export ONNX + démarrage complet

```bash
python scripts/export_onnx.py
cp .env.example .env   # puis compléter avec vos identifiants (Kaggle, Grafana...)
docker compose up -d
```

- API : http://localhost:8000/docs
- Streamlit : http://localhost:8501
- Grafana : http://localhost:3000
- Kafdrop (monitoring Kafka) : http://localhost:9000

### Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 📊 Résultats de l'entraînement (vérifiés localement, 27-28/08/2026)

Corpus final : **77 768 exemples** après fusion + déduplication (ISOT est presque
intégralement absorbé par déduplication dans WELFake — sources qui se recoupent
largement), équilibré 50/50 puis split 60/20/20 → **46 660 train / 15 554 val /
15 554 test**. Entraînement complet sur **10 époques réelles**, `distilbert-base-
multilingual-cased`, GPU T600 4 Go (batch_size=24, mixed precision, padding
dynamique), ~61 min/époque.

**Historique complet (10/10 époques, Train F1 vs Val F1) :**

| Époque | Train F1 | Val F1 | Val AUC |
|---|---|---|---|
| 1 | 0,9219 | 0,9321 | 0,9872 |
| 2 | 0,9569 | 0,9371 | 0,9892 |
| **3** | **0,9625** | **0,9392** ⭐ meilleur | 0,9902 |
| 4 | 0,9691 | 0,9337 | 0,9892 |
| 5 | 0,9747 | 0,9383 | 0,9896 |
| 6 | 0,9793 | 0,9373 | 0,9888 |
| 7 | 0,9826 | 0,9374 | 0,9887 |
| 8 | 0,9851 | 0,9378 | 0,9892 |
| 9 | 0,9862 | 0,9372 | 0,9895 |
| 10 | 0,9890 | 0,9333 | 0,9891 |

Le Train F1 continue de progresser jusqu'à 0,989 tandis que le Val F1 plafonne dès
l'époque 3 — **sur-apprentissage classique au-delà de l'époque 3**. Le modèle
**livré et exporté est celui de l'époque 3** (seule époque où le checkpoint a été
mis à jour, `val_f1 > best_f1`) : les époques 4-10 n'ont jamais écrasé ce fichier.
Un vrai early-stopping par patience (`--patience 2`, conforme au mémoire §OS 2.3)
a été ajouté à `scripts/train_model.py` pour que les prochains runs s'arrêtent
automatiquement au bon endroit.

**Métriques finales sur le jeu de TEST (15 554 exemples, jamais vus à l'entraînement
ni pour la sélection du checkpoint) :**

| Métrique | Valeur |
|---|---|
| F1-macro | **0,9370** |
| Précision | 0,9387 |
| Rappel | 0,9351 |
| ROC-AUC | 0,9899 |
| Average Precision | 0,9900 |
| Matrice de confusion | TP=7272, TN=7302, FP=475, FN=505 |
| Latence ONNX INT8 (mesurée réellement, 100 itérations) | 19,16 ms/article |

📁 **Toutes les figures (courbes d'apprentissage, matrice de confusion, ROC,
précision-rappel, corrélations, performance par source, distribution de confiance)
avec interprétations détaillées : voir [`reports/RAPPORT_ANALYSE.md`](reports/RAPPORT_ANALYSE.md)
et [`reports/figures/`](reports/figures/). Analyse qualitative des erreurs (faux
positifs/négatifs les plus confiants) : [`reports/analyse_erreurs.md`](reports/analyse_erreurs.md).
Régénérable via `python scripts/generate_reports.py`.**

⚠️ La latence ONNX mesurée (19,16 ms) est plus élevée que l'objectif documenté à
l'origine (5-6 ms) — mesurée sur le CPU de cette machine de développement, pas sur
l'infrastructure de production cible. À re-mesurer sur la machine de déploiement
finale avant de citer un chiffre définitif dans le mémoire.

---

## 🌍 Corpus utilisé

| Dataset | Exemples téléchargés | Contenu | Exemples après dédup. |
|---|---|---|---|
| ISOT | 44 898 | Reuters (réel) + sites de désinformation (fake) | ~absorbé par WELFake* |
| WELFake | 72 134 | Fusion Kaggle + Reuters + BuzzFeed + McIntire | 44 609 |
| FakeNewsNet | 23 196 | PolitiFact + GossipCop (fact-checkés) | 13 350 |
| LIAR | 12 791 | Déclarations politiques PolitiFact (6 niveaux → binaire) | 9 247 |
| MasakhaNEWS (+ RSS africain) | 21 623 | 11 langues africaines, enrichissement v2.4 | 12 559 |
| **Total** | **174 642** | | **77 768** |

\* Constat vérifié empiriquement (`data/processed/preprocessing_report.json`) : après
déduplication sur le titre, ISOT n'apparaît plus du tout dans le corpus final — ses
articles (Reuters + sites fake identifiés) sont quasiment tous déjà présents dans
WELFake, qui fusionne lui-même Reuters/Kaggle/BuzzFeed/McIntire. Ce n'est pas un bug :
c'est un recoupement connu et documenté entre ces deux datasets.

Fakeddit a été écarté (inaccessible sur toutes les sources depuis 2025 — voir
`scripts/download_datasets.sh`). Après fusion, nettoyage, déduplication (96 291
doublons supprimés), équilibrage strict 50 % fake / 50 % réel et split 60/20/20, le
corpus d'entraînement définitif (46 660 train / 15 554 val / 15 554 test) est décrit
dans `data/processed/preprocessing_report.json`.

---

## ⚠️ Limites connues

- Peu d'exemples de désinformation **authentiquement africaine** (~77) : Africa Check
  est inaccessible depuis cet environnement (blocage Cloudflare 403) — piste pour des
  travaux futurs.
- Environnement Python 3.14 non celui documenté à l'origine (3.12) — voir section 1.
- Les services Docker (Kafka/Spark/Mongo/Elasticsearch/Grafana) nécessitent une machine
  avec suffisamment de RAM disponible en simultané de tout entraînement GPU — **ne pas
  lancer `docker compose up -d` et `train_model.py` en même temps** sur une machine à
  14 Go de RAM (c'est la combinaison qui a très probablement causé l'incident du
  26/08/2026).
- Performance nettement plus faible sur LIAR (F1=0,598, accuracy=0,62) que sur
  WELFake/FakeNewsNet (F1>0,97) : les déclarations politiques courtes et sans article
  complet sont stylistiquement très différentes du reste du corpus — limite documentée
  dans `reports/RAPPORT_ANALYSE.md`, section performance par source.
- Latence ONNX INT8 mesurée à 19,16 ms/article sur cette machine de développement,
  au-dessus de l'objectif documenté (5-6 ms) — à re-mesurer sur l'infrastructure de
  déploiement cible.
- `Guide_Presentation_Soutenance_KOMOSSI_Sosso.docx` original irrécupérable (corrompu,
  aucune sauvegarde exploitable) — entièrement réécrit à partir de l'état réel du
  projet le 27/08/2026.

---

## 🧾 Historique

Cet historique Git a été réinitialisé le 27/08/2026 après la perte totale de
l'historique précédent (voir [Incident du 26/08/2026](#-incident-du-26082026-et-reprise)).
