# Pipeline Big Data de Monitoring de la Désinformation en Temps Réel

**Auteur :** KOMOSSI Sosso — Master BIG DATA IA, Institut ESI — UCAO UUT, 2025-2026
**Encadrants :** Dr Nadjime PINDRA (Directeur de Mémoire), Dr Kodjo Sena Messa APEKE (Co-Directeur)

Pipeline temps réel de détection de désinformation : ingestion Kafka (RSS + GDELT),
traitement en streaming Spark, classification par un modèle Transformer multilingue en
**apprentissage continu** (Continual-DistilBERT, ONNX INT8) avec **calibration
probabiliste et prédiction sélective**, tri-détecteur de dérive de concept à 4 canaux
(ADWIN + KSWIN + PageHinkley + ADWIN sur l'entropie), stockage MongoDB + Elasticsearch,
et deux interfaces de visualisation (Grafana, avec règles d'alerte natives, + Streamlit).
Déploiement complet via Docker Compose (13 services). Licence MIT.

---

## Architecture

Pipeline en 5 niveaux. Les chiffres (débit, latence, mémoire) sont ceux réellement
mesurés / configurés sur la machine de développement (14 Go RAM, CPU uniquement —
pilote NVML cassé depuis l'incident du 26/08/2026, torch.cuda désactivé par sécurité).

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  NIVEAU 1 — SOURCES                                        producer/         │
│                                                                              │
│   ┌───────────────────────┐        ┌────────────────────────────────────┐    │
│   │  RSS  (12 flux)       │        │  GDELT GKG  (API DOC 2.0)          │    │
│   │  AFP · Reuters · BBC  │        │  thème FAKE_NEWS,DISINFORMATION    │    │
│   │  RFI · Al Jazeera ... │        │  presse mondiale, ~15 min          │    │
│   │  scrape toutes les 60s│        │  requête toutes les 900s           │    │
│   └───────────┬───────────┘        └──────────────────┬─────────────────┘    │
└───────────────┼───────────────────────────────────────┼──────────────────────┘
                │            kafka_producer.py (asyncio)│
                └───────────────────┬───────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  NIVEAU 2 — INGESTION                     Kafka (Confluent 7.6.0) + Zookeeper│
│                                                                              │
│      topic  raw-news-stream  ───────────────►  (consommé par Spark)          │
│      topic  drift-alerts     ◄───────────────  (produit par Spark)           │
│      déduplication TTL 2 h · Kafdrop UI :9000                                │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  NIVEAU 3 — TRAITEMENT             spark-app/   (Spark Structured Streaming  │
│                                    3.5.6, micro-batch = 5 s)                 │
│                                                                              │
│   pour chaque article du micro-batch :                                       │
│     1. ┌───────────────────────────────┐   Continual-DistilBERT              │
│        │  Inférence ONNX INT8          │   multilingue, ~19 ms/article       │
│        │  + calibration (T scaling)    │   (mesuré, CPU dev)                 │
│        │  -> is_fake, verdict, p_fake  │   verdict: fake/real/uncertain      │
│        └───────────────────────────────┘                                     │
│     2. ┌───────────────────────────────┐   ADWIN(0.45)+KSWIN(0.35)+PH(0.20)  │
│        │  Tri-détecteur de drift (×4)  │   -> score composite (inchangé)     │
│        │  -> score composite OU        │   + EntropyADWIN en OU logique      │
│        │     EntropyADWIN              │   alerte 0.4 · confirmation 0.8     │
│        └───────────────────────────────┘                                     │
│     3. ┌───────────────────────────────┐   reservoir 5 000 · 1 batch / 5     │
│        │  Apprentissage en ligne (SGD) │   LR 1e-5 (stable) -> 5e-5 (dérive) │
│        │  reservoir replay équilibré   │   sync ONNX / 100 batches           │
│        └───────────────────────────────┘                                     │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  NIVEAU 4 — STOCKAGE                                       config/           │
│                                                                              │
│   ┌─────────────────────────┐          ┌──────────────────────────────┐      │
│   │  MongoDB 7.0            │          │  Elasticsearch 8.14          │      │
│   │  collections :          │          │  index :                     │      │
│   │   • articles            │          │   • articles                 │      │
│   │   • drift_events        │          │   • drift-events             │      │
│   └────────────┬────────────┘          └───────────────┬──────────────┘      │
└────────────────┼───────────────────────────────────────┼─────────────────────┘
                 │                                       │
                 ▼                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  NIVEAU 5 — PRÉSENTATION                                                      │
│                                                                               │
│   ┌──────────────────┐   lit Mongo+ES   ┌──────────────────────────────┐      │
│   │  FastAPI :8000   │◄─────────────────│  api/   (/health, /stats,    │      │
│   │  API REST + docs │                  │  /articles, /drift, /search) │      │
│   └────────┬─────────┘                  └──────────────────────────────┘      │
│            │                                                                  │
│            ├──────────────►  Streamlit :8501   (dashboard interactif, 7 pages)│
│            │                                                                  │
│   ES / Mongo ────────────►  Grafana :3000      (dashboard P1–P7 + 5 règles    │
│                                                  d'alerte natives, refresh 10s)│
│   Spark UI :4040 (driver local[2], jobs/DAG/métriques des micro-batchs)       │
└───────────────────────────────────────────────────────────────────────────────┘
```

| Composant | Rôle | Répertoire |
|---|---|---|
| Kafka + Zookeeper | Ingestion distribuée | `docker-compose.yml` |
| RSS/GDELT Producer | Scraping RSS + GDELT → Kafka | `producer/` |
| Spark Structured Streaming | Traitement micro-batch, orchestration inférence / drift / online learning | `spark-app/` |
| Continual-DistilBERT | Classification fake/réel (ONNX INT8) + apprentissage continu | `spark-app/src/nlp_classifier.py` |
| Calibration probabiliste | Temperature scaling + prédiction sélective (zone grise `uncertain`) | `spark-app/src/calibration.py`, `api/src/calibration.py`, `scripts/calibrate_model.py` |
| Tri-détecteur de drift | ADWIN + KSWIN + PageHinkley + ADWIN sur l'entropie (River) | `spark-app/src/drift_monitor.py` |
| MongoDB / Elasticsearch | Stockage documents / indexation full-text | `docker-compose.yml`, `config/` |
| API FastAPI | Endpoints REST (`/health`, `/api/v1/stats`, `/api/v1/model/calibration`, `/api/v1/articles`, `/api/v1/drift`, `/api/v1/search/web`) | `api/` |
| Grafana | Dashboard auto-provisionné (7 panneaux P1–P7) + 5 règles d'alerte natives, datasources Elasticsearch + MongoDB | `config/grafana/` |
| Streamlit | Dashboard interactif 7 pages | `streamlit-dashboard/` |

---

## Structure du projet

```
desinformation-pipeline/
├── scripts/                # download_datasets.sh, download_african_datasets.py,
│                           # preprocess_data.py, train_model.py, generate_reports.py,
│                           # export_onnx.py, calibrate_model.py, inject_drift_simulation.py,
│                           # train_baselines.py, experiment_*.py
├── notebooks/              # 01_eda_corpus, 02_error_analysis, 03_training_curves (exécutés)
├── data/
│   ├── raw/                # datasets bruts téléchargés (non versionné)
│   └── processed/          # train/val/test.csv + preprocessing_report.json
├── models/
│   ├── pretrained/         # modèle HuggingFace (config, tokenizer, safetensors)
│   ├── checkpoints/        # checkpoints intermédiaires (online learning)
│   ├── onnx/               # modèle quantifié INT8 pour l'inférence
│   └── calibration.json    # température + seuils (scripts/calibrate_model.py)
├── producer/               # producteur Kafka (RSS + GDELT)
├── spark-app/              # job Spark Structured Streaming (NLP + drift + online learning)
├── api/                    # API FastAPI
├── streamlit-dashboard/    # dashboard Streamlit
├── config/                 # config MongoDB (init.js) + Grafana (provisioning, dashboards, alerting)
├── tests/                  # tests unitaires + tests d'intégration API
├── reports/                # rapports d'analyse + figures (générés)
├── docker-compose.yml
├── LICENSE                 # MIT
└── .env.example            # à copier en .env avant `docker compose up -d`
```

---

## Démarche complète (dans l'ordre)

### 0. Environnement Python

```bash
cd ~/desinformation-pipeline
python3 -m venv venv_main && source venv_main/bin/activate
pip install -r requirements_main.txt
```

### 1. Téléchargement des données

```bash
bash scripts/download_datasets.sh            # ISOT, WELFake, FakeNewsNet, LIAR  (~500 Mo)
python scripts/download_african_datasets.py  # MasakhaNEWS (12 langues) + Google News RSS africain
```

### 2. Fusion, nettoyage, harmonisation

```bash
python scripts/preprocess_data.py
# Fusionne les 5 sources, nettoie le texte, harmonise les colonnes (title, body, label,
# source), déduplique sur le titre, équilibre 50 % fake / 50 % réel, split 60/20/20.
# Produit data/processed/{train,val,test}.csv + preprocessing_report.json
```

### 3. Entraînement du modèle

```bash
python scripts/train_model.py --epochs 10 --batch_size 24 --lr 2e-5 \
    --africa_boost 5 --save_every 200 --patience 2
```

`distilbert-base-multilingual-cased`, checkpoint de reprise atomique tous les
`--save_every` batches (relancer la même commande après un plantage). Le modèle n'est
écrit sur disque que lorsque le F1 de validation s'améliore ; `--patience 2` arrête
l'entraînement après 2 époques sans amélioration.

### 4. Rapport d'analyse

```bash
python scripts/generate_reports.py
# -> reports/RAPPORT_ANALYSE.md, reports/analyse_erreurs.md, reports/figures/*.png
```

### 5. Export ONNX INT8

```bash
python scripts/export_onnx.py
# -> models/onnx/model_quantized.onnx
```

### 6. Calibration probabiliste et prédiction sélective

```bash
python scripts/calibrate_model.py --target-precision 0.97 --max-abstention 0.15
# -> models/calibration.json (chargé en production, repli sûr T=1 si absent)
# -> reports/calibration.md, reports/figures/11_calibration_fiabilite.png,
#    reports/figures/12_risque_couverture.png
```

Corrige la sur-confiance mesurée au chapitre 4 (temperature scaling, ordre des
scores et AUC inchangés) et ajoute un troisième verdict `uncertain` entre les
seuils `tau_real`/`tau_fake` — un article dans la zone grise n'est plus tranché à
tort mais renvoyé à la vérification humaine.

### 7. Configuration et lancement Docker

```bash
cp .env.example .env          # puis renseigner GRAFANA_ADMIN_PASSWORD, KAGGLE_*, etc.
docker compose build          # construit api, spark-app, rss-producer, streamlit
docker compose up -d          # démarre les 13 services
```

Interfaces :

| Service | URL |
|---|---|
| Streamlit (dashboard) | http://localhost:8501 |
| Grafana (7 panneaux + 5 règles d'alerte) | http://localhost:3000 — `admin` / `GRAFANA_ADMIN_PASSWORD` du `.env` |
| API + Swagger | http://localhost:8000/docs |
| Kafdrop (monitoring Kafka) | http://localhost:9000 |
| Spark UI (driver local[2]) | http://localhost:4040 |

### 8. Tests

```bash
pip install pytest
pytest tests/ -v
```

### 9. Notebooks (reconstruits, déjà exécutés)

```bash
jupyter lab notebooks/
# 01_eda_corpus.ipynb        — composition du corpus, avant/après déduplication
# 02_error_analysis.ipynb    — métriques, erreurs par source (charge les artefacts
#                               reports/ existants ; ré-inférence optionnelle)
# 03_training_curves.ipynb   — courbes Train/Val F1 par époque
```

### 10. (optionnel) Expériences de validation drift / online learning / baselines

```bash
python scripts/train_baselines.py                           # -> reports/baselines_statiques.md (TF-IDF+LR / SVM vs DistilBERT)
python scripts/experiment_tri_detecteur.py --reps 30        # -> reports/tableau_4_3_tri_detecteur.md
python scripts/experiment_derive_opportunite.py             # -> reports/experience_derive_opportunite.md
python scripts/experiment_federated_simulation.py --rounds 15  # -> reports/experience_federated_simulation.md
python scripts/inject_drift_simulation.py --scenario B      # injecte une dérive dans le flux live
```

---

## Résultats (jeu de test, 15 554 exemples jamais vus)

| Métrique | Valeur |
|---|---|
| F1-macro | **0,9370** |
| Précision / Rappel | 0,9387 / 0,9351 |
| ROC-AUC / Average Precision | 0,9899 / 0,9900 |
| Matrice de confusion | TP 7272 · TN 7302 · FP 475 · FN 505 |
| Latence ONNX INT8 (mesurée, 100 itérations, CPU) | 19,16 ms/article |
| Meilleure époque retenue | 3 / 10 (early stopping) |

Détail, courbes et interprétations : [`reports/RAPPORT_ANALYSE.md`](reports/RAPPORT_ANALYSE.md)
et [`reports/figures/`](reports/figures/). Expériences tri-détecteur et « dérive = opportunité » :
[`reports/EXPERIENCES_2026-08-29.md`](reports/EXPERIENCES_2026-08-29.md).

### Comparaison aux baselines statiques (H2)

Entraînées sur le même split (`scripts/train_baselines.py`), évaluées sur le même jeu de
test — répond à un écart identifié après l'incident du 26/08/2026 (aucune baseline
n'avait été ré-entraînée) :

| Modèle | F1-macro |
|---|---|
| TF-IDF (1-2-grammes) + Régression Logistique | 0,9155 |
| TF-IDF (1-2-grammes) + Linear SVM | 0,9178 |
| **Continual-DistilBERT** | **0,9370** |

Détail : [`reports/baselines_statiques.md`](reports/baselines_statiques.md).

### Calibration probabiliste et prédiction sélective

`scripts/calibrate_model.py` — temperature scaling (T = 3,0554, ordre des scores et
AUC-ROC inchangés) + zone grise `uncertain` (`p_cal <= 0,25` → réel, `>= 0,7109` →
fake, entre les deux → incertain) :

| Mesure (jeu de test) | Avant | Après |
|---|---|---|
| ECE (erreur de calibration attendue) | 0,0455 | **0,0156** |
| Score de Brier | 0,0529 | **0,0440** |
| Couverture (articles tranchés) | 100 % | **92,12 %** |
| F1-macro sur la zone tranchée | — | **0,9678** |

La zone grise se concentre là où le modèle est réellement faible : **58,3 %** des
articles LIAR tombent en zone `uncertain` (déclarations politiques courtes, terrain
faible du modèle — cf. Limites), contre **< 1,3 %** pour WELFake/FakeNewsNet et
**0 %** pour la quasi-totalité du sous-corpus africain. Détail complet :
[`reports/calibration.md`](reports/calibration.md).

### Tri-détecteur à 4 canaux (entropie, OU logique)

`spark-app/src/drift_monitor.py` ajoute un canal ADWIN sur l'entropie H(p_fake), motivé
par le verrou du 29/08/2026 (une campagne hors-distribution ne déplace pas assez la
moyenne de p_fake pour déclencher les 3 détecteurs historiques). Une première version
pondérait ce canal dans le score composite : `reports/tableau_4_3_tri_detecteur.md` a
montré que cela **dégrade** le TPR des scénarios B/D (le canal ne se déclenche jamais
sur ces scénarios de dérive entre deux régimes confiants — voir aussi B.4 ci-dessous).
Le canal est donc appliqué en **OU logique indépendant** : le score composite à 3
détecteurs retrouve exactement ses performances d'origine (ex. TPR Scénario B = 0,27,
identique à la version sans entropie), et EntropyADWIN s'ajoute sans coût sur ces
scénarios. Détail : [`reports/tableau_4_3_tri_detecteur.md`](reports/tableau_4_3_tri_detecteur.md).

Rejoué sur la campagne LIAR exacte du 29/08/2026
([`reports/experience_derive_opportunite.md`](reports/experience_derive_opportunite.md)) —
celle où le tri-détecteur à 3 canaux ne déclenchait **jamais** — le tri-détecteur à 4
canaux **détecte désormais la dérive** (délai 80 messages, EntropyADWIN se déclenche là
où les 3 autres restent muets). La modulation automatique du LR qui s'ensuit porte le
gain dynamique en P2 à **+1,29 point de F1-macro** (contre +0,4 point le 29/08/2026 sans
détection, et légèrement au-dessus du +1,10 point obtenu ce jour-là avec un forçage
manuel du LR) — toujours sans oubli catastrophique (sonde welfake/fnn : +0,00 point).
L'hypothèse H3 reste **non confirmée** au sens strict (le modèle ne récupère pas son
niveau pré-campagne, écart de -23,9 points) : le verrou de *détection* identifié le
29/08/2026 est levé, le verrou de *budget d'adaptation* reste une limite assumée.

---

## Corpus

| Source | Téléchargé | Après déduplication |
|---|---|---|
| ISOT | 44 898 | 0 (absorbé par WELFake — recoupement connu) |
| WELFake | 72 134 | 44 609 |
| FakeNewsNet | 23 196 (index) | 13 350 (avec texte exploitable) |
| LIAR | 12 836 | 9 247 |
| Sous-corpus africain (MasakhaNEWS 12 langues + Google News RSS) | 21 623 | 10 562 |
| **Total** | **174 687** | **77 768** |

Split final après équilibrage 50/50 : **46 660 train / 15 554 val / 15 554 test**
(voir `data/processed/preprocessing_report.json`). Fakeddit a été écarté (inaccessible
depuis 2025).

---

## Limites connues

- **LIAR** : F1 = 0,60 (vs > 0,97 sur WELFake / FakeNewsNet) — déclarations politiques
  courtes, sans article complet, stylistiquement éloignées du reste du corpus.
- Peu d'exemples de désinformation **authentiquement africaine** (Africa Check inaccessible,
  blocage Cloudflare 403) — les 21 623 articles africains sont tous des dépêches réelles.
- **Ne pas lancer `docker compose up -d` et `train_model.py` en même temps** sur une
  machine à 14 Go de RAM (saturation mémoire).
- Machine de développement sous **Python 3.12.3** : les versions de bibliothèques ne sont
  pas épinglées à celles de la documentation technique (`torch==2.4.0`, etc.). CPU
  uniquement (pilote NVML cassé depuis l'incident du 26/08/2026) — l'inférence
  DistilBERT y est nettement plus lente que sur une machine de déploiement équipée
  d'un GPU fonctionnel ou d'un CPU serveur récent.
- Latence ONNX INT8 (19,16 ms) mesurée sur CPU de développement — à re-mesurer sur
  l'infrastructure de déploiement cible.
- Hypothèse « la dérive est une opportunité d'apprentissage » (H3) : **non confirmée**
  dans la configuration actuelle du pipeline (le modèle dynamique ne récupère pas son
  niveau pré-dérive) ; en revanche aucun oubli catastrophique, et le tri-détecteur à
  4 canaux (15/09/2026) a résolu le verrou de *détection* sur la campagne LIAR du
  29/08/2026 (gain dynamique porté à +1,29 pt). Détail :
  [`reports/EXPERIENCES_2026-08-29.md`](reports/EXPERIENCES_2026-08-29.md),
  [`reports/experience_derive_opportunite.md`](reports/experience_derive_opportunite.md).
- **Federated Learning** (`scripts/experiment_federated_simulation.py`) : preuve de
  concept FedAvg sur un modèle plus léger (HashingVectorizer + régression logistique),
  pas sur Continual-DistilBERT lui-même (hors de portée CPU) — voir
  [`reports/experience_federated_simulation.md`](reports/experience_federated_simulation.md).
- **Flux RSS de sources douteuses** : `RSS_SOURCES['suspicious']` reste vide
  (`producer/src/rss_sources.py`) — aucune source volontairement peu fiable n'est
  branchée, faute de liste vérifiable et maintenue de façon fiable pour ce dépôt ;
  seules des sources fact-checkées sont ingérées.
