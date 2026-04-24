# Pipeline Big Data de Monitoring de la Désinformation en Temps Réel

**Auteur :** KOMOSSI Sosso — Master 2 BIG DATA IA, UCAO-UUT  
**Encadrants :** M. TCHANTCHO Leri & M. BABA Kpatcha  
**Année :** 2025-2026

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SOURCES DE DONNÉES                                   │
│   📡 RSS Feeds (AFP, BBC, Reuters, Al Jazeera, RFI, Jeune Afrique...)   │
│   🌐 GDELT API (Articles géopolitiques multilingues)                    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │ scraping toutes les 60s
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    APACHE KAFKA (Confluent 7.6.0)                       │
│   📥 raw-news-stream (6 partitions) → 📤 classified-news + drift-alerts │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │ Spark Structured Streaming
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│          SPARK STREAMING (local[2]) — Continual-DistilBERT              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │  ONNX INT8 Infer │  │  Tri-Détecteur   │  │  Online Learning      │  │
│  │  ~5-6 ms/article │  │  ADWIN(0.45)     │  │  PyTorch AdamW        │  │
│  │  DistilBERT-ml   │  │  KSWIN(0.35)     │  │  Reservoir 5000       │  │
│  │                  │  │  PageHinkley(0.20│  │  ONNX sync/100 batch  │  │
│  └──────────────────┘  └──────────────────┘  └───────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │ bulk write
               ┌───────────┴───────────┐
               ▼                       ▼
  ┌──────────────────────┐   ┌──────────────────────────┐
  │  MongoDB 7.0         │   │  Elasticsearch 8.14.0    │
  │  articles + drift    │   │  index full-text         │ 
  └────────┬─────────────┘   └───────────┬──────────────┘
           └─────────────────┬───────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    COUCHE PRÉSENTATION                                   │
│                                                                          │
│  [Streamlit Dashboard :8501]  [FastAPI REST :8000]  [Grafana :3000]      │
│  [Kafdrop Kafka UI   :9000]   [Elasticsearch  :9200]                     │
│                                                                          │
│  --> Streamlit = interface principale (6 pages interactives)             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Prérequis

| Ressource | Minimum | Recommandé |
|-----------|---------|------------|
| RAM | 11 GB | 16 GB |
| CPU | 4 cœurs | 8 cœurs |
| Disque | 30 GB libre | 50 GB |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |
| Docker | 26.1+ | 29.1+ |
| Docker Compose | Plugin v2 | Plugin v2 |

> ⚠️ **Important** : Les bases de données (MongoDB, Elasticsearch, Kafka) doivent
> être sur un **filesystem Linux** (ext4/xfs). Un disque externe NTFS/exFAT n'est
> **pas compatible** (WiredTiger et Lucene nécessitent le POSIX file-locking).

---

## Démarrage rapide

### 1. Pré-requis Docker

```bash
# Vérifier Docker Compose
docker compose version

# Si non installé :
sudo apt-get install docker-compose-plugin

# Ajouter l'utilisateur au groupe docker (si nécessaire)
sudo usermod -aG docker $USER && newgrp docker

# Configurer vm.max_map_count pour Elasticsearch
sudo sysctl -w vm.max_map_count=262144
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
```

### 2. Vérifier les modèles ML

Les modèles doivent être présents **avant** de lancer Docker :

```bash
# Modèle ONNX quantifié (requis par spark-app)
ls models/onnx/model_quantized.onnx

# Modèle DistilBERT fine-tuné (requis par spark-app)
ls models/pretrained/config.json

# Si les dossiers sont vides → les modèles doivent être générés d'abord :
# python scripts/train_model.py   (fine-tuning DistilBERT)
# python scripts/export_onnx.py   (export ONNX INT8)
```

### 3. Lancer le pipeline

```bash
# Aller dans le répertoire du projet
cd ~/desinformation-pipeline   # adapter selon votre chemin

# Démarrer tous les services
docker compose up -d

# Suivre le démarrage en temps réel (Ctrl+C pour quitter le suivi)
docker compose ps --watch
# ou : watch -n 3 docker compose ps
```

> ⏱️ **Temps de démarrage complet : 3 à 5 minutes**
>
> Les services démarrent dans cet ordre avec des délais d'attente obligatoires :
> 1. **Zookeeper** (~10s) → **MongoDB** (~30s) → **Elasticsearch** (~60-120s)
> 2. **Kafka** (attend Zookeeper healthy) → **kafka-init** (crée les topics)
> 3. **API FastAPI** (attend MongoDB + Elasticsearch healthy)
> 4. **Streamlit** (attend API + MongoDB + Elasticsearch healthy)
> 5. **Spark-app** + **rss-producer** (attendent Kafka + MongoDB + Elasticsearch)
>
> Tant que `docker compose ps` affiche des services en `starting` ou `(health: starting)`,
> les interfaces ne sont **pas encore accessibles**. C'est normal.

```bash
# Attendre que tout soit stable, puis vérifier la santé de l'API
curl http://localhost:8000/health
# Réponse attendue : {"status":"ok","mongo":"up","elasticsearch":"up"}
```

### 4. Accéder aux interfaces

| Interface | URL | Identifiants |
|-----------|-----|--------------|
| **Streamlit Dashboard** (interface principale) | http://localhost:8501 | — |
| FastAPI Swagger (documentation API) | http://localhost:8000/docs | — |
| Grafana (métriques & dashboards) | http://localhost:3000 | admin / admin2025 |
| Kafdrop (monitoring Kafka) | http://localhost:9000 | — |
| Elasticsearch (API REST) | http://localhost:9200 | — |

> **Streamlit (port 8501)** est l'interface principale du projet. Elle regroupe
> 7 pages interactives : Tableau de bord, Articles temps réel, Recherche & Analyse,
> Drift & Apprentissage, Alertes, Infrastructure, À propos.

> **Grafana — alertes** : les règles d'alerte sont pré-configurées automatiquement.
> Elles apparaissent en état **"NoData"** tant qu'Elasticsearch ne contient pas encore
> de données (pipeline pas encore démarré). Dès que `spark-app` traite les premiers
> articles (~2 min après le démarrage complet), les alertes passent en évaluation normale.

---

## Ordre de démarrage recommandé

Pour un premier démarrage propre ou après un problème :

```bash
# Étape 1 — Infrastructure de base
docker compose up -d zookeeper mongodb elasticsearch

# Attendre que MongoDB et Elasticsearch soient healthy (30-120s)
# Le statut "(healthy)" doit apparaître dans la colonne STATUS
watch -n 5 docker compose ps

# Étape 2 — Kafka
docker compose up -d kafka

# Attendre que Kafka soit healthy (~30s)
watch -n 5 docker compose ps

# Étape 3 — Tous les autres services
docker compose up -d

# Suivre les logs du pipeline Spark
docker compose logs -f spark-app
```

> ⚠️ **Si Streamlit affiche "backends indisponibles"** après le démarrage complet :
> aller sur la page **Infrastructure** et cliquer sur **"🔄 Reconnecter"**.
> Cela vide le cache des connexions et force une reconnexion.

---

## Endpoints API REST

```
GET /health                              — Santé (MongoDB + Elasticsearch)
GET /api/v1/stats                        — Statistiques globales du pipeline
GET /api/v1/articles/recent?limit=50     — Derniers articles classifiés
GET /api/v1/articles/recent?fake_only=true — Articles fake uniquement
GET /api/v1/articles/search?q=<terme>   — Recherche full-text (Elasticsearch)
GET /api/v1/articles/virality?hours=24  — Tendance horaire du taux de fake
GET /api/v1/drift/events                — Historique des alertes de drift
GET /api/v1/drift/stats                 — Statistiques agrégées drift
```

---

## Dashboard Streamlit (port 8501)

L'interface Streamlit est construite avec 6 pages accessibles via la barre latérale :

| Page | Contenu |
|------|---------|
| **Tableau de bord** | KPIs en temps réel, répartition fake/réel, tendance horaire, derniers articles |
| **Articles** | Liste filtrée par statut/source, histogramme de confiance, cartes détaillées |
| **Recherche** | Recherche full-text Elasticsearch, scatter plot de pertinence |
| **Drift** | Timeline des alertes de dérive, formule composite, statistiques ADWIN/KSWIN/PH |
| **Infrastructure** | Etat de santé des services Docker, liens d'accès, diagramme d'architecture |
| **A propos** | Description du projet, stack technologique, endpoints API |

```bash
# Vérifier que Streamlit est bien UP
docker compose ps streamlit

# Accéder au dashboard
xdg-open http://localhost:8501
```

---

## Performances

| Métrique | Valeur |
|----------|--------|
| F1-score (validation) | **98.49%** |
| AUC-ROC | **99.89%** |
| Latence inférence ONNX INT8 | ~5-6 ms/article |
| Compression FP32 → INT8 | ~75% |
| Débit (CPU 4 cœurs) | ~200 articles/batch/5s |
| Latence end-to-end | < 10 secondes |

---

## Dépannage (Troubleshooting)

### Streamlit affiche "backends indisponibles" ou aucun article

Le message apparaît si Streamlit a tenté de se connecter avant que MongoDB/API soient prêts.
La connexion échouée est mise en cache.

**Solution :** ouvrir http://localhost:8501, aller sur la page **⚙️ Infrastructure**,
puis cliquer sur **"🔄 Reconnecter"**. Cela vide le cache et force une reconnexion.

### La page Infrastructure affiche tous les services DOWN sauf Streamlit

**Cause :** bug corrigé — la vérification utilisait `localhost` depuis l'intérieur
du conteneur Docker. `localhost` dans Docker = le conteneur lui-même, donc seul
Streamlit (qui tourne dans ce conteneur) était détecté UP.
**Solution :** reconstruire l'image Streamlit après la correction :
```bash
docker compose build streamlit
docker compose up -d streamlit
```

### MongoDB ne démarre pas

```bash
docker logs mongodb 2>&1 | grep -E "ERROR|FATAL"
```

**Cause fréquente :** données MongoDB sur un disque NTFS/exFAT.
**Solution :** vérifier que `mongodb_data` est un volume nommé Docker (pas un bind-mount sur disque externe).

```bash
# Vérifier les volumes
docker volume ls | grep desinformation
```

### Spark-app crashe en boucle

```bash
docker logs spark-app 2>&1 | tail -30
```

**Cause fréquente :** le dossier `/app/models/pretrained` est vide dans le container.
**Solution :** les modèles doivent être présents dans `./models/pretrained/` sur l'hôte. Vérifier le bind-mount :

```bash
docker exec spark-app ls -la /app/models/pretrained/
```

### Elasticsearch en manque de mémoire

```bash
# Vérifier l'utilisation mémoire
docker stats --no-stream | grep elasticsearch
```

Si la mémoire dépasse 99%, augmenter la limite dans `docker-compose.yml` :
```yaml
ES_JAVA_OPTS: -Xms384m -Xmx384m
memory: 768M
```

### Les alertes Grafana affichent "NoData"

C'est **normal** au démarrage. Grafana évalue les règles sur les données Elasticsearch.
Tant qu'aucun article n'a été traité par Spark, l'index `articles` est vide → NoData.
Les alertes passent en mode évaluation normale dès que les premiers articles arrivent (~2-5 min après démarrage complet).

Pour vérifier si des données arrivent :
```bash
curl http://localhost:9200/articles/_count
# Réponse attendue après démarrage : {"count":X,...} avec X > 0
```

### Relancer proprement

```bash
# Arrêter tout
docker compose down

# Redémarrer (attendre 3-5 min pour le démarrage complet)
docker compose up -d

# Suivre les logs
docker compose logs -f spark-app
docker compose logs -f rss-producer
```

---

## Stack technologique

| Composant | Version | Rôle |
|-----------|---------|------|
| Python | 3.12 | Langage principal |
| Apache Spark | 3.5.3 (local[2]) | Streaming Engine |
| Apache Kafka | 3.7 (Confluent 7.6.0) | Message Broker |
| DistilBERT | multilingual-cased | Modèle NLP de base |
| PyTorch | 2.4.0 (CPU) | Online Learning |
| ONNX Runtime | 1.19.0 | Inférence rapide INT8 |
| River | 0.21.2 | Concept Drift (ADWIN+KSWIN+PH) |
| MongoDB | 7.0 | Stockage documents |
| Elasticsearch | 8.14.0 | Recherche full-text |
| Grafana | 10.4.0 | Dashboards métriques |
| FastAPI | 0.113.0 | API REST |
| Streamlit | 1.38.0 | Dashboard interactif |

---

## Structure du projet

```
desinformation-pipeline/
├── producer/               # Producteur Kafka (RSS + GDELT)
│   ├── src/
│   │   ├── kafka_producer.py    — Collecte RSS + GDELT async
│   │   ├── rss_sources.py       — 12 sources RSS fiables
│   │   └── gdelt_client.py      — Client GDELT API
│   ├── Dockerfile
│   └── requirements.txt
├── spark-app/              # Job Spark Streaming + ML
│   ├── src/
│   │   ├── spark_streaming.py   — Pipeline principal (foreachBatch)
│   │   ├── nlp_classifier.py    — Continual-DistilBERT ONNX + PyTorch
│   │   ├── drift_monitor.py     — ADWIN + KSWIN + PageHinkley
│   │   └── online_trainer.py    — Orchestrateur apprentissage continu
│   └── Dockerfile
├── api/                    # API REST FastAPI
│   ├── src/
│   │   ├── main.py              — App FastAPI + routes
│   │   ├── routers/
│   │   │   ├── articles.py      — /articles/*
│   │   │   └── drift.py         — /drift/*
│   │   └── models/schemas.py    — Schémas Pydantic
│   └── Dockerfile
├── streamlit-dashboard/    # Interface interactive Streamlit
│   ├── app.py               — Dashboard multi-pages
│   ├── Dockerfile
│   └── requirements.txt
├── config/
│   ├── grafana/             — Dashboards + provisioning auto
│   ├── elasticsearch/       — Mapping index articles
│   ├── mongodb/             — Script init collections
│   └── kafka/               — server.properties
├── scripts/
│   ├── download_datasets.sh — Téléchargement FakeNewsNet + Fakeddit
│   ├── preprocess_data.py   — Prétraitement et split train/val/test
│   ├── train_model.py       — Fine-tuning DistilBERT
│   ├── export_onnx.py       — Export PyTorch → ONNX INT8
│   ├── generate_docs.py     — Génération DOCX (Documentation v7 + Mémoire v5)
│   └── start.sh             — Script de démarrage complet
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── models/
│   ├── pretrained/          — DistilBERT fine-tuné (FP32, ~540MB)
│   ├── onnx/                — Modèle quantifié INT8 (~130MB)
│   └── checkpoints/         — Checkpoints online learning
├── data/
│   ├── raw/                 — Datasets bruts (FakeNewsNet, Fakeddit)
│   └── processed/           — train.csv / val.csv / test.csv
├── tests/
│   └── test_pipeline.py     — Tests d'intégration
├── docker-compose.yml       — Orchestration complète (10 services)
├── .env                     — Variables d'environnement
└── README.md
```

---

## Variables d'environnement (.env)

Les paramètres clés configurables :

| Variable | Valeur par défaut | Description |
|----------|------------------|-------------|
| `SPARK_MICRO_BATCH_INTERVAL` | `5 seconds` | Fréquence de traitement Spark |
| `ONLINE_LR_BASE` | `1e-5` | Learning rate base (sans drift) |
| `ONLINE_LR_DRIFT` | `5e-5` | Learning rate en mode drift |
| `RESERVOIR_BUFFER_SIZE` | `5000` | Taille du buffer reservoir sampling |
| `ADWIN_DELTA` | `0.002` | Sensibilité ADWIN |
| `DRIFT_COMPOSITE_THRESHOLD` | `0.5` | Seuil de déclenchement drift |
| `GRAFANA_ADMIN_PASSWORD` | `admin2025` | Mot de passe Grafana |
