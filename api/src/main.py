# api/src/main.py — API REST FastAPI — Pipeline Désinformation
# KOMOSSI Sosso — Master BIG DATA IA, Institut ESI — UCAO-UUT, 2025-2026
import os, sys
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# ── Ajout du répertoire src/ au sys.path pour les imports relatifs ──
sys.path.insert(0, os.path.dirname(__file__))
from routers.articles   import router as articles_router
from routers.drift      import router as drift_router
from routers.web_search import router as web_search_router
from calibration        import ProbabilityCalibrator
import es_sync

load_dotenv()

FAKE_RATE_WINDOW = int(os.getenv('FAKE_RATE_WINDOW', 100))
SIMULATION_SOURCES = ['SimulationDrift', 'SimulationRecovery']
REAL_FLOW = {'source': {'$nin': SIMULATION_SOURCES}}

app = FastAPI(
    title='Disinformation Monitor API',
    description=(
        'API temps réel pour le monitoring de la désinformation — '
        'KOMOSSI Sosso, Master BIG DATA IA, Institut ESI — UCAO-UUT, 2025-2026.\n\n'
        '**Nouveau v2.0** : `/api/v1/search/web` — recherche internet + '
        'classification fake/réel en temps réel via le pipeline ML.\n\n'
        '**Nouveau v2.2** : `/api/v1/model/calibration` — température, seuils et '
        'métriques de la calibration probabiliste (prédiction sélective, verdict '
        '`uncertain`).\n\n'
        '**Nouveau v2.5** : statistiques sur trois horizons (100 derniers, 24 h, cumul du '
        'flux réel), récupération de drift par articles réels, `/api/v1/sync` '
        '(réconciliation MongoDB → Elasticsearch), rechargement automatique du modèle ONNX.'
    ),
    version='2.5.0',
    docs_url='/docs',
    redoc_url='/redoc',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── Inclusion des routers ────────────────────────────────────────────
app.include_router(articles_router)    # /api/v1/articles/*
app.include_router(drift_router)       # /api/v1/drift/*
app.include_router(web_search_router)  # /api/v1/search/web

# ── Connexions globales (santé + stats) ─────────────────────────────
mongo = MongoClient(os.getenv('MONGO_URI', 'mongodb://mongodb:27017'))
db    = mongo[os.getenv('MONGO_DB', 'disinformation_db')]
es    = Elasticsearch(os.getenv('ES_HOST', 'http://elasticsearch:9200'))


# ── Synchronisation continue MongoDB → Elasticsearch (Streamlit ≡ Grafana) ──
es_sync.start_background_sync()


@app.get('/api/v1/sync/status', tags=['monitoring'])
def sync_status():
    """Dernier bilan de la réconciliation MongoDB → Elasticsearch (thread de fond)."""
    return es_sync.last_run or {'status': 'pending', 'message': 'première réconciliation en cours'}


@app.post('/api/v1/sync/run', tags=['monitoring'])
def sync_run():
    """Lance immédiatement une réconciliation MongoDB → Elasticsearch."""
    r = es_sync.reconcile()
    es_sync.last_run.clear(); es_sync.last_run.update(r)
    return r


# ── ENDPOINT : santé des services ───────────────────────────────────
@app.get('/health', tags=['monitoring'])
def health():
    """Vérifie la disponibilité de MongoDB et Elasticsearch"""
    try:
        mongo.admin.command('ping')
        mongo_ok = True
    except Exception:
        mongo_ok = False
    try:
        es_ok = es.ping()
    except Exception:
        es_ok = False
    return {
        'status':        'ok' if (mongo_ok and es_ok) else 'degraded',
        'mongo':         'up' if mongo_ok else 'down',
        'elasticsearch': 'up' if es_ok  else 'down',
        'timestamp':     datetime.now(timezone.utc).isoformat(),
    }


# ── ENDPOINT : statistiques globales ────────────────────────────────
@app.get('/api/v1/stats', tags=['stats'])
def get_stats():
    """Statistiques globales en temps réel (MongoDB).

    Dégrade proprement (comme /health) si MongoDB est indisponible, plutôt que
    de renvoyer une erreur 500 brute — un test d'intégration (tests/test_api_health.py)
    l'a révélé : cet endpoint plantait sans grâce quand les services Docker ne
    sont pas démarrés.
    """
    try:
        # Statistiques GLOBALES calculées sur le FLUX RÉEL uniquement (RSS, GDELT,
        # recherche web) : les articles de simulation (SimulationDrift /
        # SimulationRecovery, ~20 000 sur ~24 000 en octobre 2026) écrasaient le
        # cumul — taux figé à 64 % alors que le flux réel est à ~13 %, et chaque
        # nouvel article RSS n'avait quasiment aucun effet visible. Ils restent en
        # base (non destructif) et sont comptés à part (simulation_articles).
        total   = db.articles.count_documents(REAL_FLOW)
        n_fake  = db.articles.count_documents({**REAL_FLOW, 'is_fake': 1})
        n_real  = total - n_fake
        n_sim   = db.articles.count_documents({'source': {'$in': SIMULATION_SOURCES}})
        n_drift = db.drift_events.count_documents({})
        now     = datetime.now(timezone.utc)
        cutoff  = (now - timedelta(hours=1)).isoformat()
        last_1h = db.articles.count_documents({'processed_at': {'$gte': cutoff}})
        cutoff_24h = (now - timedelta(hours=24)).isoformat()
        total_24h  = db.articles.count_documents({**REAL_FLOW, 'processed_at': {'$gte': cutoff_24h}})
        fake_24h   = db.articles.count_documents({**REAL_FLOW, 'is_fake': 1,
                                                  'processed_at': {'$gte': cutoff_24h}})
        # Taux de fake COURANT : fenêtre glissante sur les N derniers articles traités,
        # simulation INCLUSE (c'est lui qui montre le drift puis la récupération).
        # FAKE_RATE_WINDOW = 100 = taille d'une récupération : après récupération,
        # la fenêtre ne contient plus que des articles réels.
        recent = list(db.articles.find({}, {'_id': 0, 'is_fake': 1})
                      .sort('processed_at', -1).limit(FAKE_RATE_WINDOW))
        n_recent_fake = sum(1 for d in recent if d.get('is_fake') == 1)
    except Exception:
        return {
            'total_articles': 0, 'fake_articles': 0, 'real_articles': 0,
            'fake_rate': 0, 'drift_events': 0, 'articles_last_hour': 0,
            'status': 'degraded', 'error': 'MongoDB indisponible',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
    pct = lambda a, b: round(a / b * 100, 2) if b > 0 else 0
    fake_rate = pct(n_recent_fake, len(recent))
    return {
        'total_articles':      total,
        'fake_articles':       n_fake,
        'real_articles':       n_real,
        'simulation_articles': n_sim,
        'fake_rate':           fake_rate,
        'real_rate':           round(100 - fake_rate, 2) if recent else 0,
        'fake_rate_window':    len(recent),
        'fake_rate_global':    pct(n_fake, total),
        'real_rate_global':    pct(n_real, total),
        'articles_24h':        total_24h,
        'fake_rate_24h':       pct(fake_24h, total_24h),
        'real_rate_24h':       pct(total_24h - fake_24h, total_24h),
        'drift_events':        n_drift,
        'articles_last_hour':  last_1h,
        'status':              'ok',
        'timestamp':           now.isoformat(),
    }


# ── ENDPOINT : calibration probabiliste du modèle ───────────────────
@app.get('/api/v1/model/calibration', tags=['stats'])
def get_calibration():
    """État de la calibration (température, seuils, ECE/Brier avant/après).

    Charge models/calibration.json produit par scripts/calibrate_model.py.
    En son absence, retourne l'état de repli (T=1, seuil 0,75, non calibré) —
    même comportement que le classifieur en production.
    """
    return ProbabilityCalibrator.load().describe()
