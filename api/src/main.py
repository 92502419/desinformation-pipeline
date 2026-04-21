# api/src/main.py — API REST FastAPI — Pipeline Désinformation
# KOMOSSI Sosso — Master 2 IBDIA, UCAO-UUT 2025-2026
import os, sys
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# ── Ajout du répertoire src/ au sys.path pour les imports relatifs ──
sys.path.insert(0, os.path.dirname(__file__))
from routers.articles import router as articles_router
from routers.drift    import router as drift_router

load_dotenv()

app = FastAPI(
    title='Disinformation Monitor API',
    description=(
        'API temps réel pour le monitoring de la désinformation — '
        'KOMOSSI Sosso, Master 2 IBDIA, UCAO-UUT 2025-2026'
    ),
    version='2.0.0',
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
app.include_router(articles_router)   # /api/v1/articles/*
app.include_router(drift_router)      # /api/v1/drift/*

# ── Connexions globales (santé + stats) ─────────────────────────────
mongo = MongoClient(os.getenv('MONGO_URI', 'mongodb://mongodb:27017'))
db    = mongo[os.getenv('MONGO_DB', 'disinformation_db')]
es    = Elasticsearch(os.getenv('ES_HOST', 'http://elasticsearch:9200'))


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
    """Statistiques globales en temps réel (MongoDB)"""
    total   = db.articles.count_documents({})
    n_fake  = db.articles.count_documents({'is_fake': 1})
    n_real  = db.articles.count_documents({'is_fake': 0})
    n_drift = db.drift_events.count_documents({})
    cutoff  = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    last_1h = db.articles.count_documents({'processed_at': {'$gte': cutoff}})
    return {
        'total_articles':    total,
        'fake_articles':     n_fake,
        'real_articles':     n_real,
        'fake_rate':         round(n_fake / total * 100, 2) if total > 0 else 0,
        'drift_events':      n_drift,
        'articles_last_hour': last_1h,
        'timestamp':         datetime.now(timezone.utc).isoformat(),
    }
