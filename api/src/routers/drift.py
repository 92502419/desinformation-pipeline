# api/src/routers/drift.py — Router Drift Events
from fastapi import APIRouter, Query
from pymongo import MongoClient
import os

router = APIRouter(prefix='/api/v1/drift', tags=['drift'])

mongo = MongoClient(os.getenv('MONGO_URI', 'mongodb://mongodb:27017'))
db    = mongo[os.getenv('MONGO_DB', 'disinformation_db')]


@router.get('/events')
def get_drift_events(limit: int = Query(20, le=100)):
    """Historique des événements de Concept Drift (tri chronologique décroissant)"""
    events = list(db.drift_events.find({}, {'_id': 0}).sort('timestamp', -1).limit(limit))
    return {'events': events, 'count': len(events)}


@router.get('/stats')
def get_drift_stats():
    """Statistiques agrégées sur les événements de drift"""
    total      = db.drift_events.count_documents({})
    confirmed  = db.drift_events.count_documents({'drift_confirmed': True})
    last_event = db.drift_events.find_one({}, {'_id': 0}, sort=[('timestamp', -1)])
    return {
        'total_events':     total,
        'confirmed_events': confirmed,
        'confirmation_rate': round(confirmed / total * 100, 2) if total > 0 else 0,
        'last_event':        last_event,
    }
