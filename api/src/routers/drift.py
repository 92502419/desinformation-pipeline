# api/src/routers/drift.py — Router Drift Events + Injection de simulation
from fastapi import APIRouter, Query, BackgroundTasks
from pymongo import MongoClient
import os, sys, importlib

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


def _run_injection(scenario: str):
    """Tâche de fond : charge et exécute le script d'injection."""
    scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        import inject_drift_simulation as inj
        inj.run_scenario(
            scenario,
            broker=os.getenv('KAFKA_BROKER', 'kafka:29092'),
            topic=os.getenv('KAFKA_TOPIC_RAW', 'raw-news-stream'),
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f'Injection échouée : {e}')


@router.post('/inject')
def inject_drift(
    background_tasks: BackgroundTasks,
    scenario: str = Query('B', regex='^[ABCDabcd]$',
                          description='Scénario : A=abrupt, B=graduel, C=cyclique, D=incrémental'),
):
    """
    Déclenche une simulation de Concept Drift en arrière-plan.
    Résultats visibles dans Grafana et Streamlit en < 5 minutes.
    """
    background_tasks.add_task(_run_injection, scenario.upper())
    return {
        'status':   'started',
        'scenario': scenario.upper(),
        'message':  f'Injection du scénario {scenario.upper()} lancée en arrière-plan.',
        'monitor':  {
            'grafana':   'http://localhost:3000',
            'streamlit': 'http://localhost:8501',
            'api':       '/api/v1/drift/events',
        },
    }
