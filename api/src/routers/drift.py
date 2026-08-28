# api/src/routers/drift.py — Router Drift Events + Injection de simulation
from fastapi import APIRouter, Query, BackgroundTasks
from pymongo import MongoClient
import os, sys, logging

log = logging.getLogger(__name__)
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


def _get_inj_module():
    scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import inject_drift_simulation as inj
    return inj


def _run_injection(scenario: str, with_recovery: bool = True, visualization_window: int = 300):
    """Tâche de fond : charge et exécute le script d'injection + récupération.

    Le drift reste observable pendant `visualization_window` secondes avant
    que la récupération automatique ne ramène le pipeline à la normale.
    """
    try:
        inj = _get_inj_module()
        inj.run_scenario(
            scenario,
            broker=os.getenv('KAFKA_BROKER', 'kafka:29092'),
            topic=os.getenv('KAFKA_TOPIC_RAW', 'raw-news-stream'),
            with_recovery=with_recovery,
            visualization_window=visualization_window,
        )
    except Exception as e:
        log.error(f'Injection échouée : {e}')


def _run_recovery_only():
    """Tâche de fond : envoi d'articles réels pour rééquilibrer le modèle."""
    try:
        from confluent_kafka import Producer
        import inject_drift_simulation as inj
        producer = Producer({
            'bootstrap.servers': os.getenv('KAFKA_BROKER', 'kafka:29092'),
            'client.id': 'drift-recovery-api',
        })
        inj.run_recovery(producer, n_articles=100)
    except Exception as e:
        log.error(f'Récupération échouée : {e}')


@router.post('/inject')
def inject_drift(
    background_tasks: BackgroundTasks,
    scenario: str = Query('B', regex='^[ABCDabcd]$',
                          description='Scénario : A=abrupt, B=graduel, C=cyclique, D=incrémental'),
    with_recovery: bool = Query(True,
                                description='Envoyer des articles réels après le drift pour rééquilibrer le modèle'),
    visualization_window: int = Query(300, ge=60, le=600,
                                description='Secondes pendant lesquelles le drift reste observable '
                                            'avant la récupération automatique (60-600s, défaut 300s = 5 min)'),
):
    """
    Déclenche une simulation de Concept Drift en arrière-plan.
    Le drift reste observable pendant `visualization_window` secondes (5-10 min),
    puis la récupération automatique (with_recovery=True par défaut) envoie des
    articles réels pour que le modèle et les statistiques reviennent à leur état
    normal — le drift n'est jamais un état permanent.
    Résultats visibles dans Grafana et Streamlit dès l'injection.
    """
    background_tasks.add_task(_run_injection, scenario.upper(), with_recovery, visualization_window)
    return {
        'status':               'started',
        'scenario':             scenario.upper(),
        'with_recovery':        with_recovery,
        'visualization_window': visualization_window,
        'message':       (
            f'Injection du scénario {scenario.upper()} lancée en arrière-plan. '
            + (f'Drift observable ~{visualization_window//60} min, puis récupération '
               'automatique — retour à la normale sans intervention.'
               if with_recovery else 'Récupération désactivée : le drift restera actif jusqu\'à récupération manuelle.')
        ),
        'monitor': {
            'grafana':   'http://localhost:3000',
            'streamlit': 'http://localhost:8501',
            'api':       '/api/v1/drift/events',
        },
    }


@router.post('/recover')
def recover_from_drift(background_tasks: BackgroundTasks):
    """
    Lance manuellement la phase de récupération post-drift.
    Envoie 100 articles réels fiables dans Kafka pour rééquilibrer le modèle
    et le faire revenir à son comportement normal après une simulation de drift.
    """
    background_tasks.add_task(_run_recovery_only)
    return {
        'status':  'started',
        'message': 'Récupération lancée — 100 articles réels envoyés dans le flux. '
                   'Le modèle reviendra à la normale dans les prochaines minutes.',
    }
