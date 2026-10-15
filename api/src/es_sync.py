# api/src/es_sync.py — Synchronisation continue MongoDB → Elasticsearch
#
# Streamlit lit MongoDB (via cette API) et Grafana lit Elasticsearch : un article
# présent dans l'un et absent de l'autre crée un écart entre les deux tableaux de
# bord. Spark écrit normalement dans les deux, mais une écriture Elasticsearch peut
# échouer (index indisponible, document rejeté — cas constaté le 2026-10-14 : 104
# articles de recherche web à date RFC 2822 refusés) sans que MongoDB soit affecté.
#
# MongoDB est la base de référence : un thread de fond compare périodiquement les
# identifiants des deux bases et indexe dans Elasticsearch tout article manquant,
# quelle que soit sa source. Intervalle : ES_SYNC_INTERVAL_SECONDS (défaut 300 s,
# 0 = désactivé). Également déclenchable à la demande : POST /api/v1/sync/run.

import logging
import os
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pymongo import MongoClient

log = logging.getLogger('es_sync')

ES_FIELDS = ('id', 'title', 'body', 'url', 'source', 'source_category', 'language', 'timestamp',
             'processed_at', 'is_fake', 'verdict', 'confidence', 'p_fake', 'p_fake_raw',
             'gdelt_tone', 'drift_score', 'drift_active')

last_run: dict = {}


def iso(value, default):
    """Date en ISO 8601 (accepte ISO et RFC 2822), sinon `default`."""
    if not value:
        return default
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).isoformat()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError):
        return default


def _clients():
    db = MongoClient(os.getenv('MONGO_URI', 'mongodb://mongodb:27017'),
                     serverSelectionTimeoutMS=5000)[os.getenv('MONGO_DB', 'disinformation_db')]
    es = Elasticsearch(os.getenv('ES_HOST', 'http://elasticsearch:9200'), request_timeout=60)
    return db, es


def _missing_ids(es, index: str, ids: list) -> list:
    missing = []
    for i in range(0, len(ids), 1000):
        chunk = ids[i:i + 1000]
        docs = es.mget(index=index, ids=chunk, source=False)['docs']
        missing += [d['_id'] for d in docs if not d.get('found')]
    return missing


def reconcile(dry_run: bool = False) -> dict:
    """Indexe dans Elasticsearch les articles MongoDB absents. Retourne un bilan."""
    db, es = _clients()
    ids = [d['id'] for d in db.articles.find({}, {'_id': 0, 'id': 1}) if d.get('id')]
    missing = _missing_ids(es, 'articles', ids) if es.indices.exists(index='articles') else ids
    report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'mongo_articles': len(ids),
              'missing_in_es': len(missing), 'indexed': 0, 'errors': 0}
    if missing and not dry_run:
        actions = []
        for doc in db.articles.find({'id': {'$in': missing}}, {'_id': 0}):
            ts = iso(doc.get('timestamp'), doc.get('processed_at'))
            if ts != doc.get('timestamp'):
                db.articles.update_one({'id': doc['id']}, {'$set': {'timestamp': ts}})
            doc['timestamp'] = ts
            actions.append({'_index': 'articles', '_id': doc['id'],
                            '_source': {k: doc.get(k) for k in ES_FIELDS if k in doc}})
        ok, errors = bulk(es, actions, raise_on_error=False, refresh=True)
        report.update(indexed=ok, errors=len(errors))
        if errors:
            log.warning('es_sync : %d erreurs, 1re : %s', len(errors), errors[0])
    report['es_articles'] = es.count(index='articles')['count']
    report['in_sync'] = report['es_articles'] == report['mongo_articles']
    return report


def _loop(interval: int):
    time.sleep(30)   # laisse MongoDB/Elasticsearch finir de démarrer
    while True:
        try:
            r = reconcile()
            last_run.clear(); last_run.update(r)
            if r['missing_in_es']:
                log.warning('es_sync : %d articles absents d\'Elasticsearch -> %d indexés',
                            r['missing_in_es'], r['indexed'])
        except Exception as e:
            last_run.clear(); last_run.update({'error': str(e), 'timestamp': datetime.now(timezone.utc).isoformat()})
            log.warning('es_sync indisponible : %s', e)
        time.sleep(interval)


def start_background_sync():
    interval = int(os.getenv('ES_SYNC_INTERVAL_SECONDS', 300))
    if interval > 0:
        threading.Thread(target=_loop, args=(interval,), name='es-sync', daemon=True).start()
        log.info('es_sync : réconciliation MongoDB -> Elasticsearch toutes les %d s', interval)
