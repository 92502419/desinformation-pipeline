# api/src/routers/articles.py — Router Articles
from fastapi import APIRouter, Query, HTTPException
from pymongo import MongoClient
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta, timezone
import os

router = APIRouter(prefix='/api/v1/articles', tags=['articles'])

mongo = MongoClient(os.getenv('MONGO_URI', 'mongodb://mongodb:27017'))
db    = mongo[os.getenv('MONGO_DB', 'disinformation_db')]
es    = Elasticsearch(os.getenv('ES_HOST', 'http://elasticsearch:9200'))


@router.get('/recent')
def get_recent(limit: int = Query(50, le=200), fake_only: bool = False):
    """Derniers articles classifiés (tri par date de traitement décroissant)"""
    query = {'is_fake': 1} if fake_only else {}
    arts  = list(db.articles.find(query, {'_id': 0}).sort('processed_at', -1).limit(limit))
    return {'articles': arts, 'count': len(arts)}


@router.get('/search')
def search(q: str = Query(..., min_length=2, description='Terme(s) de recherche'), limit: int = Query(20, le=100)):
    """Recherche full-text via Elasticsearch (champs title×3 et body)"""
    try:
        res  = es.search(
            index='articles',
            query={'multi_match': {'query': q, 'fields': ['title^3', 'body']}},
            size=limit,
        )
        hits = [h['_source'] for h in res['hits']['hits']]
        return {'results': hits, 'total': res['hits']['total']['value']}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f'Elasticsearch indisponible : {e}')


@router.get('/virality')
def virality(hours: int = Query(24, le=168, description='Fenêtre temporelle en heures')):
    """Évolution horaire du taux de désinformation sur les N dernières heures"""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    pipeline = [
        {'$match': {'processed_at': {'$gte': cutoff}}},
        {'$group': {
            '_id':             {'$substr': ['$processed_at', 0, 13]},
            'total':           {'$sum': 1},
            'fakes':           {'$sum': '$is_fake'},
            'avg_confidence':  {'$avg': '$confidence'},
        }},
        {'$sort': {'_id': 1}},
    ]
    data = list(db.articles.aggregate(pipeline))
    return {'trend': data, 'hours': hours}
